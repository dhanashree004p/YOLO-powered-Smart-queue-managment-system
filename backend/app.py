from fastapi import FastAPI, Response, Request, UploadFile, File, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os
import cv2
import time
import json
import base64
import threading
import asyncio
import numpy as np
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ.setdefault("ULTRALYTICS_SETTINGS", "0")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

from backend.queue_logic.queue_manager import QueueManager

# ---------------------------
# CONFIGURATION AND THRESHOLDS
# ---------------------------
# Telegram Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Authentication Config
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-12345")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
# Plain text fallback for 'admin123' if bcrypt fails
ADMIN_PASSWORD_PLAIN = "admin123"

security = HTTPBearer()

def verify_password(plain_password, stored_password):
    # Simple plain text comparison as a robust fallback for environment-specific library issues
    return plain_password == stored_password

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

ALERT_QUEUE_LENGTH = 5
ROI_MODE = "manual" # Default to manual as requested: "no other roi should be there which will be set by the user only"
# WAIT_TIME_PER_PERSON removed/deprecated in favor of learned time

# Default ROI definitions
# Structured for Multiple Queues
DEFAULT_QUEUES_CONFIG = {
    "Queue_1": {
        "queue_roi": []
    }
}

DEBUG_MODE = True
INF_IMG_SZ = 640
DET_CONF = 0.4
DET_IOU = 0.5

class InferenceWorker:
    def __init__(self, source_path: str | int):
        self.source_path = source_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.tracker_yaml = "bytetrack.yaml"
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_metrics = {}
        self.history = []
        self.queues_config = dict(DEFAULT_QUEUES_CONFIG)
        self.queue_manager = None # Initialized in start or loop
        self.last_alert_time = {} # {q_name: timestamp}
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _preprocess_frame(self, frame_bgr):
        try:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            bgr = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.GaussianBlur(bgr, (3, 3), 0)
            return bgr
        except Exception:
            return frame_bgr

    def _load_model(self):
        candidates = [
            os.path.join("code", "fine-tuned_yolov8n.pt"),
            os.path.join("Source Code", "fine-tuned_yolov8n.pt"),
        ]
        weights = None
        for p in candidates:
            if os.path.isfile(p):
                weights = p
                break
        if weights is None:
            weights = "yolov8n.pt"
        model = YOLO(weights)
        model.to(self.device)
        model.classes = [0]
        return model

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.source_path)
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.running = True
        self.queue_manager = QueueManager(self.queues_config)
        # Re-create thread to allow multiple starts
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _init_default_rois(self, w, h):
        # Initialize defaults for Queue_1 if empty
        q_cfg = self.queues_config["Queue_1"]
        if not q_cfg["queue_roi"]:
            q_left = int(0.05 * w)
            q_right = int(0.75 * w)
            q_top = int(0.55 * h)
            q_bottom = int(0.95 * h)
            q_cfg["queue_roi"] = [(q_left, q_top), (q_right, q_top), (q_right, q_bottom), (q_left, q_bottom)]
        if not q_cfg["cashier_roi"]:
            c_left = int(0.78 * w)
            c_right = int(0.98 * w)
            c_top = int(0.40 * h)
            c_bottom = int(0.85 * h)
            q_cfg["cashier_roi"] = [(c_left, c_top), (c_right, c_top), (c_right, c_bottom), (c_left, c_bottom)]
        
        # Re-init manager with new ROIs
        self.queue_manager = QueueManager(self.queues_config)

    def _overlay_rois(self, frame):
        def draw_poly(poly, color):
            if not poly: return
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        
        for q_name, cfg in self.queues_config.items():
            draw_poly(cfg["queue_roi"], (255, 255, 0)) # Cyan ROI boundary

    def _annotate_ids(self, frame, xyxy, ids, q_metrics):
        """
        Debug visualization layer
        """
        if not DEBUG_MODE:
            return
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, box)
            original_pid = ids[i] if ids is not None else -1
            color = (0, 0, 255)
            status = "RAW"
            for q_name, q_m in q_metrics.items():
                if original_pid in q_m["member_ids"]:
                    color = (0, 255, 0)
                    status = "MEMBER"
                    break
                elif original_pid in q_m.get("behavior_pass_ids", []):
                    color = (0, 255, 255)
                    status = "STABLE"
                elif original_pid in q_m["cashier_ids"]:
                    color = (255, 0, 0)
                    status = "SERVICE"
                    break
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {int(original_pid)} [{status}]", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        # Draw PCA axes if available
        for q_name, q_m in q_metrics.items():
            axis = q_m.get("pca_axis")
            if axis:
                x1, y1, x2, y2 = axis
                cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2, tipLength=0.05)

    def _loop(self):
        self.last_alert_time = {} 
        print("InferenceWorker: Loop started")
        prev_t = time.time()
        fps_ema = 0.0
        while self.running and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            h, w = frame.shape[:2]
            proc = self._preprocess_frame(frame)
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            
            try:
                t0 = time.time()
                results = self.model.track(
                    rgb,
                    device=self.device,
                    classes=[0],
                    conf=DET_CONF,
                    iou=DET_IOU,
                    imgsz=INF_IMG_SZ,
                    verbose=False,
                    persist=True,
                    tracker=self.tracker_yaml
                )
                det_time = (time.time() - t0) * 1000.0
            except Exception as e:
                print(f"InferenceWorker: Track Error: {e}")
                time.sleep(0.01)
                continue

            if not results or len(results) == 0:
                time.sleep(0.01)
                continue

            result = results[0].cpu()
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy() if torch.is_tensor(boxes.xyxy) else np.array(boxes.xyxy)
            ids = boxes.id.cpu().numpy() if boxes.id is not None and torch.is_tensor(boxes.id) else (boxes.id if boxes.id is not None else None)

            tracks = []
            if ids is not None:
                for box, pid in zip(xyxy, ids):
                    tracks.append(list(box) + [pid])

            if not self.queue_manager:
                self.queue_manager = QueueManager(self.queues_config)

            # Process Queues (Stable Behavioral + PCA logic inside QueueManager)
            t1 = time.time()
            q_metrics = self.queue_manager.update(tracks)
            analytics_time = (time.time() - t1) * 1000.0
            
            # Aggregate Stats
            total_people = len(tracks)
            active_queue_length = sum(m["count"] for m in q_metrics.values())
            ewt = sum(m["est_wait_time"] for m in q_metrics.values())
            alert = any(m["alert"] for m in q_metrics.values())

            annotated = frame.copy()
            self._overlay_rois(annotated)
            self._annotate_ids(annotated, xyxy, ids, q_metrics)
            # Top-center overlay
            total_queue = sum(m["count"] for m in q_metrics.values())
            ewt_total = sum(m["est_wait_time"] for m in q_metrics.values())
            # Approx throughput from median service time
            medians = [m.get("median_service_time", m["avg_service_time"]) for m in q_metrics.values() if m.get("avg_service_time") is not None]
            throughput = 0.0
            if medians:
                ms = float(np.median(medians)) if len(medians) > 1 else float(medians[0])
                throughput = 60.0 / ms if ms > 0 else 0.0
            center_x = w // 2
            y0 = 30
            lines = [
                f"QUEUE LENGTH: {total_queue}",
                f"ESTIMATED WAIT TIME: {int(ewt_total//60)}m {int(ewt_total%60)}s",
            ]
            for i, text in enumerate(lines):
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.putText(annotated, text, (center_x - tw // 2, y0 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            # Left-side per-queue summary if debug
            if DEBUG_MODE:
                y_off = 30
                cv2.putText(annotated, f"Total: {total_people}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                y_off += 35
                for q_name, m in q_metrics.items():
                    cv2.putText(annotated, f"{q_name}: {m['count']} (EWT {m['est_wait_time']}s)", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    y_off += 30

            metrics = {
                "total_people": total_people,
                "active_queue_length": active_queue_length,
                "estimated_wait_time_sec": round(ewt, 1),
                "alert": alert,
                "queue_metrics": q_metrics,
                "timestamp_ms": int(time.time() * 1000)
            }

            with self.lock:
                self.latest_frame = annotated
                self.latest_metrics = metrics
                self.history.append(metrics)
                if len(self.history) > 500:
                    self.history = self.history[-500:]

            now = time.time()
            dt = now - prev_t
            prev_t = now
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = 0.9 * fps_ema + 0.1 * fps if fps_ema > 0 else fps
            if int(time.time()) % 2 == 0:
                print(f"Live Diag: det={det_time:.1f}ms, analytics={analytics_time:.1f}ms, fps={fps_ema:.1f}")
            time.sleep(0.01)

    def _check_and_send_alerts(self, q_metrics):
        import requests
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
            
        now = time.time()
        for q_name, m in q_metrics.items():
            if m["alert"]:
                last_time = self.last_alert_time.get(q_name, 0)
                # Only alert once every 5 minutes per queue to avoid spam
                if now - last_time > 300:
                    self.last_alert_time[q_name] = now
                    msg = (
                        f"🚨 ALERT\n"
                        f"Queue: {q_name}\n"
                        f"Current Length: {m['count']}\n"
                        f"Estimated Wait: {m['est_wait_time']}s"
                    )
                    # Send in background to not block loop
                    threading.Thread(target=self._send_telegram, args=(msg,), daemon=True).start()

    def _send_telegram(self, message):
        import requests
        try:
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                return
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            res = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
            if res.status_code != 200:
                print(f"Telegram API Error: {res.text}")
        except Exception as e:
            print(f"Telegram error: {e}")

    def set_rois(self, rois: dict):
        try:
            if not isinstance(rois, dict):
                return
            
            if self.queue_manager and self.queue_manager.queues:
                for name, cfg in rois.items():
                    q = cfg.get("queue_roi") or []
                    
                    found = False
                    for existing_q in self.queue_manager.queues:
                        if existing_q.name == name:
                            existing_q.queue_roi = np.array(q) if len(q) >= 3 else None
                            found = True
                            break
                    
                    if not found:
                        from backend.queue_logic.queue_manager import SingleQueue
                        self.queue_manager.queues.append(SingleQueue(name, q))
                self.queues_config = rois
            else:
                self.queues_config = {}
                for name, cfg in rois.items():
                    q = cfg.get("queue_roi") or []
                    self.queues_config[name] = {
                        "queue_roi": [(int(x), int(y)) for (x, y) in q] if q else []
                    }
                self.queue_manager = QueueManager(self.queues_config)
        except Exception as e:
            print(f"Error setting ROIs: {e}")
            pass

    def set_thresholds(self, alert_queue_length: int | None, wait_time_per_person: int | None):
        global ALERT_QUEUE_LENGTH
        if alert_queue_length is not None:
            ALERT_QUEUE_LENGTH = int(alert_queue_length)
        if hasattr(self, 'queue_manager') and self.queue_manager:
            for q in self.queue_manager.queues:
                q.STANDING_TIME_THRESH = 5.0
        # Sync with file_worker
        if 'file_worker' in globals():
            fw = globals()['file_worker']
            if fw:
                fw.alert_queue_length = ALERT_QUEUE_LENGTH

    def get_frame(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def get_metrics(self):
        with self.lock:
            return dict(self.latest_metrics)

    def get_history(self):
        with self.lock:
            return list(self.history)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

worker = InferenceWorker(source_path=os.path.join("source", "sample.mp4"))
worker.start()

uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp", "uploads")
os.makedirs(uploads_dir, exist_ok=True)

class FileProcessingWorker:
    """
    Background processor for pre-recorded videos.
    Streams frame-by-frame updates via an event queue for SSE.
    """
    def __init__(self, shared_model: YOLO, shared_rois: dict):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = shared_model
        self.tracker_yaml = "bytetrack.yaml"
        self.rois = shared_rois
        self.cap = None
        self.running = False
        self.paused = False
        self.frame_index = 0
        self.total_frames = 0
        self.stage = "Idle"
        self.lock = threading.Lock()
        self.event_queue = []
        self.cv = threading.Condition()
        self.thread = None
        
        # Queue Logic
        self.queue_manager = None
        self.alert_queue_length = ALERT_QUEUE_LENGTH
        self.last_alert_time = {} # {q_name: timestamp}
        self.prev_time = time.time()
        self.fps_ema = 0.0

    def _preprocess_frame(self, frame_bgr):
        try:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            bgr = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.GaussianBlur(bgr, (3, 3), 0)
            return bgr
        except Exception:
            return frame_bgr
    def _init_default_rois(self, w, h):
        # Initialize default queue if no config exists
        # We use a structure compatible with QueueManager
        if not self.rois:
            q_left = int(0.10 * w); q_right = int(0.90 * w)
            q_top = int(0.10 * h); q_bottom = int(0.90 * h)
            
            c_left = int(0.70 * w); c_right = int(0.95 * w)
            c_top = int(0.50 * h); c_bottom = int(0.90 * h)
            
            self.rois["Queue_1"] = {
                "queue_roi": [(q_left, q_top), (q_right, q_top), (q_right, q_bottom), (q_left, q_bottom)],
                "cashier_roi": [(c_left, c_top), (c_right, c_top), (c_right, c_bottom), (c_left, c_bottom)]
            }
        
        # Fallback for old structure if present
        if "queue" in self.rois and "cashier" in self.rois:
            self.rois["Queue_1"] = {
                "queue_roi": self.rois["queue"],
                "cashier_roi": self.rois["cashier"]
            }
            del self.rois["queue"]
            del self.rois["cashier"]

    def _overlay_rois(self, frame):
        def draw_poly(poly, color):
            if not poly: return
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            
        for q_name, q_cfg in self.rois.items():
            draw_poly(q_cfg.get("queue_roi"), (255, 255, 0))
            # Label the queue
            if q_cfg.get("queue_roi"):
                x, y = q_cfg["queue_roi"][0]
                cv2.putText(frame, q_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def _push_event(self, payload: dict):
        with self.cv:
            self.event_queue.append(payload)
            self.cv.notify_all()

    def start(self, video_path: str):
        with self.lock:
            if self.running:
                return
            print(f"FileWorker: Starting {video_path}")
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap or not self.cap.isOpened():
                self.running = False
                self._push_event({"type": "error", "message": f"Failed to open video: {video_path}"})
                return
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            self.frame_index = 0
            
            # Session Stats Initialization
            self.session_max_queue = 0
            self.session_sum_queue = 0
            self.session_frames_processed = 0
            self.seen_person_ids = set()
            
            # ID Normalization state
            self.id_mapping = {}  # tracker_id -> display_id
            self.next_display_id = 1
            
            self.running = True
            self.paused = False
            self.stage = "Preprocessing"
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            self._push_event({"type": "status", "stage": self.stage, "frame_index": 0, "total_frames": self.total_frames})

    def pause(self):
        with self.lock:
            self.paused = True
            self._push_event({"type": "status", "stage": "Paused"})

    def resume(self):
        with self.lock:
            self.paused = False
            self._push_event({"type": "status", "stage": self.stage})

    def stop(self):
        with self.lock:
            self.running = False
            self.paused = False
            if self.cap:
                self.cap.release()
                self.cap = None

    def _loop(self):
        try:
            self.last_alert_time = {} # Reset for new loop
            # First frame to init ROIs
            if self.cap and self.cap.isOpened():
                ok, frame = self.cap.read()
                if ok:
                    # Initialize QueueManager
                    self.queue_manager = QueueManager(self.rois)
                    # Reset cap
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while self.running and self.cap and self.cap.isOpened():
                if self.paused:
                    time.sleep(0.1)
                    continue

                ok, frame = self.cap.read()
                if not ok:
                    print("FileWorker: End of video reached")
                    break
                
                h, w = frame.shape[:2]
                self.stage = "Detection"
                proc = self._preprocess_frame(frame)
                rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
                
                try:
                    t0 = time.time()
                    results = self.model.track(
                        rgb,
                        device=self.device,
                        classes=[0],
                        conf=DET_CONF,
                        iou=DET_IOU,
                        imgsz=INF_IMG_SZ,
                        verbose=False,
                        persist=True,
                        tracker=self.tracker_yaml
                    )
                    det_time = (time.time() - t0) * 1000.0
                except Exception as e:
                    print(f"FileWorker: Track Error: {e}")
                    time.sleep(0.01)
                    continue
                
                if not results or len(results) == 0:
                    time.sleep(0.01)
                    continue
                
                result = results[0].cpu()
                boxes = result.boxes
                xyxy = boxes.xyxy.cpu().numpy() if torch.is_tensor(boxes.xyxy) else np.array(boxes.xyxy)
                ids = boxes.id.cpu().numpy() if boxes.id is not None and torch.is_tensor(boxes.id) else (boxes.id if boxes.id is not None else None)

                self.stage = "Queue Analysis"
                t1 = time.time()
                tracks = []
                if ids is not None:
                    for box, pid in zip(xyxy, ids):
                        tracks.append([box[0], box[1], box[2], box[3], pid])
                
                if not self.queue_manager:
                    self.queue_manager = QueueManager(self.rois)

                q_metrics = self.queue_manager.update(tracks)
                analytics_time = (time.time() - t1) * 1000.0
                
                total_people = len(tracks)
                active_queue_length = sum(m["count"] for m in q_metrics.values())
                ewt = sum(m["est_wait_time"] for m in q_metrics.values())
                alert = any(m["alert"] for m in q_metrics.values())
                
                self.session_max_queue = max(self.session_max_queue, active_queue_length)
                self.session_sum_queue += active_queue_length
                self.session_frames_processed += 1
                
                normalized_ids = []
                stable_ids = set()
                for q_name, qm in q_metrics.items():
                    for sid in qm.get("behavior_pass_ids", []):
                        stable_ids.add(int(sid))
                if ids is not None:
                    for pid in ids:
                        if pid is not None:
                            original_id = int(pid)
                            if original_id not in self.id_mapping:
                                self.id_mapping[original_id] = self.next_display_id
                                self.next_display_id += 1
                            display_id = self.id_mapping[original_id]
                            normalized_ids.append(display_id)
                            if original_id in stable_ids:
                                prev_sz = len(self.seen_person_ids)
                                self.seen_person_ids.add(display_id)
                                if len(self.seen_person_ids) != prev_sz:
                                    print(f"Unique visitors: {len(self.seen_person_ids)}")
                        else:
                            normalized_ids.append(None)
                
                annotated = frame.copy()
                self._overlay_rois(annotated)
                
                if DEBUG_MODE:
                    for i, box in enumerate(xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        pid = normalized_ids[i] if i < len(normalized_ids) else -1
                        original_pid = ids[i] if ids is not None else -1
                        color = (0, 0, 255)
                        status = "RAW"
                        for q_name, q_m in q_metrics.items():
                            if original_pid in q_m["member_ids"]:
                                color = (0, 255, 0)
                                status = "MEMBER"
                                break
                            elif original_pid in q_m.get("behavior_pass_ids", []):
                                color = (0, 255, 255)
                                status = "STABLE"
                            elif original_pid in q_m["cashier_ids"]:
                                color = (255, 0, 0)
                                status = "SERVICE"
                                break
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"ID {pid} [{status}]", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                    for q_name, q_m in q_metrics.items():
                        axis = q_m.get("pca_axis")
                        if axis:
                            x1a, y1a, x2a, y2a = axis
                            cv2.arrowedLine(annotated, (int(x1a), int(y1a)), (int(x2a), int(y2a)), (255, 255, 0), 2, tipLength=0.05)
                
                total_queue = active_queue_length
                ewt_total = ewt
                medians = [m.get("median_service_time", m["avg_service_time"]) for m in q_metrics.values() if m.get("avg_service_time") is not None]
                throughput = 0.0
                if medians:
                    ms = float(np.median(medians)) if len(medians) > 1 else float(medians[0])
                    throughput = 60.0 / ms if ms > 0 else 0.0
                center_x = w // 2
                y0 = 30
                lines = [
                    f"QUEUE LENGTH: {total_queue}",
                    f"ESTIMATED WAIT TIME: {int(ewt_total//60)}m {int(ewt_total%60)}s",
                    f"UNIQUE VISITORS: {len(self.seen_person_ids)}",
                    f"THROUGHPUT: {throughput:.1f} persons/min"
                ]
                for i, text in enumerate(lines):
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.putText(annotated, text, (center_x - tw // 2, y0 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                ok, buf = cv2.imencode(".jpg", annotated)
                if not ok:
                    print("FileWorker: Frame encoding failed")
                    continue
                frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                
                avg_q = 0 if self.session_frames_processed == 0 else self.session_sum_queue / self.session_frames_processed
                
                metrics = {
                    "total_people": total_people,
                    "active_queue_length": active_queue_length,
                    "estimated_wait_time_sec": round(ewt, 1),
                    "throughput_per_min": round(throughput, 2),
                    "alert": alert,
                    "queue_metrics": q_metrics,
                    "session": {
                        "max_queue_length": self.session_max_queue,
                        "avg_queue_length": round(avg_q, 1),
                        "total_unique_people": len(self.seen_person_ids)
                    },
                    "timestamp_ms": int(time.time() * 1000)
                }

                self.frame_index += 1
                now = time.time()
                dt = now - self.prev_time
                self.prev_time = now
                fps = 1.0 / dt if dt > 0 else 0.0
                self.fps_ema = 0.9 * self.fps_ema + 0.1 * fps if self.fps_ema > 0 else fps
                if self.frame_index % 30 == 0:
                    print(f"Diag: det={det_time:.1f}ms, analytics={analytics_time:.1f}ms, fps={self.fps_ema:.1f}")
                event = {
                    "type": "frame",
                    "frame_index": self.frame_index,
                    "total_frames": self.total_frames,
                    "stage": self.stage,
                    "metrics": metrics,
                    "frame_b64": frame_b64
                }
                self._push_event(event)
                time.sleep(0.01)

        except Exception as e:
            print(f"FileWorker Fatal Error: {e}")
            import traceback
            traceback.print_exc()
            self._push_event({"type": "error", "message": str(e)})
        finally:
            with self.lock:
                self.running = False
            
            # Create session summary
            avg_q = 0 if self.session_frames_processed == 0 else self.session_sum_queue / self.session_frames_processed
            self.session_summary = {
                "total_unique_people": len(self.seen_person_ids),
                "avg_queue_length": round(avg_q, 1),
                "max_queue_length": self.session_max_queue,
                "total_alerts": sum(1 for t in self.last_alert_time.values() if t > 0)
            }
            self._push_event({"type": "complete", "summary": self.session_summary})

    def set_rois(self, rois: dict):
        """Allows updating ROIs for file processing without stopping."""
        try:
            if not isinstance(rois, dict):
                return
            
            if self.queue_manager and self.queue_manager.queues:
                for name, cfg in rois.items():
                    q = cfg.get("queue_roi") or []
                    found = False
                    for existing_q in self.queue_manager.queues:
                        if existing_q.name == name:
                            existing_q.queue_roi = np.array(q) if len(q) >= 3 else None
                            found = True
                            break
                    if not found:
                        from backend.queue_logic.queue_manager import SingleQueue
                        self.queue_manager.queues.append(SingleQueue(name, q))
                self.rois = rois
            else:
                self.rois = rois
                self.queue_manager = QueueManager(self.rois)
        except Exception as e:
            print(f"FileWorker: Error setting ROIs: {e}")
            pass

file_worker = FileProcessingWorker(shared_model=worker.model, shared_rois=worker.queues_config)

@app.get("/", response_class=HTMLResponse)
def index():
    if os.path.isdir(frontend_dir):
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())
    return HTMLResponse("Frontend not found")

@app.post("/login")
async def login(request: Request):
    body = await request.json()
    username = body.get("username")
    password = body.get("password")
    
    # Use the plain text constant for reliable login in this environment
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD_PLAIN:
        token = create_access_token({"sub": username})
        return {"access_token": token, "token_type": "bearer"}
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/config")
def get_config(user: str = Depends(get_current_user)):
    return {
        "alert_queue_length": ALERT_QUEUE_LENGTH,
        "rois": worker.queues_config,
        "roi_mode": ROI_MODE
    }


@app.post("/config")
async def set_config(request: Request, user: str = Depends(get_current_user)):
    global ROI_MODE
    body = await request.json()
    
    new_mode = body.get("roi_mode")
    if new_mode in ["manual", "auto"]:
        ROI_MODE = new_mode
        
    rois = body.get("rois")
    if isinstance(rois, dict):
        worker.set_rois(rois)
        file_worker.set_rois(rois)
        
    aql = body.get("alert_queue_length")
    worker.set_thresholds(aql, None)
    return get_config(user)

def mjpeg_generator():
    boundary = b"--frame"
    while True:
        frame = worker.get_frame()
        if frame is None:
            time.sleep(0.02)
            continue
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            time.sleep(0.01)
            continue
        jpg = buf.tobytes()
        yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.03)


@app.get("/video")
def video_stream():
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace;boundary=frame")


@app.get("/metrics")
def metrics():
    return JSONResponse(worker.get_metrics())


@app.get("/analytics/history")
def analytics_history():
    return JSONResponse(worker.get_history())


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Accepts a video file (.mp4, .avi) and stores it under tmp/uploads.
    """
    name = file.filename or f"video_{int(time.time())}.mp4"
    ext = os.path.splitext(name)[1].lower()
    if ext not in [".mp4", ".avi"]:
        return JSONResponse({"error": "Unsupported format"}, status_code=400)
    save_path = os.path.join(uploads_dir, f"{int(time.time()*1000)}_{os.path.basename(name)}")
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    
    # Reset both workers' configs
    worker.queues_config = {}
    worker.queue_manager = QueueManager({})
    
    file_worker.stop()
    file_worker.rois = {}
    file_worker.queue_manager = QueueManager({})
    
    return {"path": save_path}

@app.post("/first-frame")
async def first_frame(req: Request):
    """
    Returns the first frame of a given uploaded video as base64 JPEG.
    Enables ROI drawing before starting processing.
    """
    try:
        body = await req.json()
        path = body.get("path")
        if not path or not os.path.isfile(path):
            print(f"First-frame Error: Invalid path {path}")
            return JSONResponse({"error": "Invalid path"}, status_code=400)
        
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
             print(f"First-frame Error: Could not open video {path}")
             return JSONResponse({"error": "Could not open video file"}, status_code=500)
             
        ok, frame = cap.read()
        cap.release()
        
        if not ok or frame is None:
            print(f"First-frame Error: Failed to read frame from {path}")
            return JSONResponse({"error": "Failed to read first frame"}, status_code=500)
            
        ok2, buf = cv2.imencode(".jpg", frame)
        if not ok2:
            return JSONResponse({"error": "Failed to encode frame"}, status_code=500)
            
        frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        h, w = frame.shape[:2]
        return {"frame_b64": frame_b64, "width": w, "height": h}
    except Exception as e:
        print(f"First-frame Fatal Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/start-processing")
async def start_processing(req: Request):
    """
    Starts file processing for a previously uploaded video.
    """
    body = await req.json()
    path = body.get("path")
    if not path or not os.path.isfile(path):
        return JSONResponse({"error": "Invalid path"}, status_code=400)
    
    # Stop the live stream worker to free up resources and avoid model conflicts
    worker.stop()
    
    file_worker.stop()
    file_worker.start(path)
    return {"status": "started", "path": path}


@app.post("/pause-processing")
async def pause_processing(user: str = Depends(get_current_user)):
    file_worker.pause()
    return {"status": "paused"}


@app.post("/resume-processing")
async def resume_processing(user: str = Depends(get_current_user)):
    file_worker.resume()
    return {"status": "resumed"}


@app.post("/stop-processing")
async def stop_processing_post(user: str = Depends(get_current_user)):
    file_worker.stop()
    return {"status": "stopped"}


@app.get("/events")
async def sse_events(request: Request):
    """
    Server-Sent Events stream of processing updates:
    Emits status updates and frame payloads with base64 image and metrics.
    """
    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                
                payload = None
                with file_worker.cv:
                    if file_worker.event_queue:
                        payload = file_worker.event_queue.pop(0)
                
                if payload:
                    data = json.dumps(payload)
                    yield f"data: {data}\n\n"
                    if payload.get("type") == "complete":
                        break
                else:
                    await asyncio.sleep(0.05)
        except Exception as e:
            print(f"SSE Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")

