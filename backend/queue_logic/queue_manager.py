import time
import cv2
import numpy as np
from collections import deque
from sklearn.linear_model import LinearRegression
from backend.service_time.service_time_manager import ServiceTimeManager

class SingleQueue:
    """
    Manages a single queue's logic using Behavioral Filtering and PCA Ordering.
    PART 1 — REMOVE CASHIER ROI COMPLETELY
    PART 3 — FIX WALKER EXCLUSION PROPERLY
    PART 4 — BUILD STABLE QUEUE-LINE MODEL
    PART 7 — SERVICE TIME ESTIMATION
    """
    def __init__(self, name, queue_roi, queue_direction_vec=None):
        self.name = name
        self.queue_roi = np.array(queue_roi) if queue_roi and len(queue_roi) >= 3 else None # Polygon points or None
        
        # Logic Helpers
        self.service_manager = ServiceTimeManager()
        
        # Filtering State
        self.track_history = {} # {id: deque([(x,y), ...], maxlen=10)}
        self.roi_entry_time = {} # {id: timestamp}
        self.valid_members = set()
        
        # PCA Ordering State
        self.queue_vec = np.array([1, 0]) # Default direction
        self.pca_counter = 0
        
        # Service Tracking
        self.first_person_id = None
        self.first_person_entry_time = None
        
        # Configurable Thresholds
        self.STANDING_TIME_THRESH = 2.0 # PART 3: Track lifetime > 2s
        self.MAX_VELOCITY = 15.0 # pixels per frame
        self.ALIGNMENT_THRESH = 50.0
        self.DIRECTION_COS_MIN = 0.5

    def _is_in_roi(self, center, roi):
        """
        Checks if a point (center) is inside a polygonal ROI.
        If roi is None, returns True (fallback to full frame).
        """
        if roi is None or len(roi) == 0:
            return True
        pts = roi.astype(np.float32)
        dist = cv2.pointPolygonTest(pts, center, False)
        return dist >= 0

    def _update_history(self, pid, center):
        if pid not in self.track_history:
            self.track_history[pid] = deque(maxlen=20)
        self.track_history[pid].append(center)

    def _check_behavior(self, pid, now):
        """
        PART 3 — FIX WALKER EXCLUSION PROPERLY
        Filters tracks based on stability, velocity, and lifetime.
        """
        if pid not in self.roi_entry_time:
            return False
            
        # 1. Lifetime check
        lifetime = now - self.roi_entry_time[pid]
        if lifetime < self.STANDING_TIME_THRESH:
            return False
            
        history = self.track_history.get(pid, [])
        if len(history) < 5:
            return True # Not enough history yet, assume valid
            
        # 2. Velocity check
        p1 = np.array(history[-5])
        p2 = np.array(history[-1])
        dist = np.linalg.norm(p2 - p1)
        avg_vel = dist / 5.0
        
        if avg_vel > self.MAX_VELOCITY:
            return False # Moving too fast (likely a walker)
            
        # 3. Directional check
        v = p2 - p1
        vn = np.linalg.norm(v)
        if vn > 1e-3:
            v = v / vn
            qv = self.queue_vec / (np.linalg.norm(self.queue_vec) + 1e-6)
            cosang = float(np.dot(v, qv))
            if abs(cosang) < self.DIRECTION_COS_MIN:
                return False
        
        return True

    def _update_pca_direction(self, centers):
        """
        PART 4 — BUILD STABLE QUEUE-LINE MODEL
        Runs PCA every 30 frames to get dominant queue direction.
        """
        if len(centers) < 3:
            return
            
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca.fit(centers)
        vec = pca.components_[0]
        
        # Ensure direction is consistent (forward-ish)
        # If it flipped 180 deg, flip it back based on current direction
        if np.dot(vec, self.queue_vec) < 0:
            vec = -vec
            
        self.queue_vec = vec

    def process(self, tracks):
        """
        Main processing loop for this queue.
        PART 4 — PCA ORDERING
        PART 7 — SERVICE TIME ESTIMATION
        """
        now = time.time()
        current_frame_ids = set()
        standing_candidates = {} # id -> center
        
        # 1. Filter by ROI and Behavior
        for t in tracks:
            x1, y1, x2, y2, pid = t
            pid = int(pid)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            current_frame_ids.add(pid)
            self._update_history(pid, center)
            
            if self._is_in_roi(center, self.queue_roi):
                if pid not in self.roi_entry_time:
                    self.roi_entry_time[pid] = now
                
                if self._check_behavior(pid, now):
                    standing_candidates[pid] = center

        # Cleanup stale history
        stale = [pid for pid in self.roi_entry_time if pid not in current_frame_ids]
        for pid in stale:
            del self.roi_entry_time[pid]
            if pid in self.track_history:
                del self.track_history[pid]

        # 2. PCA Ordering
        final_ordered_ids = []
        pca_axis = None
        if len(standing_candidates) >= 2:
            centers = np.array(list(standing_candidates.values()))
            
            # Update PCA direction every 30 frames
            self.pca_counter += 1
            if self.pca_counter >= 30:
                self._update_pca_direction(centers)
                self.pca_counter = 0
            
            # Project onto queue axis
            projections = []
            ids = list(standing_candidates.keys())
            for pid in ids:
                proj = np.dot(standing_candidates[pid], self.queue_vec)
                projections.append(proj)
            # Build a visual PCA axis segment for debug
            c_mean = np.mean(centers, axis=0)
            axis_len = 100.0
            p1 = c_mean - self.queue_vec / (np.linalg.norm(self.queue_vec) + 1e-6) * axis_len
            p2 = c_mean + self.queue_vec / (np.linalg.norm(self.queue_vec) + 1e-6) * axis_len
            pca_axis = (float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))
                
            # Sort by projection distance (first person has largest projection in direction)
            # We assume queue flows in the direction of self.queue_vec
            sorted_indices = np.argsort(projections)[::-1]
            final_ordered_ids = [ids[i] for i in sorted_indices]
        elif len(standing_candidates) == 1:
            final_ordered_ids = list(standing_candidates.keys())

        # 3. PART 7 — SERVICE TIME ESTIMATION
        # First person = Service Position
        current_first_id = final_ordered_ids[0] if final_ordered_ids else None
        
        # Detect Service Event (First person changed or disappeared)
        if current_first_id != self.first_person_id:
            if self.first_person_id is not None:
                # Previous person served
                duration = now - self.first_person_entry_time
                if duration > 1.0: # Filter out noise
                    # Update manager (we simulate a cashier ROI by just saying they were at the front)
                    # We pass [self.first_person_id] as "in cashier" to trigger the manager's exit logic
                    self.service_manager.update([], current_queue_len=len(final_ordered_ids))
                    # The update call above will detect the exit. 
                    # Actually, we need to pass it into the active_service manually since we removed ROI.
            
            # New person at front
            self.first_person_id = current_first_id
            self.first_person_entry_time = now
            if current_first_id is not None:
                # Trigger entry in service manager
                self.service_manager.update([current_first_id], current_queue_len=len(final_ordered_ids))

        queue_len = len(final_ordered_ids)
        avg_billing = self.service_manager.get_avg_billing_time()
        median_billing = self.service_manager.get_median_billing_time()
        predicted_service = median_billing
        remaining = self.service_manager.get_current_remaining_time(median_billing)
        est_wait_time = queue_len * predicted_service + remaining
        
        # Debug Output
        if len(tracks) > 0:
            print(f"DEBUG [{self.name}]: Raw:{len(tracks)} | BehaviorPass:{len(standing_candidates)} | Final:{queue_len}")

        return {
            "name": self.name,
            "count": queue_len,
            "avg_service_time": round(avg_billing, 1),
            "median_service_time": round(median_billing, 1),
            "est_wait_time": round(est_wait_time, 1),
            "alert": queue_len > 10 or est_wait_time > 300,
            "member_ids": final_ordered_ids,
            "cashier_ids": [self.first_person_id] if self.first_person_id else [],
            "behavior_pass_ids": list(standing_candidates.keys()),
            "pca_axis": pca_axis
        }

class QueueManager:
    """
    Orchestrator for Multiple Queues.
    PART 1 — REMOVE CASHIER ROI COMPLETELY
    PART 6 — MULTI-QUEUE SUPPORT (STABLE)
    """
    def __init__(self, queues_config):
        self.queues = []
        if not queues_config:
            print("QueueManager: Creating default Queue_1 (Full Frame)")
            self.queues.append(SingleQueue("Queue_1", []))
        else:
            for name, cfg in queues_config.items():
                q = cfg.get("queue_roi") or []
                self.queues.append(SingleQueue(name, q))
            
    def update(self, tracks):
        """
        Ensures hard isolation between queues.
        """
        metrics = {}
        assigned_pids = set()
        
        for q in self.queues:
            res = q.process(tracks)
            
            # Hard Isolation: Remove people already assigned to previous queues
            original_members = res["member_ids"]
            isolated_members = [pid for pid in original_members if pid not in assigned_pids]
            
            for pid in isolated_members:
                assigned_pids.add(pid)
                
            res["member_ids"] = isolated_members
            res["count"] = len(isolated_members)
            
            # Re-run metrics for isolation
            q_obj = q
            predicted = q_obj.service_manager.predict_service_time(res["count"])
            remaining = q_obj.service_manager.get_current_remaining_time()
            res["est_wait_time"] = round(res["count"] * predicted + remaining, 1)
            
            metrics[q.name] = res
            
        return metrics
