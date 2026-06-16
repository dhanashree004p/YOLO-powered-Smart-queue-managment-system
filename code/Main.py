import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN

def preprocess(image):

    # Normalize the image
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    # Apply Gaussian filter
    blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)

    return blurred_image


def open_cap(filename):

    video_path = os.path.join('source', filename)

    if not os.path.isfile(video_path):
        print("Invalid video path. Please try again.")
        return None, None

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Fallback if FPS is invalid/zero
    if not fps or fps <= 1:
        fps = 30

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join('out', filename[:-4] + 'output.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    return cap, out

def count_people_in_queue_dbscan(boxes_xywh):
    
    # boxes_xywh is expected to be in xywh format with center coordinates
    # Convert to numpy array safely
    if boxes_xywh is None:
        return 0

    if torch.is_tensor(boxes_xywh):
        arr = boxes_xywh.cpu().numpy()
    else:
        arr = np.array(boxes_xywh)

    if arr.size == 0:
        return 0

    # Extract center points (x_center, y_center)
    centers = arr[:, :2]

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=400, min_samples=2).fit(centers)
    labels = clustering.labels_

    # Count only members assigned to a cluster (label != -1)
    people = int(np.sum(labels != -1))

    return people


def calculate_wait_time(path):

    if path is None:
        print("No video provided")
        return

    # Open the video
    cap, out = open_cap(path)

    if cap is None:
        return

    print("Processing the video...")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap is not None else 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform inference
            try:
                results = model.track(frame, classes=0, conf=0.3, verbose=False, device=device)
            except Exception as e:
                print(f"Inference error on frame {frame_idx}: {e}")
                frame_idx += 1
                continue
            result = results[0].cpu()

            # Visualize the results on the frame
            annotated_frame = result.plot()

            # Get the bounding boxes
            boxes = result.boxes

            # --- New queue ordering and visualization logic ---
            # Extract centers from xywh (x_center, y_center)
            xywh = boxes.xywh
            if torch.is_tensor(xywh):
                arr_xywh = xywh.cpu().numpy()
            else:
                arr_xywh = np.array(xywh)

            if arr_xywh.size > 0:
                # xywh returned by Ultralytics uses top-left x,y and width,height.
                # Compute centers explicitly.
                centers = np.column_stack((arr_xywh[:, 0] + arr_xywh[:, 2] / 2.0,
                                           arr_xywh[:, 1] + arr_xywh[:, 3] / 2.0))

                # Perform DBSCAN clustering to filter queue members
                clustering = DBSCAN(eps=400, min_samples=2).fit(centers)
                labels = clustering.labels_

                # Indices of queue members (exclude noise label -1)
                queue_indices = [idx for idx, lbl in enumerate(labels) if lbl != -1]

                # Sort queue members left-to-right by x coordinate of center
                sorted_queue = sorted(queue_indices, key=lambda i: centers[i][0])

                # Recalculate person_count using only filtered queue members
                person_count = len(sorted_queue)

                # Plot queue position annotations for each filtered, sorted member
                if person_count > 0:
                    xyxy = boxes.xyxy
                    if torch.is_tensor(xyxy):
                        arr_xyxy = xyxy.cpu().numpy()
                    else:
                        arr_xyxy = np.array(xyxy)

                    for rank, idx in enumerate(sorted_queue, start=1):
                        x1, y1, x2, y2 = arr_xyxy[idx]
                        # Place annotation at bottom-left of the box to avoid overlap with YOLO label
                        text = f"Queue Pos: {rank}"
                        x_text = int(x1)
                        y_text = int(min(y2 - 6, annotated_frame.shape[0] - 6))

                        # Draw a small opaque background for readability
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        thickness = 2
                        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        cv2.rectangle(
                            annotated_frame,
                            (x_text, y_text - text_h - baseline),
                            (x_text + text_w, y_text + baseline),
                            (0, 0, 0),
                            -1,
                        )
                        cv2.putText(
                            annotated_frame,
                            text,
                            (x_text, y_text),
                            font,
                            font_scale,
                            (255, 255, 255),
                            thickness,
                            cv2.LINE_AA,
                        )
                else:
                    person_count = 0
            else:
                person_count = 0

            # Update top-left annotations using new person_count and EWT
            cv2.putText(annotated_frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Estimated Waiting Time: {person_count*5} seconds", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Write the frame to the output video file
            store = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(store)

            # Simple progress output every 50 frames
            frame_idx += 1
            if total_frames > 0 and frame_idx % 50 == 0:
                pct = (frame_idx / total_frames) * 100
                print(f"Processed {frame_idx}/{total_frames} frames ({pct:.1f}%).")

            # print(f"Person Count: {person_count}")
            # print(f"Estimated Waiting Time: {person_count*5} seconds")
            continue

        cap.release()
        out.release()
        print("Video processing complete.")

def cli():
    while True:
        video_path = input("Enter the path of a video file (or 'quit' or 'q' to exit): ")
        
        if video_path == 'quit' or video_path == 'q':
            print("Exiting the program...")
            break
        
        # Calculate the waiting time
        frames = calculate_wait_time(video_path)

if __name__ == "__main__":

    # Define the Model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = YOLO("code/fine-tuned_yolov8n.pt")  
    cli()