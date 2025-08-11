import cv2
import numpy as np
import argparse
from yolov10 import setup_model, post_process_yolov10, IMG_SIZE, CLASSES

# --- Global variables for UI and mode selection ---
current_mode = 0
modes = {
    0: 'Full',
    1: 'Top',
    2: 'Bottom',
    3: 'Side',
    4: 'Exit'
}
button_width = 70
button_height = 30
button_gap = 10
start_y = 10
monitor_width = 1920
monitor_height = 720
webcam_width = 1280
webcam_height = 720

# --- Class Definitions and Colors ---
TARGET_CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
    "train", "truck", "boat", "traffic light"
]
TARGET_CLASSES_MAP = {name: i for i, name in enumerate(CLASSES)}
TARGET_CLASS_IDS = [TARGET_CLASSES_MAP[name] for name in TARGET_CLASS_NAMES]
CLASS_TOGGLES = {cls: True for cls in TARGET_CLASS_NAMES}
np.random.seed(0)
CLASS_COLORS = {cls: (int(c[0]), int(c[1]), int(c[2])) for cls, c in zip(TARGET_CLASS_NAMES, np.random.randint(0, 255, size=(len(TARGET_CLASS_NAMES), 3)))}

# --- Mouse Callback (Unchanged) ---
def mouse_callback(event, x, y, flags, param):
    global current_mode, CLASS_TOGGLES
    if event == cv2.EVENT_LBUTTONDOWN:
        # Mode buttons
        for mode_id in range(4):
            start_x = 10 + (button_width + button_gap) * mode_id
            if start_x < x < start_x + button_width and start_y < y < start_y + button_height:
                current_mode = mode_id
                print(f"Mode changed to: {modes[current_mode]}")
                return
        # Exit button
        exit_button_x, exit_button_y = 10, monitor_height - button_height - 10
        if exit_button_x < x < exit_button_x + button_width and exit_button_y < y < exit_button_y + button_height:
            current_mode = 4
            print("Exit button clicked.")
        # Class toggles
        y_start_cls = 60
        for i, cls in enumerate(TARGET_CLASS_NAMES):
            y_cls = y_start_cls + i * 25
            if 10 < x < 30 and y_cls < y < y_cls + 20:
                CLASS_TOGGLES[cls] = not CLASS_TOGGLES[cls]
                print(f"Toggled class '{cls}': {'Enabled' if CLASS_TOGGLES[cls] else 'Disabled'}")
                return

# --- SIMPLIFIED: Simple Tracker ---
# This tracker works directly with the coordinates of the current canvas.
class SimpleTracker:
    def __init__(self, max_age=30):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age

    def update(self, boxes):
        detected_centers = [((box[0] + box[2]) // 2, (box[1] + box[3]) // 2) for box in boxes]
        unmatched_detections = list(range(len(detected_centers)))
        matched_track_ids = set()

        for tid, track in self.tracks.items():
            if not track['positions']: continue
            last_pos = track['positions'][-1]
            best_match_idx, min_dist = -1, 75  # Distance threshold
            for i in unmatched_detections:
                dist = np.hypot(detected_centers[i][0] - last_pos[0], detected_centers[i][1] - last_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = i
            
            if best_match_idx != -1:
                track['positions'].append(detected_centers[best_match_idx])
                track['missed_frames'] = 0
                matched_track_ids.add(tid)
                unmatched_detections.remove(best_match_idx)

        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                self.tracks[tid]['missed_frames'] += 1
                if self.tracks[tid]['missed_frames'] >= self.max_age:
                    del self.tracks[tid]
        
        for i in unmatched_detections:
            self.tracks[self.next_id] = {'positions': [detected_centers[i]], 'missed_frames': 0}
            self.next_id += 1

    def draw_tracks(self, img):
        for tid, track_data in self.tracks.items():
            positions = track_data['positions']
            if len(positions) > 1:
                cv2.polylines(img, [np.array(positions)], isClosed=False, color=(0, 255, 0), thickness=2)
                last_pos = positions[-1]
                cv2.circle(img, last_pos, 5, (0, 0, 255), -1)
                cv2.putText(img, f'ID:{tid}', last_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# --- UI Drawing Functions (Unchanged) ---
def draw_buttons(frame):
    for mode_id, mode_name in modes.items():
        if mode_id == 4: continue
        start_x = 10 + (button_width + button_gap) * mode_id
        color = (0, 255, 0) if mode_id == current_mode else (100, 100, 100)
        cv2.rectangle(frame, (start_x, start_y), (start_x + button_width, start_y + button_height), color, -1)
        cv2.rectangle(frame, (start_x, start_y), (start_x + button_width, start_y + button_height), (255, 255, 255), 2)
        text_size = cv2.getTextSize(mode_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = start_x + (button_width - text_size[0]) // 2
        text_y = start_y + (button_height + text_size[1]) // 2
        cv2.putText(frame, mode_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    start_x_exit, start_y_exit = 10, frame.shape[0] - button_height - 10
    color_exit = (0, 0, 255) if current_mode == 4 else (50, 50, 200)
    cv2.rectangle(frame, (start_x_exit, start_y_exit), (start_x_exit + button_width, start_y_exit + button_height), color_exit, -1)
    cv2.rectangle(frame, (start_x_exit, start_y_exit), (start_x_exit + button_width, start_y_exit + button_height), (255, 255, 255), 2)
    text_size_exit = cv2.getTextSize(modes[4], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x_exit = start_x_exit + (button_width - text_size_exit[0]) // 2
    text_y_exit = start_y_exit + (button_height + text_size_exit[1]) // 2
    cv2.putText(frame, modes[4], (text_x_exit, text_y_exit), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw_class_toggles(frame):
    y_start = 60
    for i, cls in enumerate(TARGET_CLASS_NAMES):
        y_pos = y_start + i * 25
        checkbox_color = (0, 255, 0) if CLASS_TOGGLES[cls] else (0, 0, 255)
        cv2.rectangle(frame, (10, y_pos), (30, y_pos + 20), checkbox_color, -1)
        cv2.rectangle(frame, (10, y_pos), (30, y_pos + 20), (255, 255, 255), 2)
        cv2.putText(frame, cls, (40, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[cls], 1, cv2.LINE_AA)

# --- Core Logic Functions ---
def draw_objects(frame, boxes, scores, classes, offset_x=0, offset_y=0):
    for box, score, cls_id in zip(boxes, scores, classes):
        cls_name = CLASSES[cls_id]
        if CLASS_TOGGLES.get(cls_name, False):
            x1, y1, x2, y2 = box
            color = CLASS_COLORS[cls_name]
            cv2.rectangle(frame, (x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y), color, 2)
            cv2.putText(frame, f'{cls_name} {score:.2f}', (x1 + offset_x, y1 + offset_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def pad_and_resize(img, target_size=(640, 640)):
    """Pads an image to a square shape and resizes it to the target size."""
    h, w = img.shape[:2]
    max_dim = max(h, w)
    
    # Calculate padding
    pad_h = max_dim - h
    pad_w = max_dim - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # Pad the image
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    
    # Resize to target size
    resized_img = cv2.resize(padded_img, target_size, interpolation=cv2.INTER_LINEAR)
    
    return resized_img, (left, top, w, h)

def run_detection(frame, model, platform):
    """
    Pads the frame to a square, runs detection, and then scales the bounding boxes
    back to the original frame's coordinate system.
    """
    if frame is None or frame.size == 0:
        return [], [], []

    original_h, original_w = frame.shape[:2]

    # Step 1: Pad and resize the image for inference
    inference_img, padding_info = pad_and_resize(frame, IMG_SIZE)
    pad_left, pad_top, original_w_in_padded, original_h_in_padded = padding_info

    img_rgb = cv2.cvtColor(inference_img, cv2.COLOR_BGR2RGB)
    input_data = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255. if platform in ['pytorch', 'onnx'] else img_rgb
    
    outputs = model.run([input_data])
    boxes, classes, scores = post_process_yolov10(outputs)
    
    filtered_boxes, filtered_classes, filtered_scores = [], [], []
    if boxes is not None and len(boxes) > 0:
        # Step 2: Scale and offset the bounding boxes back to the original frame's coordinates
        scale_x = original_w / original_w_in_padded
        scale_y = original_h / original_h_in_padded
        
        for i in range(len(boxes)):
            if classes[i] in TARGET_CLASS_IDS:
                b = boxes[i]
                
                # Scale the box from the 640x640 padded image to the padded image's dimensions
                x1_scaled = b[0] * (original_w_in_padded / IMG_SIZE[0])
                y1_scaled = b[1] * (original_h_in_padded / IMG_SIZE[1])
                x2_scaled = b[2] * (original_w_in_padded / IMG_SIZE[0])
                y2_scaled = b[3] * (original_h_in_padded / IMG_SIZE[1])
                
                # Remove the padding offset to get coordinates in the original frame's space
                x1_final = x1_scaled - pad_left
                y1_final = y1_scaled - pad_top
                x2_final = x2_scaled - pad_left
                y2_final = y2_scaled - pad_top
                
                # Ensure coordinates are within the original frame boundaries
                x1_final = max(0, int(x1_final))
                y1_final = max(0, int(y1_final))
                x2_final = min(original_w, int(x2_final))
                y2_final = min(original_h, int(y2_final))

                if x1_final < x2_final and y1_final < y2_final:
                    filtered_boxes.append([x1_final, y1_final, x2_final, y2_final])
                    filtered_classes.append(classes[i])
                    filtered_scores.append(scores[i])

    return filtered_boxes, filtered_classes, filtered_scores

def main():
    global current_mode
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./yolov10.rknn')
    parser.add_argument('--target', type=str, default='rk3576')
    parser.add_argument('--device_id', type=str, default=None)
    parser.add_argument('--camera_id',  default=45)
    args = parser.parse_args()

    model, platform = setup_model(args)
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    window_name = 'YOLOv10 Tracking - Direct Processing'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)

    tracker = SimpleTracker()
    last_mode = -1
    
    # --- Add frame counter and detection interval ---
    frame_counter = 0
    detection_interval = 5  # Run detection every 5 frames
    
    # Store results to be drawn on the display canvas
    last_boxes, last_classes, last_scores = [], [], []

    # Store the cropped frame used for the last detection
    last_cropped_frame = None

    while True:
        ret, original_frame = cap.read()
        if not ret: break
        
        frame_counter += 1

        # Reset tracker and last results if mode changes
        if current_mode != last_mode:
            tracker = SimpleTracker()
            last_boxes, last_classes, last_scores = [], [], []
            last_mode = current_mode

        h, w = original_frame.shape[:2]
        h_half = h // 2
        w_half = w // 2
        
        cropped_frame_for_detection = None
        
        # --- Prepare the frame for detection based on mode ---
        if current_mode == 0:
            # Full frame
            cropped_frame_for_detection = original_frame
        elif current_mode == 1:
            # Top half
            cropped_frame_for_detection = original_frame[0:h_half, 0:w]
        elif current_mode == 2:
            # Bottom half
            cropped_frame_for_detection = original_frame[h_half:h, 0:w]
        elif current_mode == 3:
            # Side (left half)
            cropped_frame_for_detection = original_frame[0:h, 0:w_half]
        elif current_mode == 4:
            break
        
        # --- Run Detection Periodically ---
        if frame_counter % detection_interval == 0:
            if cropped_frame_for_detection is not None:
                last_boxes, last_classes, last_scores = run_detection(cropped_frame_for_detection, model, platform)
                # Store the frame for tracking/drawing
                last_cropped_frame = cropped_frame_for_detection.copy()
            else:
                last_boxes, last_classes, last_scores = [], [], []
            
            # The tracker needs to be updated with the boxes in the cropped frame's coordinates
            tracker.update(last_boxes)

        # --- Prepare Display Canvas (every frame) ---
        display_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        offset_x, offset_y = 0, 0

        if current_mode == 0:
            display_canvas = original_frame.copy()
            # No offset for full frame
        elif current_mode == 1:
            display_canvas = original_frame[0:h_half, 0:w].copy()
            # No offset, detections are already in this frame's coord system
        elif current_mode == 2:
            display_canvas = original_frame[h_half:h, 0:w].copy()
            # No offset, detections are already in this frame's coord system
        elif current_mode == 3:
            display_canvas = original_frame[0:h, 0:w_half].copy()
            # No offset, detections are already in this frame's coord system

        # --- Draw on the display canvas using the last detection results ---
        if display_canvas is not None and last_cropped_frame is not None:
            # The tracker's positions are relative to the last cropped frame.
            # We need to adjust them if we draw on a different canvas.
            # In this updated logic, we draw directly on the cropped part of the original_frame,
            # so the coordinates are correct.
            
            # We also need to get the latest cropped frame for drawing.
            current_display_frame = None
            if current_mode == 0:
                current_display_frame = original_frame.copy()
            elif current_mode == 1:
                current_display_frame = original_frame[0:h_half, 0:w].copy()
            elif current_mode == 2:
                current_display_frame = original_frame[h_half:h, 0:w].copy()
            elif current_mode == 3:
                current_display_frame = original_frame[0:h, 0:w_half].copy()
            
            if current_display_frame is not None:
                draw_objects(current_display_frame, last_boxes, last_scores, last_classes, 0, 0)
                tracker.draw_tracks(current_display_frame)
                
                # Now, create the final display frame from the annotated cropped frame
                final_display_frame = cv2.resize(current_display_frame, (monitor_width, monitor_height))
                draw_buttons(final_display_frame)
                draw_class_toggles(final_display_frame)
                cv2.imshow(window_name, final_display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    model.release()

if __name__ == '__main__':
    main()