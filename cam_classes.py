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
webcam_width = 640
webcam_height = 480

# Define the specific classes we care about
TARGET_CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
    "train", "truck", "boat", "traffic light"
]
TARGET_CLASSES_MAP = {name: i for i, name in enumerate(CLASSES)}
TARGET_CLASS_IDS = [TARGET_CLASSES_MAP[name] for name in TARGET_CLASS_NAMES]

# Class toggles and colors
CLASS_TOGGLES = {cls: True for cls in TARGET_CLASS_NAMES}
np.random.seed(0)  # for consistent colors
CLASS_COLORS = {cls: (int(c[0]), int(c[1]), int(c[2])) for cls, c in zip(TARGET_CLASS_NAMES, np.random.randint(0, 255, size=(len(TARGET_CLASS_NAMES), 3)))}

# Mouse callback function to handle button clicks
def mouse_callback(event, x, y, flags, param):
    global current_mode, modes, button_width, button_height, button_gap, start_y, monitor_height, CLASS_TOGGLES
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check for mode buttons at the top
        for mode_id in range(4):
            start_x = 10 + (button_width + button_gap) * mode_id
            end_x = start_x + button_width
            end_y = start_y + button_height
            if start_x < x < end_x and start_y < y < end_y:
                current_mode = mode_id
                print(f"Mode changed to: {modes[current_mode]}")
                return
        
        # Check for the exit button at the bottom left
        exit_button_x = 10
        exit_button_y = monitor_height - button_height - 10
        if exit_button_x < x < exit_button_x + button_width and exit_button_y < y < exit_button_y + button_height:
            current_mode = 4
            print("Exit button clicked.")

        # Check for class toggle clicks
        class_list_x_start = 10
        class_list_y_start = 60
        for i, cls in enumerate(TARGET_CLASS_NAMES):
            y_start = class_list_y_start + i * 25
            y_end = y_start + 20
            # Checkbox area
            if class_list_x_start < x < class_list_x_start + 20 and y_start < y < y_end:
                CLASS_TOGGLES[cls] = not CLASS_TOGGLES[cls]
                print(f"Toggled class '{cls}': {'Enabled' if CLASS_TOGGLES[cls] else 'Disabled'}")
                return



class SimpleTracker:
    def __init__(self, max_age=30):
        """
        Initializes the tracker with a maximum age for tracks.
        max_age: The number of frames a track can be 'missed' before it is deleted.
        """
        self.tracks = {}  # {id: {'positions': [(x, y), ...], 'age': int, 'missed_frames': int}}
        self.next_id = 0
        self.max_age = max_age

    def update(self, boxes):
        """
        Updates the tracker with new bounding boxes.
        """
        # Step 1: Predict the next position for existing tracks
        predicted_tracks = {}
        for tid, track_data in self.tracks.items():
            positions = track_data['positions']
            if len(positions) > 1:
                # Simple linear prediction based on last two positions
                last_pos = np.array(positions[-1])
                prev_pos = np.array(positions[-2])
                velocity = last_pos - prev_pos
                predicted_pos = tuple((last_pos + velocity).astype(int))
            elif len(positions) == 1:
                # No velocity, just use last position
                predicted_pos = positions[-1]
            else:
                continue
            
            predicted_tracks[tid] = {
                'pos': predicted_pos,
                'data': track_data
            }

        # Step 2: Associate new detections with existing tracks
        matched_track_ids = set()
        matched_box_indices = set()
        new_tracks = {}
        
        for i, box in enumerate(boxes):
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            
            best_match_id = None
            min_dist = float('inf')

            for tid, pred_data in predicted_tracks.items():
                if tid in matched_track_ids:
                    continue

                pred_pos = pred_data['pos']
                dist = np.hypot(cx - pred_pos[0], cy - pred_pos[1])
                
                if dist < 75 and dist < min_dist: # Increase the distance threshold for prediction
                    min_dist = dist
                    best_match_id = tid
            
            if best_match_id is not None:
                # Match found: update the track
                positions = self.tracks[best_match_id]['positions']
                self.tracks[best_match_id]['positions'] = positions + [(cx, cy)]
                self.tracks[best_match_id]['missed_frames'] = 0
                matched_track_ids.add(best_match_id)
                matched_box_indices.add(i)
                new_tracks[best_match_id] = self.tracks[best_match_id]

        # Step 3: Handle unmatched tracks and new detections
        # Update missed frames for unmatched tracks
        for tid, track_data in self.tracks.items():
            if tid not in matched_track_ids:
                track_data['missed_frames'] += 1
                if track_data['missed_frames'] < self.max_age:
                    new_tracks[tid] = track_data
        
        # Add new tracks for unmatched boxes
        for i, box in enumerate(boxes):
            if i not in matched_box_indices:
                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                new_tracks[self.next_id] = {'positions': [(cx, cy)], 'missed_frames': 0}
                self.next_id += 1
        
        self.tracks = new_tracks

    def draw_tracks(self, img, mode, original_dims, working_dims):
        for tid, track_data in self.tracks.items():
            positions = track_data['positions']
            if positions:
                transformed_positions = self._transform_positions(positions, mode, original_dims, working_dims)
                if transformed_positions:
                    for i in range(1, len(transformed_positions)):
                        cv2.line(img, transformed_positions[i - 1], transformed_positions[i], (0, 255, 0), 2)
                    cv2.circle(img, transformed_positions[-1], 5, (0, 0, 255), -1)
                    cv2.putText(img, f'ID:{tid}', transformed_positions[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def _transform_positions(self, positions, mode, original_dims, working_dims):
        transformed = []
        orig_h, orig_w = original_dims
        
        if mode == 0:  # Full
            for x, y in positions:
                transformed.append((x, y))
        elif mode == 1:  # Top
            for x, y in positions:
                if y < orig_h // 2:
                    transformed.append((x, y))
        elif mode == 2:  # Bottom
            for x, y in positions:
                if y >= orig_h // 2:
                    transformed.append((x, y - orig_h // 2))
        elif mode == 3:  # Side-by-side
            for x, y in positions:
                if y < orig_h // 2:  # Top half (left side)
                    transformed.append((x, y))
                elif y >= orig_h // 2:  # Bottom half (right side)
                    transformed.append((x + orig_w, y - orig_h // 2))
        return transformed

def draw_buttons(frame):
    global current_mode, modes, button_width, button_height, button_gap, start_y, monitor_height
    for mode_id in range(4):
        mode_name = modes[mode_id]
        start_x = 10 + (button_width + button_gap) * mode_id
        end_x = start_x + button_width
        end_y = start_y + button_height
        color = (0, 255, 0) if mode_id == current_mode else (100, 100, 100)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, -1)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
        text_size = cv2.getTextSize(mode_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = start_x + (button_width - text_size[0]) // 2
        text_y = start_y + (button_height + text_size[1]) // 2
        cv2.putText(frame, mode_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    exit_button_name = modes[4]
    start_x_exit = 10
    start_y_exit = frame.shape[0] - button_height - 10
    end_x_exit = start_x_exit + button_width
    end_y_exit = start_y_exit + button_height
    color_exit = (0, 0, 255) if current_mode == 4 else (50, 50, 200)
    cv2.rectangle(frame, (start_x_exit, start_y_exit), (end_x_exit, end_y_exit), color_exit, -1)
    cv2.rectangle(frame, (start_x_exit, start_y_exit), (end_x_exit, end_y_exit), (255, 255, 255), 2)
    text_size_exit = cv2.getTextSize(exit_button_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x_exit = start_x_exit + (button_width - text_size_exit[0]) // 2
    text_y_exit = start_y_exit + (button_height + text_size_exit[1]) // 2
    cv2.putText(frame, exit_button_name, (text_x_exit, text_y_exit), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw_class_toggles(frame):
    global CLASS_TOGGLES, CLASS_COLORS
    class_list_x_start = 10
    class_list_y_start = 60
    for i, cls in enumerate(TARGET_CLASS_NAMES):
        y_start = class_list_y_start + i * 25
        y_end = y_start + 20
        # Draw checkbox
        checkbox_color = (0, 255, 0) if CLASS_TOGGLES[cls] else (0, 0, 255)
        cv2.rectangle(frame, (class_list_x_start, y_start), (class_list_x_start + 20, y_end), checkbox_color, -1)
        cv2.rectangle(frame, (class_list_x_start, y_start), (class_list_x_start + 20, y_end), (255, 255, 255), 2)
        
        # Draw class name and color
        text_x = class_list_x_start + 30
        text_y = y_start + 15
        cv2.putText(frame, cls, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[cls], 1, cv2.LINE_AA)

def draw_objects(frame, boxes, scores, classes):
    global CLASS_TOGGLES, CLASS_COLORS
    for box, score, cls_id in zip(boxes, scores, classes):
        cls_name = CLASSES[cls_id]
        if CLASS_TOGGLES.get(cls_name, False):  # Only draw if the class is enabled
            x1, y1, x2, y2 = box
            color = CLASS_COLORS[cls_name]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{cls_name} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def run_detection(frame, model, platform):
    frame_height, frame_width = frame.shape[:2]
    inference_frame = cv2.resize(frame, IMG_SIZE)
    img_rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
    
    if platform in ['pytorch', 'onnx']:
        input_data = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.
    else:
        input_data = img_rgb
    
    outputs = model.run([input_data])
    boxes, classes, scores = post_process_yolov10(outputs)
    
    filtered_boxes, filtered_classes, filtered_scores = [], [], []

    if boxes is not None and len(boxes) > 0:
        scale_x = frame_width / IMG_SIZE[0]
        scale_y = frame_height / IMG_SIZE[1]
        
        for i in range(len(boxes)):
            cls_id = classes[i]
            if cls_id in TARGET_CLASS_IDS:
                b = boxes[i]
                filtered_boxes.append([int(b[0] * scale_x), int(b[1] * scale_y), int(b[2] * scale_x), int(b[3] * scale_y)])
                filtered_classes.append(cls_id)
                filtered_scores.append(scores[i])

    return filtered_boxes, filtered_classes, filtered_scores

def prepare_display_frame(frame, current_mode):
    frame_height, frame_width = frame.shape[:2]
    if current_mode == 0:
        working_frame = frame.copy()
    elif current_mode == 1:
        working_frame = frame[0:frame_height // 2, 0:frame_width]
    elif current_mode == 2:
        working_frame = frame[frame_height // 2:frame_height, 0:frame_width]
    elif current_mode == 3:
        top_half = frame[0:frame_height // 2, 0:frame_width]
        bottom_half = frame[frame_height // 2:frame_height, 0:frame_width]
        working_frame = np.concatenate((top_half, bottom_half), axis=1)
    else:
        working_frame = None
    return working_frame

def render_frame(working_frame, tracker, last_detected_boxes, last_detected_classes, last_detected_scores, current_mode, original_dims):
    working_frame_dims = working_frame.shape[:2]

    # 1. Transform the source-of-truth bounding boxes for the display frame
    transformed_boxes = transform_boxes(last_detected_boxes, current_mode, original_dims, working_frame_dims)
    
    # 2. Draw the transformed bounding boxes
    if len(transformed_boxes) > 0:
        draw_objects(working_frame, transformed_boxes, last_detected_scores, last_detected_classes)
    
    # 3. Draw the tracker's history
    tracker.draw_tracks(working_frame, current_mode, original_dims, working_frame_dims)

    # 4. Final display resize and UI
    final_display_frame = cv2.resize(working_frame, (monitor_width, monitor_height))
    draw_buttons(final_display_frame)
    draw_class_toggles(final_display_frame)
    
    return final_display_frame

def transform_boxes(boxes, mode, original_dims, working_frame_dims):
    if not boxes:
        return []

    orig_h, orig_w = original_dims
    transformed_boxes = []
    if mode == 0:
        transformed_boxes = boxes
    elif mode == 1:
        transformed_boxes = boxes
    elif mode == 2:
        transformed_boxes = [[box[0], box[1] - orig_h // 2, box[2], box[3] - orig_h // 2] for box in boxes ]
    elif mode == 3:
        transformed_boxes = boxes
        # transformed_boxes_top = [box for box in boxes]
        # transformed_boxes_bottom = [[box[0] + orig_w, box[1] - orig_h // 2, box[2] + orig_w, box[3] - orig_h // 2] for box in boxes ]
        # transformed_boxes = transformed_boxes_top + transformed_boxes_bottom
    
    return transformed_boxes

def main():
    global current_mode, monitor_width, monitor_height, webcam_width, webcam_height
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./yolov10.rknn')
    parser.add_argument('--target', type=str, default='rk3576')
    parser.add_argument('--device_id', type=str, default=None)
    parser.add_argument('--camera_id', default=45)
    args = parser.parse_args()

    model, platform = setup_model(args)
    # cap = cv2.VideoCapture('ppvideo.mp4')
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    window_name = 'YOLOv10 Tracking'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)

    tracker = SimpleTracker()
    last_mode = -1
    frame_counter = 0
    detection_interval = 2
    last_detected_boxes, last_detected_classes, last_detected_scores = [], [], []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        original_dims = frame.shape[:2]

        if current_mode != last_mode:
            tracker.tracks.clear()
            last_detected_boxes, last_detected_classes, last_detected_scores = [], [], []
            last_mode = current_mode
        
        if frame_counter % detection_interval == 0:
            last_detected_boxes, last_detected_classes, last_detected_scores = run_detection(frame, model, platform)
            tracker.update(last_detected_boxes)

        working_frame = prepare_display_frame(frame, current_mode)
        if working_frame is None:
            break
        
        final_display_frame = render_frame(working_frame, tracker, last_detected_boxes, last_detected_classes, last_detected_scores, current_mode, original_dims)
        cv2.imshow(window_name, final_display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or current_mode == 4:
            break

    cap.release()
    cv2.destroyAllWindows()
    model.release()

if __name__ == '__main__':
    main()