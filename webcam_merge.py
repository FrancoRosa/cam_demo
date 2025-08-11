import cv2
import numpy as np
import argparse
from yolov10 import setup_model, post_process_yolov10, draw, IMG_SIZE, CLASSES

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

# Camera and Monitor Resolutions
monitor_width = 1920
monitor_height = 720
webcam_width = 320
webcam_height = 240

cam1_id = 45
cam2_id = 47

# Mouse callback function to handle button clicks
def mouse_callback(event, x, y, flags, param):
    global current_mode, modes, button_width, button_height, button_gap, start_y, monitor_height
    if event == cv2.EVENT_LBUTTONDOWN:
        for mode_id in range(4):
            start_x = 10 + (button_width + button_gap) * mode_id
            end_x = start_x + button_width
            end_y = start_y + button_height
            if start_x < x < end_x and start_y < y < end_y:
                current_mode = mode_id
                print(f"Mode changed to: {modes[current_mode]}")
                return
        exit_button_x = 10
        exit_button_y = monitor_height - button_height - 10
        if exit_button_x < x < exit_button_x + button_width and exit_button_y < y < exit_button_y + button_height:
            current_mode = 4
            print("Exit button clicked.")

class SimpleTrackerSimple:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, boxes):
        new_tracks = {}
        for box in boxes:
            cx, cy, cam_id = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2), box[4]
            assigned = False
            for tid, positions in self.tracks.items():
                if positions:
                    px, py, pcam_id = positions[-1]
                    # Only track within the same camera feed
                    if pcam_id == cam_id and np.hypot(cx - px, cy - py) < 50:
                        new_tracks[tid] = positions + [(cx, cy, cam_id)]
                        assigned = True
                        break
            if not assigned:
                new_tracks[self.next_id] = [(cx, cy, cam_id)]
                self.next_id += 1
        self.tracks = new_tracks

    def draw_tracks(self, img, mode, webcam_width):
        for tid, positions in self.tracks.items():
            if positions:
                transformed_positions = self._transform_positions(positions, mode, webcam_width)
                if transformed_positions:
                    for i in range(1, len(transformed_positions)):
                        cv2.line(img, transformed_positions[i - 1], transformed_positions[i], (0, 255, 0), 2)
                    cv2.circle(img, transformed_positions[-1], 5, (0, 0, 255), -1)
                    cv2.putText(img, f'ID:{tid}', transformed_positions[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def _transform_positions(self, positions, mode, webcam_width):
        transformed = []
        if mode == 0 or mode == 3:
            for x, y, cam_id in positions:
                if cam_id == 0:
                    transformed.append((x, y))
                elif cam_id == 1:
                    transformed.append((x + webcam_width, y))
        elif mode == 1:
            for x, y, cam_id in positions:
                if cam_id == 0:
                    transformed.append((x, y))
        elif mode == 2:
            for x, y, cam_id in positions:
                if cam_id == 1:
                    transformed.append((x, y))
        return transformed

class SimpleTracker:
    def __init__(self, max_age=100):
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

def letterbox(im, new_shape, color=(114, 114, 114)):
    # Resize and pad image to a new_shape, preserving aspect ratio
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width, height padding

    # Divide padding into 2
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def transform_boxes(boxes, mode, webcam_width):
    transformed_boxes = []
    for x1, y1, x2, y2, cam_id in boxes:
        if mode == 0 or mode == 3:
            if cam_id == 0:
                transformed_boxes.append([x1, y1, x2, y2])
            elif cam_id == 1:
                transformed_boxes.append([x1 + webcam_width, y1, x2 + webcam_width, y2])
        elif mode == 1:
            if cam_id == 0:
                transformed_boxes.append([x1, y1, x2, y2])
        elif mode == 2:
            if cam_id == 1:
                transformed_boxes.append([x1, y1, x2, y2])
    return transformed_boxes

def main():
    global current_mode, monitor_width, monitor_height, webcam_width, webcam_height
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./yolov10.rknn')
    parser.add_argument('--target', type=str, default='rk3576')
    parser.add_argument('--device_id', type=str, default=None)
    parser.add_argument('--camera_id_1', type=int, default=cam1_id)
    parser.add_argument('--camera_id_2', type=int, default=cam2_id)
    args = parser.parse_args()

    model, platform = setup_model(args)
    cap1 = cv2.VideoCapture(args.camera_id_1)
    cap2 = cv2.VideoCapture(args.camera_id_2)

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Cannot open one or both webcams")
        return

    window_name = 'YOLOv10 Tracking'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)

    tracker = SimpleTracker()
    last_mode = -1
    
    frame_counter = 0
    detection_interval = 10
    
    last_detected_boxes_combined = []
    last_detected_classes_combined = []
    last_detected_scores_combined = []
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        
        frame_counter += 1
        
        if current_mode != last_mode:
            tracker.tracks.clear()
            last_detected_boxes_combined, last_detected_classes_combined, last_detected_scores_combined = [], [], []
            last_mode = current_mode
        
        # --- ROBUST DETECTION LOGIC FOR BOTH CAMERAS ---
        if frame_counter % detection_interval == 0:
            last_detected_boxes_combined, last_detected_classes_combined, last_detected_scores_combined = [], [], []
            
            # Create a combined frame for a single inference pass
            combined_frame_original = np.concatenate((frame1, frame2), axis=1)
            
            # Letterbox the combined frame for the model
            letterboxed_frame, ratio, (dw, dh) = letterbox(combined_frame_original, IMG_SIZE)
            
            # Format input data based on the platform
            img_rgb = cv2.cvtColor(letterboxed_frame, cv2.COLOR_BGR2RGB)
            if platform in ['pytorch', 'onnx']:
                input_data = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.
            else:
                input_data = img_rgb

            # Run a single inference pass
            outputs = model.run([input_data])
            boxes, classes, scores = post_process_yolov10(outputs)

            if boxes is not None and len(boxes) > 0:
                # Adjust bounding box coordinates from letterbox space back to original space
                boxes_in_original_space = []
                for b in boxes:
                    x1, y1, x2, y2 = b
                    # Rescale from IMG_SIZE to padded size
                    x1 = (x1 - dw) / ratio
                    y1 = (y1 - dh) / ratio
                    x2 = (x2 - dw) / ratio
                    y2 = (y2 - dh) / ratio
                    boxes_in_original_space.append([int(x1), int(y1), int(x2), int(y2)])

                # Split boxes by camera and add camera ID
                for i in range(len(boxes_in_original_space)):
                    b = boxes_in_original_space[i]
                    class_id = classes[i]
                    score = scores[i]
                    x1, y1, x2, y2 = b

                    if x2 <= webcam_width: # Box is in camera 1
                        last_detected_boxes_combined.append([x1, y1, x2, y2, 0])
                        last_detected_classes_combined.append(class_id)
                        last_detected_scores_combined.append(score)
                    elif x1 >= webcam_width: # Box is in camera 2
                        last_detected_boxes_combined.append([x1 - webcam_width, y1, x2 - webcam_width, y2, 1])
                        last_detected_classes_combined.append(class_id)
                        last_detected_scores_combined.append(score)
            
            tracker.update(last_detected_boxes_combined)
        
        # --- NEW ROBUST DISPLAY LOGIC ---
        working_frame = None
        if current_mode == 0 or current_mode == 3:
            working_frame = np.concatenate((frame1, frame2), axis=1)
        elif current_mode == 1:
            working_frame = frame1.copy()
        elif current_mode == 2:
            working_frame = frame2.copy()
        else:
            break
        
        # Transform bounding boxes for the display frame
        transformed_boxes = transform_boxes(last_detected_boxes_combined, current_mode, webcam_width)
        
        # Draw the transformed bounding boxes and tracks on the working frame
        if len(transformed_boxes) > 0:
            draw(working_frame, transformed_boxes, last_detected_scores_combined, last_detected_classes_combined)
        
        tracker.draw_tracks(working_frame, current_mode, webcam_width)
        
        final_display_frame = cv2.resize(working_frame, (monitor_width, monitor_height))
        draw_buttons(final_display_frame)
        cv2.imshow(window_name, final_display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or current_mode == 4:
            break
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    model.release()

if __name__ == '__main__':
    main()