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

# monitor_width = 1920
# monitor_height = 550

# monitor_width = 1920
# monitor_height = 720
monitor_width = 1920
monitor_height = 720
webcam_width=640
webcam_height=480

# Mouse callback function to handle button clicks
def mouse_callback(event, x, y, flags, param):
    global current_mode, modes, button_width, button_height, button_gap, start_y, monitor_height
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

class SimpleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, boxes):
        # Update tracker with consistent coordinates (relative to the full original frame)
        new_tracks = {}
        for box in boxes:
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            assigned = False
            for tid, positions in self.tracks.items():
                if positions:
                    px, py = positions[-1]
                    if np.hypot(cx - px, cy - py) < 50:
                        new_tracks[tid] = positions + [(cx, cy)]
                        assigned = True
                        break
            if not assigned:
                new_tracks[self.next_id] = [(cx, cy)]
                self.next_id += 1
        self.tracks = new_tracks

    def draw_tracks(self, img, mode, original_dims, working_dims):
        # Draw tracks based on the transformed coordinates
        for tid, positions in self.tracks.items():
            if positions:
                transformed_positions = self._transform_positions(positions, mode, original_dims, working_dims)

                # ADDED CHECK: Ensure transformed_positions is not empty before drawing
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
    # Draw mode selection buttons at the top
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

    # Draw the Exit button at the bottom left
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

# This function transforms bounding boxes based on the current mode
def transform_boxes(boxes, mode, original_dims, working_frame_dims):
    if not boxes:
        return [], [], []

    orig_h, orig_w = original_dims
    work_h, work_w = working_frame_dims
    
    transformed_boxes = []

    if mode == 0: # Full
        for x1, y1, x2, y2 in boxes:
            transformed_boxes.append([x1, y1, x2, y2])
    elif mode == 1: # Top
        for x1, y1, x2, y2 in boxes:
            if y2 < orig_h // 2: # Only keep boxes in the top half
                transformed_boxes.append([x1, y1, x2, y2])
    elif mode == 2: # Bottom
        for x1, y1, x2, y2 in boxes:
            if y1 > orig_h // 2: # Only keep boxes in the bottom half
                transformed_boxes.append([x1, y1 - orig_h // 2, x2, y2 - orig_h // 2])
    elif mode == 3: # Side-by-side
        for x1, y1, x2, y2 in boxes:
            if y2 < orig_h // 2: # Top half (left side)
                transformed_boxes.append([x1, y1, x2, y2])
            elif y1 > orig_h // 2: # Bottom half (right side)
                transformed_boxes.append([x1 + orig_w, y1 - orig_h // 2, x2 + orig_w, y2 - orig_h // 2])
    
    return transformed_boxes

def main():
    global current_mode, monitor_width, monitor_height
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to model (.pt/.rknn/.onnx)', default='./yolov10.rknn')
    parser.add_argument('--target', type=str, default='rk3576')
    parser.add_argument('--device_id', type=str, default=None)
    parser.add_argument('--camera_id', type=int, default=45)
    args = parser.parse_args()

    model, platform = setup_model(args)
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
    last_detected_boxes = []
    last_detected_classes = []
    last_detected_scores = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1

        if current_mode != last_mode:
            tracker.tracks.clear()
            last_detected_boxes, last_detected_classes, last_detected_scores = [], [], []
            last_mode = current_mode
        
        frame_height, frame_width = frame.shape[:2]
        
        if frame_counter % detection_interval == 0:
            inference_frame = cv2.resize(frame, IMG_SIZE)
            img_rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
            
            if platform in ['pytorch', 'onnx']:
                input_data = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.
            else:
                input_data = img_rgb
            
            outputs = model.run([input_data])
            boxes, classes, scores = post_process_yolov10(outputs)

            if boxes is not None and len(boxes) > 0:
                scale_x = frame_width / IMG_SIZE[0]
                scale_y = frame_height / IMG_SIZE[1]
                last_detected_boxes = [[int(b[0] * scale_x), int(b[1] * scale_y), int(b[2] * scale_x), int(b[3] * scale_y)] for b in boxes]
                last_detected_classes = classes
                last_detected_scores = scores
            else:
                last_detected_boxes, last_detected_classes, last_detected_scores = [], [], []
                
            tracker.update(last_detected_boxes)
        
        # --- NEW ROBUST DISPLAY LOGIC ---
        working_frame = None
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
            break
        
        working_frame_dims = working_frame.shape[:2]
        
        # 1. Transform the source-of-truth bounding boxes for the display frame
        transformed_boxes = transform_boxes(last_detected_boxes, current_mode, (frame_height, frame_width), working_frame_dims)

        # 2. Draw the transformed bounding boxes and tracks on the working frame
        if len(transformed_boxes) > 0:
            draw(working_frame, transformed_boxes, last_detected_scores, last_detected_classes)
        
        # 3. Draw the tracker's history with the new transformation rules
        tracker.draw_tracks(working_frame, current_mode, (frame_height, frame_width), working_frame_dims)
        
        # 4. Final display
        final_display_frame = cv2.resize(working_frame, (monitor_width, monitor_height))
        draw_buttons(final_display_frame)
        cv2.imshow(window_name, final_display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or current_mode == 4:
            break


    cap.release()
    cv2.destroyAllWindows()
    model.release()

if __name__ == '__main__':
    main()