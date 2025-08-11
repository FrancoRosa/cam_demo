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

# --- Mouse Callback ---
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
            if class_list_x_start < x < class_list_x_start + 20 and y_start < y < y_end:
                CLASS_TOGGLES[cls] = not CLASS_TOGGLES[cls]
                print(f"Toggled class '{cls}': {'Enabled' if CLASS_TOGGLES[cls] else 'Disabled'}")
                return

# --- NEW: Coordinate Mapper Class ---
# This class handles all coordinate transformations, decoupling the rendering from the detection logic.
# --- REVISED: Coordinate Mapper Class ---
# This class handles all coordinate transformations, decoupling the rendering from the detection logic.
class CoordinateMapper:
    def __init__(self, mode, original_dims):
        self.mode = mode
        self.orig_h, self.orig_w = original_dims
        self.half_h = self.orig_h // 2

    def map_point(self, x, y):
        """Maps a point from the original frame coordinates to the current display canvas coordinates."""
        if self.mode == 0:  # Full View
            return (x, y), True
        
        elif self.mode == 1:  # Top View
            if y < self.half_h:
                return (x, y), True
            return None, False

        elif self.mode == 2:  # Bottom View
            if y >= self.half_h:
                return (x, y - self.half_h), True
            return None, False

        elif self.mode == 3:  # Side-by-Side View
            if y < self.half_h: # Top half maps to left side
                return (x, y), True
            else: # Bottom half maps to right side
                return (x + self.orig_w, y - self.half_h), True
        
        return None, False

    def map_box(self, box):
        """
        Maps a bounding box from original coordinates to display coordinates,
        clipping it to the boundaries of the visible view to handle boxes that cross the midline.
        """
        x1, y1, x2, y2 = box

        # 1. Determine the visible Y-range in the original frame for the current mode
        if self.mode == 1:  # Top
            visible_y_range = (0, self.half_h)
        elif self.mode == 2:  # Bottom
            visible_y_range = (self.half_h, self.orig_h)
        else:  # Full or Side-by-side (all Y values are potentially visible)
            visible_y_range = (0, self.orig_h)
        
        # 2. Check for any overlap. If the box is entirely outside the visible range, ignore it.
        if y2 < visible_y_range[0] or y1 >= visible_y_range[1]:
            return None, False
            
        # 3. Clip the box to the visible Y-range. This is the crucial fix.
        clipped_y1 = max(y1, visible_y_range[0])
        clipped_y2 = min(y2, visible_y_range[1])
        clipped_box_original_coords = (x1, clipped_y1, x2, clipped_y2)

        # 4. Transform the corners of the *clipped* box.
        # Since the box is now clipped, its coordinates are guaranteed to be in the visible
        # region, so map_point will not return None for them.
        p1_new, _ = self.map_point(clipped_box_original_coords[0], clipped_box_original_coords[1])
        p2_new, _ = self.map_point(clipped_box_original_coords[2], clipped_box_original_coords[3])
        
        # This check provides an extra layer of safety.
        if p1_new is None or p2_new is None:
            return None, False

        # 5. Return the new, transformed box coordinates.
        return (p1_new[0], p1_new[1], p2_new[0], p2_new[1]), True


# --- REFACTORED: Simple Tracker ---
class SimpleTracker:
    def __init__(self, max_age=30):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age

    def update(self, boxes):
        """Updates the tracker with new bounding boxes in ORIGINAL frame coordinates."""
        # This update logic can be improved, but it's kept the same as the original for now.
        # The key point is that it operates purely on original frame coordinates.
        detected_centers = [((box[0] + box[2]) // 2, (box[1] + box[3]) // 2) for box in boxes]
        
        unmatched_detections = list(range(len(detected_centers)))
        matched_track_ids = set()

        # Try to match existing tracks
        for tid, track in self.tracks.items():
            if not track['positions']: continue
            last_pos = track['positions'][-1]
            
            best_match_idx, min_dist = -1, float('inf')
            for i in unmatched_detections:
                dist = np.hypot(detected_centers[i][0] - last_pos[0], detected_centers[i][1] - last_pos[1])
                if dist < 75 and dist < min_dist:
                    min_dist = dist
                    best_match_idx = i
            
            if best_match_idx != -1:
                track['positions'].append(detected_centers[best_match_idx])
                track['missed_frames'] = 0
                matched_track_ids.add(tid)
                unmatched_detections.remove(best_match_idx)

        # Handle unmatched tracks (increment missed frames or delete)
        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                self.tracks[tid]['missed_frames'] += 1
                if self.tracks[tid]['missed_frames'] >= self.max_age:
                    del self.tracks[tid]
        
        # Create new tracks for unmatched detections
        for i in unmatched_detections:
            self.tracks[self.next_id] = {'positions': [detected_centers[i]], 'missed_frames': 0}
            self.next_id += 1

    def draw_tracks(self, display_canvas, coord_mapper):
        """Draws the track history onto the display canvas using the coordinate mapper."""
        for tid, track_data in self.tracks.items():
            positions = track_data['positions']
            if len(positions) < 2:
                continue

            transformed_path = []
            for x, y in positions:
                # Map each point in the history
                new_point, is_visible = coord_mapper.map_point(x, y)
                if is_visible:
                    transformed_path.append(new_point)
                else:
                    # If a point is not visible, break the line segment
                    if len(transformed_path) > 1:
                        cv2.polylines(display_canvas, [np.array(transformed_path)], isClosed=False, color=(0, 255, 0), thickness=2)
                    transformed_path = []
            
            # Draw the last continuous segment
            if len(transformed_path) > 1:
                cv2.polylines(display_canvas, [np.array(transformed_path)], isClosed=False, color=(0, 255, 0), thickness=2)
            
            # Draw the ID at the last visible point
            if transformed_path:
                last_pos = transformed_path[-1]
                cv2.circle(display_canvas, last_pos, 5, (0, 0, 255), -1)
                cv2.putText(display_canvas, f'ID:{tid}', last_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# --- UI and Drawing Functions ---
def draw_buttons(frame):
    # This function remains unchanged
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
    # This function remains unchanged
    global CLASS_TOGGLES, CLASS_COLORS
    class_list_x_start = 10
    class_list_y_start = 60
    for i, cls in enumerate(TARGET_CLASS_NAMES):
        y_start = class_list_y_start + i * 25
        y_end = y_start + 20
        checkbox_color = (0, 255, 0) if CLASS_TOGGLES[cls] else (0, 0, 255)
        cv2.rectangle(frame, (class_list_x_start, y_start), (class_list_x_start + 20, y_end), checkbox_color, -1)
        cv2.rectangle(frame, (class_list_x_start, y_start), (class_list_x_start + 20, y_end), (255, 255, 255), 2)
        text_x = class_list_x_start + 30
        text_y = y_start + 15
        cv2.putText(frame, cls, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[cls], 1, cv2.LINE_AA)

# --- REFACTORED: Drawing Functions that use the Mapper ---
def draw_objects(display_canvas, boxes, scores, classes, coord_mapper):
    """Draws detected objects using the coordinate mapper."""
    for box, score, cls_id in zip(boxes, scores, classes):
        cls_name = CLASSES[cls_id]
        if CLASS_TOGGLES.get(cls_name, False):
            # Map the box from original coordinates to the current display's coordinates
            mapped_box, is_visible = coord_mapper.map_box(box)
            if is_visible:
                x1, y1, x2, y2 = mapped_box
                color = CLASS_COLORS[cls_name]
                cv2.rectangle(display_canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_canvas, f'{cls_name} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

# --- Core Logic ---
def run_detection(frame, model, platform):
    """Runs detection ONCE on the full, original frame."""
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
            if classes[i] in TARGET_CLASS_IDS:
                b = boxes[i]
                # Store boxes in original frame dimensions
                filtered_boxes.append([int(b[0] * scale_x), int(b[1] * scale_y), int(b[2] * scale_x), int(b[3] * scale_y)])
                filtered_classes.append(classes[i])
                filtered_scores.append(scores[i])

    return filtered_boxes, filtered_classes, filtered_scores

def prepare_display_canvas(original_frame, current_mode):
    """Prepares the background canvas for the current view mode from the original frame."""
    h, w = original_frame.shape[:2]
    if current_mode == 0:  # Full
        return original_frame.copy()
    elif current_mode == 1:  # Top
        return original_frame[0:h // 2, 0:w]
    elif current_mode == 2:  # Bottom
        return original_frame[h // 2:h, 0:w]
    elif current_mode == 3:  # Side-by-Side
        top_half = original_frame[0:h // 2, 0:w]
        bottom_half = original_frame[h // 2:h, 0:w]
        # Note: This creates a canvas of shape (h/2, 2*w)
        return np.concatenate((top_half, bottom_half), axis=1)
    else:
        return None

def main():
    global current_mode, monitor_width, monitor_height, webcam_width, webcam_height
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./yolov10.rknn')
    parser.add_argument('--target', type=str, default='rk3576')
    parser.add_argument('--device_id', type=str, default=None)
    parser.add_argument('--camera_id', default=45)
    args = parser.parse_args()

    model, platform = setup_model(args)
    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    window_name = 'YOLOv10 Tracking - Refactored'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)

    tracker = SimpleTracker()
    frame_counter = 0
    detection_interval = 12 # Detect every other frame
    
    # "Source of Truth" storage
    last_detected_boxes, last_detected_classes, last_detected_scores = [], [], []
    
    while True:
        ret, original_frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        original_dims = original_frame.shape[:2]

        # --- DETECTION & TRACKING PIPELINE (operates on original_frame) ---
        if frame_counter % detection_interval == 0:
            boxes, classes, scores = run_detection(original_frame, model, platform)
            tracker.update(boxes)
            # Update the "source of truth"
            last_detected_boxes = boxes
            last_detected_classes = classes
            last_detected_scores = scores
        
        # --- DISPLAY PIPELINE (operates on a canvas derived from original_frame) ---
        
        # 1. Prepare the blank canvas based on the current mode
        display_canvas = prepare_display_canvas(original_frame, current_mode)
        if display_canvas is None: # Happens if mode is 'Exit'
            break
        
        # 2. Create a mapper for the current frame and mode
        coord_mapper = CoordinateMapper(current_mode, original_dims)
        
        # 3. Draw objects and tracks onto the canvas using the mapper
        draw_objects(display_canvas, last_detected_boxes, last_detected_scores, last_detected_classes, coord_mapper)
        tracker.draw_tracks(display_canvas, coord_mapper)
        
        # 4. Resize for monitor and draw UI elements
        final_display_frame = cv2.resize(display_canvas, (monitor_width, monitor_height))
        draw_buttons(final_display_frame)
        draw_class_toggles(final_display_frame)
        
        # 5. Show the final result
        cv2.imshow(window_name, final_display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or current_mode == 4:
            break

    cap.release()
    cv2.destroyAllWindows()
    model.release()

if __name__ == '__main__':
    main()