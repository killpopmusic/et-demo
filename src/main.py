import cv2
import sys
import os
import time
import random
import json
import numpy as np
from datetime import datetime
from screeninfo import get_monitors
from screeninfo import get_monitors

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_mesh_detector import FaceMeshDetector
from distance_estimator import DistanceEstimator
from head_controller import HeadController
from message_service import MessageService
from user_interface import UserInterface
from gaze_predictor import GazePredictor

def load_gallery_images(width, height):
    images = []

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gallery_dir = os.path.join(base_dir, "gallery")
    
    if os.path.exists(gallery_dir):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        file_list = sorted([f for f in os.listdir(gallery_dir) 
                          if f.lower().endswith(valid_extensions) and not f.startswith("._")])
        
        for filename in file_list:
            filepath = os.path.join(gallery_dir, filename)
            img = cv2.imread(filepath)
            if img is not None:
                img = cv2.resize(img, (width, height))
                images.append(img)
                print(f"Loaded {filename}")

    if not images:
        print("Warning: No images found in 'gallery' folder.")

    return images

def main():
    detector = FaceMeshDetector()
    estimator = DistanceEstimator()
    head_controller = HeadController(estimator)
    ui = UserInterface()

    predictor = GazePredictor(
        model_path="src/models/gc_1x3_loso.pth", 
        scaler_path="src/scalers/scaler_1x3_loso.pkl"
    )

    try:
        monitor = get_monitors()[0]
        screen_width = monitor.width
        screen_height = monitor.height

    except Exception:
        screen_width = 1920
        screen_height = 1080

    cap = cv2.VideoCapture(0)
    
    window_name = 'Eye Tracker Demo'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    time.sleep(0.1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
    
    welcome_screen = ui.create_welcome_screen(screen_width, screen_height)
    cv2.imshow(window_name, welcome_screen)
    cv2.waitKey(0)

    # Mode Selection
    mode = 0
    selection_screen = ui.create_mode_selection_screen(screen_width, screen_height)
    while True:
        cv2.imshow(window_name, selection_screen)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('1'):
            mode = 1
            break
        elif key == ord('2'):
            mode = 2
            break
        elif key == 27: # ESC
            cap.release()
            cv2.destroyAllWindows()
            return

    if mode == 1:
        # Evaluation setup
        start_time = time.time()
        evaluation_started = False
        evaluation_data = []
        performance_data = []

        target_column = None
        last_target_change_time = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Start Latency Measurement
            t_frame_start = time.perf_counter()

            current_time = time.time()
            if current_time - last_target_change_time > 1.0:
                target_column = random.randint(0, 2)
                last_target_change_time = current_time

            # Measure Detection
            t0 = time.perf_counter()
            results = detector.process(image)
            t_detection = time.perf_counter() - t0

            is_stable = False
            predicted_column = None

            horizontal_ok, vertical_ok, distance_ok = False, False, False
            distance = 0
            
            t_validation = 0
            t_prediction = 0

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Measure Validation
                    t0 = time.perf_counter()
                    pixel_face_width = estimator.calculate_pixel_distance(
                        face_landmarks.landmark[33],
                        face_landmarks.landmark[263],
                        image.shape[1],
                        image.shape[0],
                    )

                    horizontal_ok, vertical_ok = head_controller.is_head_position_valid(
                        face_landmarks.landmark,
                        image.shape[1],
                        image.shape[0],
                    )

                    distance_ok, distance = head_controller.is_distance_valid(pixel_face_width)
                    t_validation = time.perf_counter() - t0
                    
                    if horizontal_ok and vertical_ok and distance_ok:
                        is_stable = True
                        # Predict Gaze
                        # Measure Prediction
                        t0 = time.perf_counter()
                        predicted_column = predictor.predict(face_landmarks)
                        t_prediction = time.perf_counter() - t0

                        # Evaluation logic
                        if not evaluation_started:
                            if (current_time - start_time) > 30.0:
                                evaluation_started = True
                                print("Evaluation started.")
                        
                        if evaluation_started and target_column is not None and predicted_column is not None:
                            # 500ms delay to account for reaction time
                            if (current_time - last_target_change_time) > 0.5:
                                evaluation_data.append({
                                    "timestamp": current_time,
                                    "target": target_column,
                                    "prediction": int(predicted_column)
                                })

            # Measure UI
            t0 = time.perf_counter()
            if is_stable:
                # highlight the prediction 
                display_frame = ui.create_frame(image, highlight_column=predicted_column, target_column=target_column)

            else:
                if results.multi_face_landmarks:
                    image = detector.draw_landmarks(image, results)
                MessageService.display_feedback(image, horizontal_ok, vertical_ok, distance_ok, distance)

                height, width, _ = image.shape
                cv2.line(image, (width // 2, 0), (width // 2, height), (0, 255, 255), 2)
                cv2.line(image, (0, int(height / 3)), (width, int(height / 3)), (255, 255, 0), 2)
                display_frame = cv2.resize(image, (screen_width, screen_height))

            cv2.imshow(window_name, display_frame)
            t_ui = time.perf_counter() - t0
            
            # End Latency Measurement
            t_total_latency = time.perf_counter() - t_frame_start
            
            performance_data.append({
                "timestamp": current_time,
                "detection_ms": t_detection * 1000,
                "validation_ms": t_validation * 1000,
                "prediction_ms": t_prediction * 1000,
                "ui_ms": t_ui * 1000,
                "latency_ms": t_total_latency * 1000
            })
            
            if cv2.waitKey(5) & 0xFF == 27: 
                break

        if evaluation_data or performance_data:
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(results_dir, f"experiment_{timestamp_str}.json")
            
            output = {}

            if evaluation_data:
                total_samples = len(evaluation_data)
                correct_predictions = sum(1 for d in evaluation_data if d['target'] == d['prediction'])
                accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
                duration = evaluation_data[-1]['timestamp'] - evaluation_data[0]['timestamp'] if evaluation_data else 0

                output["evaluation"] = {
                    "summary": {
                        "total_samples": total_samples,
                        "accuracy_percent": round(accuracy, 2),
                        "duration_seconds": round(duration, 2),
                        "start_time": datetime.fromtimestamp(evaluation_data[0]['timestamp']).isoformat()
                    },
                    "raw_data": evaluation_data
                }

            if performance_data:
                keys = ["detection_ms", "validation_ms", "prediction_ms", "ui_ms", "latency_ms"]
                stats = {}
                for k in keys:
                    values = [d[k] for d in performance_data if d[k] > 0]
                    if values:
                        stats[k] = {
                            "min": min(values),
                            "max": max(values),
                            "avg": sum(values) / len(values)
                        }
                output["efficiency"] = {
                    "summary": stats,
                    "raw_data": performance_data
                }

            try:
                with open(filename, 'w') as f:
                    json.dump(output, f, indent=4)
                print(f"Experiment results saved to {filename}")
            except Exception as e:
                print(f"Failed to save results: {e}")

    elif mode == 2:
        # Gallery Mode
        images_list = load_gallery_images(screen_width, screen_height)
        current_idx = 0
        dwell_time = 0.0
        last_frame_time = time.time()
        cooldown = 0.0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
               continue

            current_time = time.time()
            dt = current_time - last_frame_time
            last_frame_time = current_time
            
            if cooldown > 0:
                cooldown -= dt

            results = detector.process(image)
            is_stable = False
            predicted_column = None
            
            horizontal_ok, vertical_ok, distance_ok = False, False, False
            distance = 0

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pixel_face_width = estimator.calculate_pixel_distance(
                        face_landmarks.landmark[33], face_landmarks.landmark[263],
                        image.shape[1], image.shape[0])
                    horizontal_ok, vertical_ok = head_controller.is_head_position_valid(
                        face_landmarks.landmark, image.shape[1], image.shape[0])
                    distance_ok, distance = head_controller.is_distance_valid(pixel_face_width)
                    
                    if horizontal_ok and vertical_ok and distance_ok:
                        is_stable = True
                        predicted_column = predictor.predict(face_landmarks)
            
            hover_state = None
            progress = 0.0
            
            if is_stable and cooldown <= 0:
                if predicted_column == 0: 
                    hover_state = 'prev'
                    dwell_time += dt
                elif predicted_column == 2: 
                    hover_state = 'next'
                    dwell_time += dt
                else:
                    dwell_time = 0.0 
                
                # Check thresholds
                if dwell_time > 2.5:
                    if hover_state == 'prev':
                        current_idx = (current_idx - 1) % len(images_list)
                    elif hover_state == 'next':
                        current_idx = (current_idx + 1) % len(images_list)
                    
                    dwell_time = 0.0
                    cooldown = 1.0 
            else:
                 dwell_time = 0.0

            if dwell_time > 0:
                progress = min(dwell_time / 2.5, 1.0)

            if not is_stable:
                 if results.multi_face_landmarks:
                    image = detector.draw_landmarks(image, results)
                 MessageService.display_feedback(image, horizontal_ok, vertical_ok, distance_ok, distance)
                 
                 height, width, _ = image.shape
                 cv2.line(image, (width // 2, 0), (width // 2, height), (0, 255, 255), 2)
                 cv2.line(image, (0, int(height / 3)), (width, int(height / 3)), (255, 255, 0), 2)
                 
                 display_frame = cv2.resize(image, (screen_width, screen_height))
            else:
                 display_frame = ui.create_gallery_interface(images_list[current_idx], hover_state, progress)
            
            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(5) & 0xFF == 27: 
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
