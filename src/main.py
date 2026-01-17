import cv2
import sys
import os
import time
import random
import json
from datetime import datetime
from screeninfo import get_monitors

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_mesh_detector import FaceMeshDetector
from distance_estimator import DistanceEstimator
from head_controller import HeadController
from message_service import MessageService
from user_interface import UserInterface
from gaze_predictor import GazePredictor

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
    
    # Show welcome screen
    welcome_screen = ui.create_welcome_screen(screen_width, screen_height)
    cv2.imshow(window_name, welcome_screen)
    cv2.waitKey(0)

    # Evaluation setup
    start_time = time.time()
    evaluation_started = False
    evaluation_data = []

    target_column = None
    last_target_change_time = 0
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        current_time = time.time()
        if current_time - last_target_change_time > 1.0:
            target_column = random.randint(0, 2)
            last_target_change_time = current_time

        results = detector.process(image)

        is_stable = False
        predicted_column = None
        
        # Check head position as in eye collector
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
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
                
                if horizontal_ok and vertical_ok and distance_ok:
                    is_stable = True
                    # Predict Gaze
                    predicted_column = predictor.predict(face_landmarks)

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

                if not is_stable:
                    image = detector.draw_landmarks(image, results)
                    MessageService.display_feedback(image, horizontal_ok, vertical_ok, distance_ok, distance)

                    height, width, _ = image.shape
                    cv2.line(image, (width // 2, 0), (width // 2, height), (0, 255, 255), 2)
                    cv2.line(image, (0, int(height / 3)), (width, int(height / 3)), (255, 255, 0), 2)

        if is_stable:
            # highlight the prediction 
            display_frame = ui.create_frame(image, highlight_column=predicted_column, target_column=target_column)

        else:
            display_frame = cv2.resize(image, (screen_width, screen_height))

        cv2.imshow(window_name, display_frame)
        
        if cv2.waitKey(5) & 0xFF == 27: 
            break
            
    cap.release()
    cv2.destroyAllWindows()

    if evaluation_data:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(results_dir, f"evaluation_{timestamp_str}.json")
        
        total_samples = len(evaluation_data)
        correct_predictions = sum(1 for d in evaluation_data if d['target'] == d['prediction'])
        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        
        duration = evaluation_data[-1]['timestamp'] - evaluation_data[0]['timestamp'] if evaluation_data else 0

        output = {
            "summary": {
                "total_samples": total_samples,
                "accuracy_percent": round(accuracy, 2),
                "duration_seconds": round(duration, 2),
                "start_time": datetime.fromtimestamp(evaluation_data[0]['timestamp']).isoformat()
            },
            "raw_data": evaluation_data
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(output, f, indent=4)
            print(f"Evaluation results saved to {filename}")
        except Exception as e:
            print(f"Failed to save results: {e}")

if __name__ == "__main__":
    main()
