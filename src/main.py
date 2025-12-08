import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_mesh_detector import FaceMeshDetector
from distance_estimator import DistanceEstimator
from head_controller import HeadController
from message_service import MessageService

def main():
    detector = FaceMeshDetector()
    estimator = DistanceEstimator()
    head_controller = HeadController(estimator)

    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        
        results = detector.process(image)
        image = detector.draw_landmarks(image, results)
        
        # Check head position
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

                MessageService.display_feedback(image, horizontal_ok, vertical_ok, distance_ok, distance)

                height, width, _ = image.shape
                cv2.line(image, (width // 2, 0), (width // 2, height), (0, 255, 255), 2)
                cv2.line(image, (0, int(height / 3)), (width, int(height / 3)), (255, 255, 0), 2)

        cv2.imshow('MediaPipe Face Mesh & Iris', image)
        
        if cv2.waitKey(5) & 0xFF == 27: 
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
