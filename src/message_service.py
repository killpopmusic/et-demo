import cv2
from typing import Tuple

class MessageService:
    @staticmethod
    def display_feedback(
        frame, horizontal_ok, vertical_ok, distance_ok, distance
    ):
        height, width, _ = frame.shape

        if not (horizontal_ok and vertical_ok and distance_ok):
            if not horizontal_ok:
                MessageService.display_message(
                    frame,
                    "Align nose horizontally",
                    (width - 500, height - 50),
                    (0, 0, 255),
                )
            if not vertical_ok:
                MessageService.display_message(
                    frame,
                    "Align eyes vertically",
                    (width - 500, height - 100),
                    (0, 0, 255),
                )
            if distance is not None:
                color = (0, 255, 0) if distance_ok else (0, 0, 255)
                MessageService.display_message(
                    frame,
                    f"Distance: {distance:.2f} cm",
                    (width - 500, height - 150),
                    color,
                )
        else:
             MessageService.display_message(
                    frame,
                    f"Position OK. Distance: {distance:.2f} cm",
                    (width - 500, height - 150),
                    (0, 255, 0),
                )

    @staticmethod
    def display_message(
        frame, message, position: Tuple[float, float], color=(0, 255, 0)
    ):
        pos_x, pos_y = int(position[0]), int(position[1])
        cv2.putText(
            frame,
            message,
            (pos_x, pos_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
