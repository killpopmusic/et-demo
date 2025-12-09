import cv2
import numpy as np

class UserInterface: #For now 3 columns 
    def __init__(self):
        self.line_color = (255, 255, 255) 
        self.line_thickness = 2

    def apply_gray_overlay(self, frame):
        gray_screen = frame.copy()
        gray_screen[:, :, :] = int(255 * 0.18) 
        return gray_screen

    def create_frame(self, frame, highlight_column=None):
        display_frame = self.apply_gray_overlay(frame)
        height, width, _ = display_frame.shape
        
        num_cols = 3
        col_width = width // num_cols

        if highlight_column is not None:
            x_start = int(highlight_column * col_width)
            x_end = int((highlight_column + 1) * col_width)

            overlay = display_frame.copy()
            cv2.rectangle(overlay, (x_start, 0), (x_end, height), (100, 100, 100), -1) 
            
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

        for i in range(1, num_cols):
            x = int(i * col_width)
            cv2.line(display_frame, (x, 0), (x, height), self.line_color, self.line_thickness)
            
        return display_frame
