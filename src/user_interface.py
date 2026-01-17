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

    def create_welcome_screen(self, width, height):
        # Create a dark background
        info_screen = np.zeros((height, width, 3), dtype=np.uint8)
        info_screen[:] = (30, 30, 30)  # Dark gray background

        # Constants for layout
        center_x = width // 2
        start_y = height // 4
        line_height = 60
        font_scale = 1.0
        font_thickness = 2
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (240, 240, 240)

        # Title
        title = "Eye Tracking Demo"
        self._draw_centered_text(info_screen, title, center_x, start_y, font, 2.0, 3, (255, 255, 255))
        
        current_y = start_y + 120

        instructions = [
            "INSTRUCTIONS:",
            "",
            "- Please maintain a constant distance and head position.",
            "- Focus your gaze on the moving point on the screen.",
            "- The section you are looking at will be highlighted.",
            "",
            "PRIVACY NOTICE:",
            "We respect your privacy. No images or data are stored.",
            "",
            "",
            "[ Press any key to start ]"
        ]

        for line in instructions:
            scale = font_scale
            thickness = font_thickness
            line_color = color

            if "INSTRUCTIONS" in line or "PRIVACY" in line:
                scale = 1.2
                line_color = (200, 200, 255) # Light blue for headers
            elif "[" in line:
                scale = 1.1
                line_color = (100, 255, 100) # Green for action
                current_y += 20 # Extra padding

            self._draw_centered_text(info_screen, line, center_x, current_y, font, scale, thickness, line_color)
            current_y += line_height

        return info_screen

    def _draw_centered_text(self, img, text, x, y, font, scale, thickness, color):
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        cv2.putText(img, text, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)

    def create_frame(self, frame, highlight_column=None, target_column=None):
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

        if target_column is not None:
            x_center = int((target_column + 0.5) * col_width)
            y_center = height // 2
            radius = 50
            # Blue circle (BGR: 255, 0, 0)
            cv2.circle(display_frame, (x_center, y_center), radius, (255, 0, 0), -1)

        for i in range(1, num_cols):
            x = int(i * col_width)
            cv2.line(display_frame, (x, 0), (x, height), self.line_color, self.line_thickness)
            
        return display_frame
