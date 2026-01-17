import cv2
import numpy as np

class UserInterface: #3 columns 
    def __init__(self):
        self.line_color = (255, 255, 255) 
        self.line_thickness = 2

    def apply_gray_overlay(self, frame):
        gray_screen = frame.copy()
        gray_screen[:, :, :] = int(255 * 0.18) 
        return gray_screen

    def create_welcome_screen(self, width, height):
        info_screen = np.zeros((height, width, 3), dtype=np.uint8)
        info_screen[:] = (30, 30, 30)  # Dark gray background

        center_x = width // 2
        start_y = height // 4
        line_height = 60
        font_scale = 1.0
        font_thickness = 2
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (240, 240, 240)

        title = "Eye Tracking Demo"
        self._draw_centered_text(info_screen, title, center_x, start_y, font, 2.0, 3, (255, 255, 255))
        
        current_y = start_y + 120

        instructions = [
            "INSTRUCTIONS:",
            "",
            "- Please maintain a constant distance and head position.",
            "- If necessary, you will be given instructions to recover the correct position.",
            "- The section you are looking at will be highlighted.",
            "",
            "PRIVACY NOTICE:",
            "We respect your privacy. No images or data are stored.",
            "",
            "[ Press any key to select a mode ]"
        ]

        for line in instructions:
            scale = font_scale
            thickness = font_thickness
            line_color = color

            if "INSTRUCTIONS" in line or "PRIVACY" in line:
                scale = 1.2
                line_color = (200, 200, 255) 
            elif "[" in line:
                scale = 1.1
                line_color = (100, 255, 100) 
                current_y += 20 
            self._draw_centered_text(info_screen, line, center_x, current_y, font, scale, thickness, line_color)
            current_y += line_height

        return info_screen

    def _draw_centered_text(self, img, text, x, y, font, scale, thickness, color):
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        cv2.putText(img, text, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)

    def create_mode_selection_screen(self, width, height):
        screen = np.zeros((height, width, 3), dtype=np.uint8)
        screen[:] = (30, 30, 30)
        
        center_x = width // 2
        start_y = height // 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        self._draw_centered_text(screen, "CHOOSE MODE", center_x, start_y, font, 2.0, 3, (255, 255, 255))
        
        self._draw_centered_text(screen, "1. EVALUATION - follow the moving target with your eyes", center_x, start_y + 100, font, 1.0, 2, (200, 200, 200))
        self._draw_centered_text(screen, "2. EYE-CONTROLLED GALLERY - browse the pictures by directing your gaze", center_x, start_y + 160, font, 1.0, 2, (200, 200, 200))
        
        self._draw_centered_text(screen, "[ Press 1 or 2 to select ]", center_x, start_y + 250, font, 1.0, 2, (100, 255, 100))
        
        return screen

    def create_gallery_interface(self, image, hover_state=None, progress=0.0):

        display = image.copy()
        height, width, _ = display.shape
        side_width = width // 6 

        if hover_state == 'prev':
            overlay = display[:, :side_width].copy()
            cv2.rectangle(overlay, (0, 0), (side_width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, display[:, :side_width], 0.7, 0, display[:, :side_width])
        elif hover_state == 'next':
            overlay = display[:, width-side_width:].copy()
            cv2.rectangle(overlay, (0, 0), (side_width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, display[:, width-side_width:], 0.7, 0, display[:, width-side_width:])

        left_center = (side_width // 2, height // 2)
        l_color = (255, 255, 255) if hover_state == 'prev' else (150, 150, 150)
        l_thickness = 4 if hover_state == 'prev' else 2
        self._draw_chevron(display, left_center, size=30, direction='left', color=l_color, thickness=l_thickness)

        right_center = (width - side_width // 2, height // 2)
        r_color = (255, 255, 255) if hover_state == 'next' else (150, 150, 150)
        r_thickness = 4 if hover_state == 'next' else 2
        self._draw_chevron(display, right_center, size=30, direction='right', color=r_color, thickness=r_thickness)

        if hover_state and progress > 0:
            center = left_center if hover_state == 'prev' else right_center
            radius = 50

            cv2.circle(display, center, radius, (100, 100, 100), 2, cv2.LINE_AA)

            axes = (radius, radius)
            angle = 0
            startAngle = -90
            endAngle = -90 + (360 * progress)

            prog_color = (int(255 * (1-progress)), 255, int(255 * (1-progress)))
            
            cv2.ellipse(display, center, axes, angle, startAngle, endAngle, prog_color, 4, cv2.LINE_AA)

        return display

    def _draw_chevron(self, img, center, size, direction, color, thickness):
        x, y = center
        pts = []
        if direction == 'left':
            pts = np.array([
                [x + size // 2, y - size],
                [x - size // 2, y],
                [x + size // 2, y + size]
            ], np.int32)
        else:
            pts = np.array([
                [x - size // 2, y - size],
                [x + size // 2, y],
                [x - size // 2, y + size]
            ], np.int32)
            
        cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)

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
