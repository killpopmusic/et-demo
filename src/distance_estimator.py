import math

class DistanceEstimator:
    def __init__(self, focal_length=900, real_face_width=14.0):
        self.focal_length = focal_length
        self.real_face_width = real_face_width

    def calculate_pixel_distance(self, landmark1, landmark2, image_width, image_height):
        x1, y1 = int(landmark1.x * image_width), int(landmark1.y * image_height)
        x2, y2 = int(landmark2.x * image_width), int(landmark2.y * image_height)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def estimate_distance(self, pixel_face_width):
        return (
            (self.real_face_width * self.focal_length) / pixel_face_width
            if pixel_face_width
            else None
        )
