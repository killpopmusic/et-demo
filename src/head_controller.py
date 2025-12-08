class HeadController:
    def __init__(
        self,
        distance_estimator,
        width_threshold=0.1,
        height_threshold=0.05,
        min_distance=50,
        max_distance=70,
    ):
        self.distance_estimator = distance_estimator
        self.width_threshold = width_threshold
        self.height_threshold = height_threshold
        self.min_distance = min_distance
        self.max_distance = max_distance

    def is_head_position_valid(self, landmarks, image_width, image_height):

        left_eye_y = int((landmarks[159].y + landmarks[145].y) / 2 * image_height)
        right_eye_y = int((landmarks[386].y + landmarks[374].y) / 2 * image_height)
        nose_x = int(landmarks[1].x * image_width)

        middle_x = image_width // 2
        # Target position for eyes is at 1/3 of the screen height
        target_y = int(image_height * 1 / 3)

        threshold_x = image_width * self.width_threshold
        threshold_y = image_height * self.height_threshold

        horizontal_ok = abs(nose_x - middle_x) <= threshold_x
        vertical_ok = (
            abs(left_eye_y - target_y) <= threshold_y
            and abs(right_eye_y - target_y) <= threshold_y
        )

        return horizontal_ok, vertical_ok

    def is_distance_valid(self, pixel_face_width):
        if not pixel_face_width:
            return False, None

        distance = self.distance_estimator.estimate_distance(pixel_face_width)
        if distance is None:
             return False, None
             
        return self.min_distance <= distance <= self.max_distance, distance
