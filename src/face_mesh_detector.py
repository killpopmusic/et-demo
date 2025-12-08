import mediapipe as mp
import cv2

class FaceMeshDetector:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, image):

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        image.flags.writeable = True
        return results

    def draw_landmarks(self, image, results):

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Face mesh
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                
                # Contours
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                
                # Irises
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        return image
