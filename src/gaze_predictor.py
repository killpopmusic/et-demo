import torch
import joblib
import numpy as np
from gaze_classifier import GazeClassifier

class GazePredictor:
    def __init__(self, model_path, scaler_path, device='cpu'):
        self.device = torch.device(device)
        
        self.KEY_LANDMARK_INDICES = [

            # Right eye
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,

            # Right iris
            468, 469, 470, 471, 472,

            # Left eye
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,

            # Left iris
            473, 474, 475, 476, 477,

            # Right upper eyelid
            124, 113, 247, 30, 29, 27, 28, 56, 190, 189, 221, 222, 223, 224, 225,

            # Right lower eyelid 
            130, 226, 31, 228, 229, 230, 231, 232, 233, 245, 244, 112, 26, 22, 23, 24, 110, 25,243,

            #Left upper eyelid 
            413, 414, 286, 258, 257, 259, 260, 467, 342, 353, 445, 444, 443, 442, 441,

            # Left lower eyelid 
            464, 465, 453, 452, 451, 450, 449, 448, 261, 446, 359, 255, 339, 254, 253, 252, 256, 341,

            #Nose 
            1, 4, 5, 195, 197, 6, 168, 8, 9, 

            #Chin 
            152,  175, 428, 199, 208
        ]

        # Model
        try:
            input_dim = len(self.KEY_LANDMARK_INDICES) * 2
            checkpoint = torch.load(model_path, map_location=self.device)
            
            num_classes = 3 # Default
            state_dict = None

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
                if 'num_classes' in checkpoint:
                    num_classes = checkpoint['num_classes']
                    print(f"Found num_classes in checkpoint: {num_classes}")
            else:
                state_dict = checkpoint
                keys = list(state_dict.keys())
                if keys:
                    last_weight_key = keys[-2] 
                    if 'weight' in last_weight_key:
                         num_classes = state_dict[last_weight_key].shape[0]
                         print(f"Inferred num_classes from state_dict: {num_classes}")

            self.model = GazeClassifier(input_features=input_dim, num_classes=num_classes)
            self.model.to(self.device)

            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Model loaded successfully from state_dict.")
            
        except RuntimeError as e:
            print(f"Error loading model weights: {e}")
            print("Mismatch between GazeClassifier architecture and the loaded state_dict.")
            print("Please ensure src/gaze_classifier.py matches your training model exactly.")
            self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        
        # Scaler
        try:
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None

    def _extract_landmarks(self, face_landmarks):

        coords = []
        for idx in self.KEY_LANDMARK_INDICES:
            lm = face_landmarks.landmark[idx]
            coords.append(lm.x)
            coords.append(lm.y)
            
        return np.array(coords)

    def predict(self, face_landmarks):

        if self.model is None or self.scaler is None:
            return None

        raw_features = self._extract_landmarks(face_landmarks)

        features_reshaped = raw_features.reshape(1, -1)

        scaled_features = self.scaler.transform(features_reshaped)

        with torch.inference_mode():
            input_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(self.device)
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)

            return predicted.item()
