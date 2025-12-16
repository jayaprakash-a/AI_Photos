import cv2
import numpy as np

class AestheticScorer:
    def __init__(self):
        pass

    def calculate_blur(self, image_path: str) -> float:
        """
        Calculate blur using Laplacian variance.
        Higher is sharper.
        Threshold ~100 is usually considered "not blurry".
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 0.0
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(variance)
        except Exception as e:
            print(f"Error calculating blur: {e}")
            return 0.0

    def predict_nima(self, image_path: str) -> float:
        """
        Placeholder for NIMA scoring.
        Returns a score 1-10.
        """
        # TODO: Load NIMA model
        return 5.0

scorer = AestheticScorer()
