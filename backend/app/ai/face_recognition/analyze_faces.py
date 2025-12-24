"""
Standalone Face Analysis Script
Detect and recognize all faces in an image

Requirements:
- torch
- torchvision
- facenet-pytorch
- numpy
- Pillow
- matplotlib
- opencv-python

Usage:
    python analyze_faces.py --image photo.jpg --model-dir path/to/models
"""
import warnings

warnings.filterwarnings(
    "ignore",
    message="Protobuf gencode version.*"
)
import os
import argparse
import json
import logging
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from facenet_pytorch import MTCNN, InceptionResnetV1
from transformers import AutoImageProcessor, MobileNetV2ForImageClassification
import matplotlib.pyplot as plt
import matplotlib.patches as patches

torch.nn.Module.dump_patches = False
warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)



# ============================================================================
# CONFIGURATION (Embedded - no external config file needed)
# ============================================================================

DEFAULT_CONFIG = {
    'detection': {
        'confidence_threshold': 0.95,
        'min_face_size': 40,
        'image_size': 160,
        'nms_iou_threshold': 0.4,
        'min_recognition_confidence': 0.70,
        'padding': 0.2
    }
}


# ============================================================================
# UTILITY FUNCTIONS (Embedded - no external utils module needed)
# ============================================================================

def setup_logger(verbose: bool = False) -> logging.Logger:
    """Setup logging"""
    logger = logging.getLogger('FaceAnalysis')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_image(image_path: str) -> np.ndarray:
    """Load image from file"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def non_max_suppression(boxes: List[np.ndarray], probs: List[float], 
                        iou_threshold: float = 0.5) -> Tuple[List[np.ndarray], List[float]]:
    """Apply Non-Maximum Suppression to remove overlapping boxes"""
    if len(boxes) == 0:
        return [], []
    
    boxes_array = np.array([box for box in boxes])
    probs_array = np.array(probs)
    
    # Sort by confidence
    indices = np.argsort(probs_array)[::-1]
    
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes_array[current]
        remaining_boxes = boxes_array[indices[1:]]
        
        # Calculate intersection
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * \
                         (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = current_area + remaining_areas - intersection
        
        # Calculate IoU
        iou = intersection / union
        
        # Keep boxes with IoU below threshold
        indices = indices[1:][iou < iou_threshold]
    
    filtered_boxes = [boxes[i] for i in keep]
    filtered_probs = [probs[i] for i in keep]
    
    return filtered_boxes, filtered_probs


# ============================================================================
# FACE ANALYZER CLASS
# ============================================================================

class FaceAnalyzer:
    """Standalone face analyzer - detect and recognize faces"""
    
    def __init__(self, model_dir: str, config: Optional[Dict[str, Any]] = None, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize face analyzer.
        
        Args:
            model_dir: Directory containing model files (best_model.pth, label_mapping.json)
            config: Optional configuration dictionary (uses defaults if not provided)
            logger: Optional logger instance
        """
        self.model_dir = model_dir
        self.config = config or DEFAULT_CONFIG
        self.logger = logger or setup_logger()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize face detector (MTCNN)
        detection_config = self.config['detection']
        self.detector = MTCNN(
            image_size=detection_config['image_size'],
            margin=0,
            min_face_size=detection_config['min_face_size'],
            thresholds=[0.6, 0.7, detection_config['confidence_threshold']],
            device=self.device,
            keep_all=True
        )
        
        # Image transformations for recognition
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load recognition model and labels
        self.recognition_model = None
        self.label_to_name = None
        self.load_recognition_model()
    
    def load_recognition_model(self):
        """Load trained face recognition model"""
        model_path = os.path.join(self.model_dir, "best_model.pth")
        
        if not os.path.exists(model_path):
            self.logger.warning(f"Recognition model not found at {model_path}")
            self.logger.warning("Face detection will work, but recognition will be disabled")
            return
        
        # Load label mapping
        label_map_path = os.path.join(self.model_dir, "label_mapping.json")
        with open(label_map_path, 'r') as f:
            self.label_to_name = json.load(f)
            self.label_to_name = {int(k): v for k, v in self.label_to_name.items()}
        
        num_classes = len(self.label_to_name)
        
        # Initialize and load model
        self.recognition_model = InceptionResnetV1(pretrained='vggface2', classify=False)
        self.recognition_model.logits = nn.Linear(512, num_classes)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.recognition_model.load_state_dict(checkpoint['model_state_dict'])
        self.recognition_model.to(self.device)
        self.recognition_model.eval()
        
        self.logger.info(f"Recognition model loaded with {num_classes} identities")
    
    def detect_faces(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """Detect all faces in image"""
        pil_image = Image.fromarray(image)
        boxes, probs = self.detector.detect(pil_image)
        
        if boxes is None:
            return [], []
        
        # Apply confidence threshold and NMS
        threshold = self.config['detection']['confidence_threshold']
        valid_indices = [i for i, prob in enumerate(probs) if prob >= threshold]
        
        boxes = [boxes[i] for i in valid_indices]
        probs = [probs[i] for i in valid_indices]
        
        iou_threshold = self.config['detection']['nms_iou_threshold']
        boxes, probs = non_max_suppression(boxes, probs, iou_threshold)
        
        return boxes, probs
    
    def recognize_face(self, face_image: np.ndarray) -> Dict[str, Any]:
        """Recognize identity of a face"""
        if self.recognition_model is None:
            return {
                'identity': 'Unknown',
                'confidence': 0.0,
                'label': -1,
                'top3': []
            }
        
        # Preprocess face
        face_pil = Image.fromarray(face_image)
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            embeddings = self.recognition_model(face_tensor)
            outputs = self.recognition_model.logits(embeddings)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
        
        predicted_label = predicted.item()
        confidence_score = confidence.item()
        
        # Get minimum confidence threshold
        min_confidence = self.config['detection']['min_recognition_confidence']
        
        # Get top-3 predictions
        top3_probs, top3_indices = probs.topk(min(3, len(self.label_to_name)), dim=1)
        top3_predictions = []
        for prob, idx in zip(top3_probs[0], top3_indices[0]):
            top3_predictions.append({
                'identity': self.label_to_name[idx.item()],
                'confidence': prob.item()
            })
        
        # Check if confidence meets threshold
        if confidence_score < min_confidence:
            return {
                'identity': 'Unknown',
                'confidence': confidence_score,
                'label': -1,
                'top3': top3_predictions
            }
        
        return {
            'identity': self.label_to_name[predicted_label],
            'confidence': confidence_score,
            'label': predicted_label,
            'top3': top3_predictions
        }
    
    def crop_face(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Crop face from image with padding"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Calculate padding
        padding = self.config['detection']['padding']
        face_w = x2 - x1
        face_h = y2 - y1
        pad_w = int(face_w * padding)
        pad_h = int(face_h * padding)
        
        # Apply padding with bounds checking
        x1 = max(0, int(x1 - pad_w))
        y1 = max(0, int(y1 - pad_h))
        x2 = min(w, int(x2 + pad_w))
        y2 = min(h, int(y2 + pad_h))
        
        return image[y1:y2, x1:x2]
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image for all faces and identify them"""
        self.logger.info(f"Analyzing image: {image_path}")
        
        # Load image
        image = load_image(image_path)
        
        # Detect faces
        boxes, detection_probs = self.detect_faces(image)
        
        self.logger.info(f"Detected {len(boxes)} faces")
        
        # Recognize each face
        faces = []
        for i, (bbox, det_prob) in enumerate(zip(boxes, detection_probs)):
            # Crop face
            face_crop = self.crop_face(image, bbox)
            
            # Recognize
            recognition_result = self.recognize_face(face_crop)
            
            # Store results
            faces.append({
                'face_id': i,
                'bbox': bbox.tolist() if isinstance(bbox, np.ndarray) else bbox,
                'detection_confidence': float(det_prob),
                'identity': recognition_result['identity'],
                'recognition_confidence': recognition_result['confidence'],
                'top3_predictions': recognition_result['top3']
            })
            
            self.logger.info(f"  Face {i}: {recognition_result['identity']} "
                           f"(confidence: {recognition_result['confidence']:.2%})")
        
        # Filter duplicate identities - keep only highest confidence for each identity
        identity_best_faces = {}
        for face in faces:
            identity = face['identity']
            confidence = face['recognition_confidence']
            
            # Skip "Unknown" from deduplication
            if identity == 'Unknown':
                continue
            
            # Keep the face with highest confidence for this identity
            if identity not in identity_best_faces or confidence > identity_best_faces[identity]['recognition_confidence']:
                identity_best_faces[identity] = face
        
        # Collect all Unknown faces and best faces for each identity
        filtered_faces = []
        for face in faces:
            if face['identity'] == 'Unknown':
                filtered_faces.append(face)
            elif face in identity_best_faces.values():
                filtered_faces.append(face)
        
        # Log filtering results
        if len(faces) != len(filtered_faces):
            removed_count = len(faces) - len(filtered_faces)
            self.logger.info(f"Filtered {removed_count} duplicate identity detections")
        
        # Re-assign face IDs after filtering
        for i, face in enumerate(filtered_faces):
            face['face_id'] = i
        
        # Filter out Unknown identities from final results
        final_faces = [f for f in filtered_faces if f['identity'] != 'Unknown']
        
        # Get unique identities (excluding Unknown)
        unique_identities = list(set([f['identity'] for f in final_faces]))
        
        # Log filtering results
        unknown_count = len(filtered_faces) - len(final_faces)
        if unknown_count > 0:
            self.logger.info(f"Filtered out {unknown_count} Unknown faces from final results")
        
        return {
            'image_path': image_path,
            'num_faces': len(final_faces),
            'faces': final_faces,
            'unique_identities': unique_identities,
            'image_shape': image.shape
        }

class EyeAnalyzer:
    """Analyzer for eyeglasses detection and eye state (open/closed)"""
    
    def __init__(self, model_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initialize eye analyzer models and processor.
        
        Args:
            model_dir: Directory containing model files
            logger: Optional logger instance
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger or setup_logger()
        self.model_dir = model_dir

        # 1. Initialize Face/Landmark Detector
        self.detector = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            device=self.device,
            post_process=False,
            select_largest=True
        )
        
        # 2. Load Glasses Model
        self.glasses_model = self._load_glasses_model(os.path.join(model_dir, "glasses_detector.pth"))
        
        # 3. Load Eye State Model & Processor
        self.eye_processor, self.eye_model = self._load_eye_state_model()
        
        # Transforms for Glasses (ImageNet compatible)
        self.glasses_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_glasses_model(self, path: str) -> Optional[nn.Module]:
        if not os.path.exists(path):
            self.logger.warning(f"Glasses model weights not found at {path}")
            return None
        try:
            # Direct loading as per user preference (assumes whole model saved)
            model = torch.load(path, map_location=self.device)
            # Fix AvgPool2d compatibility if needed (common patch for some versions)
            for module in model.modules():
                if isinstance(module, nn.AvgPool2d):
                    module.divisor_override = None
            model.to(self.device).eval()
            self.logger.info(f"Glasses model loaded from {path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading glasses model: {e}")
            return None

    def _load_eye_state_model(self) -> Tuple[Optional[AutoImageProcessor], Optional[nn.Module]]:
        model_name = "MichalMlodawski/open-closed-eye-classification-mobilev2"
        # Try local first, then Hub
        local_path = os.path.join(self.model_dir, "eye_state_detector")
        load_path = local_path if os.path.exists(local_path) else model_name
        
        try:
            processor = AutoImageProcessor.from_pretrained(load_path, use_fast=True)
            model = MobileNetV2ForImageClassification.from_pretrained(load_path)
            model.to(self.device).eval()
            self.logger.info(f"Eye state model loaded from {load_path}")
            return processor, model
        except Exception as e:
            self.logger.error(f"Error loading eye state model: {e}")
            return None, None

    def analyze(self, image: np.ndarray, face: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run eyeglasses and eye state analysis on a specific face crop.
        
        Args:
            image: Full RGB image as numpy array
            face: Dictionary containing face info including 'bbox'
            
        Returns:
            Dictionary with results (has_glasses, eyes_open, etc.)
        """
        bbox = face['bbox']
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        # Apply same padding as used in FaceAnalyzer
        padding = DEFAULT_CONFIG['detection']['padding']
        pw, ph = int((x2-x1) * padding), int((y2-y1) * padding)
        
        cx1, cy1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
        cx2, cy2 = min(w, int(x2 + pw)), min(h, int(y2 + ph))
        
        face_crop_np = image[cy1:cy2, cx1:cx2]
        face_crop_pil = Image.fromarray(face_crop_np)

        results = {
            "has_glasses": False,
            "glasses_label": "no-glasses",
            "eyes_open": True,
            "details": {}
        }

        # 1. Glasses Detection
        if self.glasses_model:
            input_tensor = self.glasses_transform(face_crop_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.glasses_model(input_tensor)
                _, pred = torch.max(out, 1)
            classes = ['glasses', 'no-glasses']
            results["glasses_label"] = classes[pred.item()]
            results["has_glasses"] = (pred.item() == 0) # Index 0 is 'glasses'
        
        # 2. Eye Landmarks and State
        # Detect landmarks on the face crop
        _, _, landmarks = self.detector.detect(face_crop_pil, landmarks=True)
        
        if landmarks is not None and len(landmarks) > 0:
            landmark = landmarks[0]
            # results["landmarks"] = landmark.tolist() # Landmarks relative to face crop
            
            # Eye state analysis using the MobileNetV2 model
            if self.eye_model and self.eye_processor:
                # Note: The model currently processes the entire face crop, 
                # but we could crop to specific eyes if needed for higher accuracy.
                # Here we use the simplified logic from the previous iteration.
                
                inputs = self.eye_processor(images=face_crop_pil, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.eye_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    conf, pred = torch.max(probs, 1)
                
                # Assume pred.item() == 1 is 'open' (based on MichalMlodawski/open-closed-eye-classification-mobilev2)
                is_open = (pred.item() == 1)
                results["eyes_open"] = is_open
                results["details"].update({
                    "confidence": conf.item(),
                    "label": "open" if is_open else "closed"
                })

        return results

class Analyzer:
    def __init__(self, model_dir: str, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger()
        self.face_analyzer = FaceAnalyzer(model_dir)
        self.eye_analyzer = EyeAnalyzer(model_dir)  
    
    def analyze(self, image_path: str, output_path: str):
        """Run full analysis (faces, glasses, eyes)"""
        # Load image once
        image = load_image(image_path)
        
        # 1. Detect and Recognize Faces
        results = self.face_analyzer.analyze_image(image_path)
        
        # 2. Analyze Eyes (passing the already loaded image)
        for i, face in enumerate(results['faces']):
            eye_results = self.eye_analyzer.analyze(image, face)
            results['faces'][i].update(eye_results)

        return results
    def visualize_results(self, image_path: str, results: Dict[str, Any], output_path: str) -> str:
        image = load_image(image_path)
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.imshow(image)
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for face in results['faces']:
            x1, y1, x2, y2 = face['bbox']
            color = colors[face['face_id'] % 10]
            ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor=color, facecolor='none'))
            
            label = f"{face['identity']}\n"
            label += f"Glass: {'Y' if face['has_glasses'] else 'N'}"
            # Show eye state if eyes detected and either open or glasses not strictly blocking
            label += f" | Eyes: {'Open' if face['eyes_open'] else 'Closed'}"
            
            ax.text(x1, y1-10, label, fontsize=10, color='white', weight='bold', bbox=dict(facecolor=color, alpha=0.8))

        ax.set_title(f"Found {results['num_faces']} faces", fontsize=16)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Complete Face & Eye Analysis (DL)")
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--save-json', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    if not os.path.exists(args.output): os.makedirs(args.output, exist_ok=True)
    
    analyzer = Analyzer(args.model_dir, logger=setup_logger(verbose=args.verbose))
    results = analyzer.analyze(args.image, args.output)
    
    if args.save_json:
        with open(os.path.join(args.output, 'results.json'), 'w') as f: json.dump(results, f, indent=2)
            
    viz_fn = os.path.basename(args.image).replace('.', '_analyzed.')
    viz_path = analyzer.visualize_results(args.image, results, os.path.join(args.output, viz_fn))
    
    print(f"\n✅ Done! Viz: {viz_path}")
    for f in results['faces']:
        eye = "Open" if f['eyes_open'] else "Closed"
        if f['has_glasses']: eye = "N/A"
        print(f"Face #{f['face_id']}: {f['identity']} | Glasses: {f['has_glasses']} | Eyes: {eye}")

if __name__ == "__main__": main()
