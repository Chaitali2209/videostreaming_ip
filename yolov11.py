import torch

from ultralytics import YOLO

class YOLOv11Detector:
    def __init__(self, model_path="yolo11n.pt", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        print(f"✅ Using device: {self.device}")
        self.classes_to_detect = [0]     # class 0 for person

    def predict(self, frame, conf_thres):
        results = self.model.predict(frame, conf=conf_thres, device=self.device, classes=self.classes_to_detect, verbose=False)
        return results