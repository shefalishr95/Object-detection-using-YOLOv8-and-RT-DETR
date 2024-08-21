from ultralytics import YOLO


class YOLOv8Model:
    def __init__(self, config_path, weights_path=None):
        self.model = YOLO(config_path)
        if weights_path:
            self.model = self.model.load(weights_path)

    def train(self, data, epochs, imgsz, device):
        return self.model.train(data=data, epochs=epochs, imgsz=imgsz, device=device)

    def evaluate(self, data):
        return self.model.val(data=data)

    def predict(self, image):
        return self.model(image)
