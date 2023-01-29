from yolov5.helpers import YOLOv5


class Yolo_v5:
    def __init__(self):
        self.model = YOLOv5('keremberke/yolov5s-csgo')
        self.model.load_model()

    def predict(self, image_list):
        results = self.model.predict(image_list, size=2000)
        return results
