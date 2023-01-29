import yolov5
from yolov5.helpers import YOLOv5

# load model


class Yolo_v5:
    def __init__(self, conf=0.25, iou=0.45, agnostic=False, multi_label=False, max_det=1000):
        self.conf = conf
        self.iou = iou
        self.agnostic = agnostic
        self.multi_label = multi_label
        self.max_det = max_det
        self.model = YOLOv5('keremberke/yolov5s-csgo')
        self.model.load_model()

    def predict(self, image_list):
        results = self.model.predict(image_list, size=2000)
        return results
