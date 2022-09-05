import torch
import cv2

class TorchHubEngine:
    def __init__(self, github,type, path, sourceType, classes, conf_thres, iou_thres, augment, size):
        print("loading weights from ", github, type, path, sourceType)
        if path:
            self.model = torch.hub.load(github, type, path=path, source=sourceType, force_reload=True)
        else:
            self.model = torch.hub.load(github, type, source=sourceType, force_reload=True)
        self.model.conf = conf_thres
        self.model.iou = iou_thres
        self.augment = augment
        self.size = size

    def infer(self, img):
        img_resized = cv2.resize(img, (self.size, self.size))
        pred = self.model([img_resized])
        classes = pred.pandas().xywh[0]["name"].tolist()
        pred = pred.xywh[0]
        return pred, classes

    def get_names(self):
        return self.model.module.names if hasattr(self.model, 'module') else self.model.names