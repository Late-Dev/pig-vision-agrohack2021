from typing import List
from dataclasses import dataclass

import numpy as np
import cv2
import tensorflow as tf

import config


@dataclass
class Coords:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    score: float
    mask: np.array
    tracking_id: int = None
    class_name = "Pig"

    def __post_init__(self):
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.bbox = (self.x_min, self.y_min, self.width, self.height)


class MaskRCNN:
    def __init__(self, model_dir: str) -> None:
        self._set_device()
        self._load_model(model_dir)

    def detect(self, img: np.array) -> List[Coords]:
        contours = []
        raw_result = self.infer(tf.convert_to_tensor(np.expand_dims(img, axis=0)))
        # iter trough detection candidates
        for score, label, mask, box in zip(
            raw_result['detection_scores'][0].numpy(), 
            raw_result['detection_classes'][0].numpy(), 
            raw_result['detection_masks'][0].numpy(),
            raw_result['detection_boxes'][0].numpy()):
            if score < config.probability_threshold:
                continue

            y_min, x_min = int(box[0] * img.shape[0]), int(box[1] * img.shape[1])
            y_max, x_max = int(box[2] * img.shape[0]), int(box[3] * img.shape[1])

            mask = (mask > config.mask_proba_threshold).astype(np.uint8)
            mask = cv2.resize(mask, (x_max - x_min, y_max - y_min))

            if not (config.visibility_zone[0][0] < (x_min + x_max) // 2 < config.visibility_zone[1][0]
                and config.visibility_zone[0][1] < (y_min + y_max) // 2 < config.visibility_zone[1][1]):
                continue

            is_blind = False

            for point1, point2 in config.blind_zones:
                if point1[0] < (x_min + x_max) // 2 < point2[0] and point1[1] < (y_min + y_max) // 2 < point2[1]:
                    is_blind = True

            if is_blind:
                continue
            contours.append(Coords(x_min, y_min, x_max, y_max, score, mask))
        return contours
            
    def _set_device(self, gpu_id=0):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)

    def _load_model(self, model_dir):
        loaded = tf.saved_model.load(model_dir)
        self.infer = loaded.signatures["serving_default"]

