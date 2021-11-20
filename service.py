from typing import List
import numpy as np
import cv2
from tqdm import tnrange

from detection.model import MaskRCNN, Coords
from tracking.deep_sort.detection import Detection
from tracking.wrapper import DeepsortTracker


class PigMonitoringService:
    def __init__(self, pigs_segmentator: MaskRCNN, tracker: DeepsortTracker) -> None:
        self.pigs_segmnetator = pigs_segmentator
        self.deepsort_tracker = tracker

    def process_frame(self, frame: np.array):
        # detect pigs
        frame_boxes = self.pigs_segmnetator.detect(frame)
        bboxes = [i.bbox for i in frame_boxes]
        scores = [i.score for i in frame_boxes]
        names = [i.class_name for i in frame_boxes]
        masks = [i.mask for i in frame_boxes]
        tracked_boxes = self.deepsort_tracker.track_boxes(
            frame=frame, 
            boxes=bboxes, 
            scores=scores, 
            names=names,
            masks=masks
            )
    
        return tracked_boxes
