import numpy as np
import cv2

from detection.model import Coords
from tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.tracker import Tracker
from tracking.deep_sort import generate_detections as gdet


class DeepsortTracker:
    #initialize deep sort object
    max_cosine_distance = 0.7
    nn_budget = None

    def __init__(self, model_path: str):
        self.encoder = gdet.create_box_encoder(model_path, batch_size=1)
        self.metric = NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)

    def track_boxes(self, frame, boxes, scores, names, masks):
        """
        
        bbox - (x_min, y_min, w, h)
        """
        features = np.array(self.encoder(frame, boxes))

        detections = [
            Detection(bbox, score, class_name, feature, mask) 
            for bbox, score, class_name, feature, mask 
            in zip(boxes, scores, names, features, masks)
            ]

        # Pass detections to the deepsort object and obtain the track information.
        self.tracker.predict()
        self.tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            class_mask = track.get_mask()
            tracking_id = track.track_id # Get the ID for the particular track
            # index = key_list[val_list.index(class_name)] # Get predicted object index by object name

            x_min, y_min, x_max, y_max = [min(max(int(i), 0), 1700) for i in bbox.tolist()]

            class_mask = cv2.resize(class_mask, (x_max - x_min, y_max - y_min))

            det_obj = Coords(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                score=None,
                tracking_id=tracking_id,
                mask=class_mask
                )
            tracked_bboxes.append(det_obj) # Structure data, that we could use it with our draw_bbox function
        return tracked_bboxes
