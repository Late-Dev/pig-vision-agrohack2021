"""
    This module contains tracker wrapper
"""
import cv2
import numpy as np

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import preprocessing
from image_encoder import ImageEncoder


class TrackerWrapper:
    """
        Wrapper for tracking
    """

    def __init__(self,
                 max_cosine_distance=0.2,
                 min_confidence=0.8,
                 nms_max_overlap=1.0,
                 min_detection_height=0,
                 nn_budget=100,
                 patch_shape=(128, 64),
                 host="0.0.0.0",
                 port=6379):
        """Create tracker wrapper.

            Parameters
            ----------
            max_cosine_distance : float
                Gating threshold for cosine distance metric (object appearance).
            min_confidence : float
                Detection confidence threshold. Disregard all detections that have
                a confidence lower than this value.
            nms_max_overlap: float
                Maximum detection overlap (non-maxima suppression threshold).
            min_detection_height : int
                Detection height threshold. Disregard all detections that have
                a height lower than this value.
            nn_budget : Optional[int]
                Maximum size of the appearance descriptor gallery. If None, no budget
                is enforced.
            patch_shape : Optional[array_like]
                This parameter can be used to enforce a desired patch shape
                (height, width). First, the `bbox` is adapted to the aspect ratio
                of the patch shape, then it is clipped at the image boundaries.
                If None, the shape is computed from :arg:`bbox`.
            """
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.min_detection_height = min_detection_height
        self.image_encoder = ImageEncoder(host, port)
        self.patch_shape = patch_shape
        

    def process_detections(self, detections, image):
        """Process new detection and return tracking result

        Args
        ----------
            detections (List[Dict]): List of detections for current frame, detection has type as
                                     {"bbox":..., "confidence":...}
            image (np.ndarray): image in numpy array format

        Returns
        ----------
            List[Dict]: list of detections in format {"track_id": ..., "bbox": ...}
        """
        
        detections = self.__create_detections(detections, image, self.min_detection_height)
        detections = [d for d in detections if d.confidence >= self.min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        self.tracker.predict()
        self.tracker.update(detections)


        results = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append({"id": track.track_id, 
                            "bbox": tuple(bbox.astype(int)),
                            "time_since_update": track.time_since_update})

        return results


    def __create_detections(self, detections, image, min_height=0):
        detection_list = []
        for detection in detections:
            bbox, confidence = detection["bbox"], detection["confidence"]
            feature = self.__extrac_features(image, bbox)

            if bbox[3] < min_height:
                continue
            detection_list.append(Detection(bbox, confidence, feature))

        return detection_list


    def __extrac_features(self, image, bbox):
        patch = self.__extract_image_patch(image, bbox, self.patch_shape)
        features = self.image_encoder(patch)

        return features
        

    @staticmethod
    def __extract_image_patch(image, bbox, patch_shape):
        bbox = np.array(bbox)
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image
