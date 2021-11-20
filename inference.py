import os
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import trange

from service import PigMonitoringService
from save_script import savePredict


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser('Track pigs on video')
    parser.add_argument('--video_path', '-v', type=str, help='mp4 test file path')
    parser.add_argument('--out_path', '-op', type=str, help='output file path')
    parser.add_argument('--out_name', '-on', type=str, help='output filename')
    return parser


def build_pig_tracking_service():
    from detection.model import MaskRCNN, Coords
    from tracking.wrapper import DeepsortTracker
    from service import PigMonitoringService

    pig_segmentator = MaskRCNN('models/mask_rcnn_inception_resnet_v2/saved_model/')
    deep_sort_tracker = DeepsortTracker('models/mars-small128.pb')

    pig_tracking_service = PigMonitoringService(pigs_segmentator=pig_segmentator, tracker=deep_sort_tracker)
    return pig_tracking_service


def setColors_range(n_range: int):
    color_map = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(n_range)}
    return color_map


def draw_predictions(frame, coords, colormap, track_mean_points):
    from config import visibility_zone, blind_zones, probability_threshold, mask_proba_threshold

    cv2.rectangle(frame, visibility_zone[0], visibility_zone[1], (0, 0, 255), 3)

    # for point1, point2 in blind_zones:
    #     cv2.rectangle(frame, point1, point2, (255, 0, 0), 3)

    fontScale = 1
    thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    full_mask = np.zeros(frame.shape[:2])

    count = 0
    for coord in coords:
        color = colormap[coord.tracking_id]
        mask = coord.mask
        x_min, y_min, x_max, y_max = coord.x_min, coord.y_min, coord.x_max, coord.y_max

        full_mask[y_min:y_max, x_min:x_max] += mask
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
        
        if len(track_mean_points[coord.tracking_id]) > 1:
            cv2.polylines(frame, np.array([track_mean_points[coord.tracking_id]]), False, color, thickness)

        count += 1

    cv2.putText(frame, f'Pig count: {count}', (1300, 200), font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

    full_mask = np.clip(full_mask, 0, 1).astype(np.uint8)

    frame = (frame * 0.75).astype(np.uint8)
    frame[:, :, 1] += full_mask * 64
    return frame


def process_video(video_path: Path, tracking_service: PigMonitoringService):
    frames_preds = []
    track_mean_points = {}
    colormap = setColors_range(150)

    capture = cv2.VideoCapture(video_path)
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(f'output/{os.path.basename(video_path)}', fourcc, fps, (width, height))

    for _ in trange(length, desc='Processing video...'):
        ret, frame = capture.read()
        if frame is None:
            break
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        tracked_boxes = pig_tracking_service.process_frame(new_frame)
        
        for track_box in tracked_boxes:
            track_mean_points.setdefault(track_box.tracking_id, []).append([(track_box.x_min+track_box.x_max) // 2, (track_box.y_min+track_box.y_max) // 2])
        
        frames_preds.append(tracked_boxes)
        new_frame = draw_predictions(new_frame, tracked_boxes, colormap, track_mean_points)
        
        writer.write(cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR))

    writer.release()     
    capture.release()
    return frames_preds



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    video_path = args.video_path
    pig_tracking_service = build_pig_tracking_service()
    frames_preds = process_video(video_path, pig_tracking_service)
    # write predictions
    boxes = []
    masks = []
    num_pigs = []
    track_ids = []
    for frame_pred in frames_preds:
        frame_boxes = [[i.x_min, i.y_min, i.x_max, i.y_max] for i in frame_pred]
        frame_masks = [i.mask for i in frame_pred]
        frame_tracks = [i.tracking_id for i in frame_pred]

        boxes.append(frame_boxes)
        masks.append(frame_masks)
        num_pigs.append(len(frame_boxes))
        track_ids.append(frame_tracks)

    savePredict(
        Path=args.out_path,
        Name=args.out_name,
        boxs=boxes,
        masks=masks,
        num_pigs=num_pigs,
        track_ids=track_ids
    )
