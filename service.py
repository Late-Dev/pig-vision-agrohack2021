import cv2
from tqdm import tnrange

from detection.model import MaskRCNN, Coords


class PigMonitoringService:
    def __init__(self, segmentator: MaskRCNN, tracker: None) -> None:
        pass

    def _init_tracker():
        pass

    def process_frames(file_path: str, save_path: str):
        w, h = 1700, 1700
        fps = 10
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        c = 0
        capture = cv2.VideoCapture(file_path)
        for i in trange(60*10 + 1):
            ret, frame = capture.read()
            if frame is None:
                break
            new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            for frame_coords in segmentator.detect(new_frame):
            
                new_frame = segmentator.draw_predictions(new_frame, frame_coords)
            
                writer.write(cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR))

        writer.release() 
            
        capture.release()
