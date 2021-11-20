import os
import tensorflow as tf
import numpy as np
import cv2
from tqdm import trange


def draw_predictions(new_frame, raw_result):
    from config import visibility_zone, blind_zones, probability_threshold, mask_proba_threshold

    cv2.rectangle(new_frame, visibility_zone[0], visibility_zone[1], (0, 0, 255), 3)

    for point1, point2 in blind_zones:
        cv2.rectangle(new_frame, point1, point2, (255, 0, 0), 3)

    fontScale = 1
    color = (0, 255, 0)
    thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    full_mask = np.zeros(new_frame.shape[:2])

    pig_count = 0
    for score, label, mask, box in zip(
            raw_result['detection_scores'][0].numpy(),
            raw_result['detection_classes'][0].numpy(),
            raw_result['detection_masks'][0].numpy(),
            raw_result['detection_boxes'][0].numpy()):

        if score < probability_threshold:
            continue

        y_min, x_min = int(box[0] * new_frame.shape[0]), int(box[1] * new_frame.shape[1])
        y_max, x_max = int(box[2] * new_frame.shape[0]), int(box[3] * new_frame.shape[1])

        if not (visibility_zone[0][0] < (x_min + x_max) // 2 < visibility_zone[1][0]
                and visibility_zone[0][1] < (y_min + y_max) // 2 < visibility_zone[1][1]):
            continue

        is_blind = False

        for point1, point2 in blind_zones:
            if point1[0] < (x_min + x_max) // 2 < point2[0] and point1[1] < (y_min + y_max) // 2 < point2[1]:
                is_blind = True

        if is_blind:
            continue

        mask = cv2.resize(mask, (x_max - x_min, y_max - y_min))
        mask = (mask > mask_proba_threshold).astype(int)

        full_mask[y_min:y_max, x_min:x_max] += mask

        cv2.rectangle(new_frame, (x_min, y_min), (x_max, y_max), color, thickness)

        org = (x_min, y_min)

        cv2.putText(new_frame, str(round(score, 3)), org, font, fontScale, color, thickness, cv2.LINE_AA)

        pig_count += 1

    cv2.putText(new_frame, f'Pig count: {pig_count}', (1300, 200), font, fontScale, color, thickness, cv2.LINE_AA)

    full_mask = np.clip(full_mask, 0, 1).astype(np.uint8)

    new_frame = (new_frame * 0.75).astype(np.uint8)
    new_frame[:, :, 1] += full_mask * 64
    return new_frame


def run_inference(video_path):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    loaded = tf.saved_model.load('models/mask_rcnn_inception_resnet_v2/saved_model/')
    infer = loaded.signatures["serving_default"]

    capture = cv2.VideoCapture(video_path)
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(f'output/{os.path.basename(video_path)}', fourcc, fps, (width, height))

    for _ in trange(length):
        ret, frame = capture.read()
        if frame is None:
            break
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        raw_result = infer(tf.convert_to_tensor(np.expand_dims(new_frame, axis=0)))
        new_frame = draw_predictions(new_frame, raw_result)

        writer.write(cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR))

    writer.release()

    capture.release()


if __name__ == '__main__':
    run_inference('data/train/Movie_1.mkv')
