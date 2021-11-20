# ObjectTracking-DeepSORT-YOLOv3-TF2
This repository implements YOLOv3 and Deep SORT in order to perform real-time object tracking. My Blog post https://medium.com/analytics-vidhya/object-tracking-using-deepsort-in-tensorflow-2-ec013a2eeb4f

## Installation

First, clone or download this GitHub repository. Install requirements and download pretrained weights:

```
https://github.com/anushkadhiman/ObjectTracking-DeepSORT-YOLOv3-TF2.git
cd ObjectTracking-DeepSORT-YOLOv3-TF2
````

```
pip install -r ./requirements.txt
`````

```
# yolov3
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

# yolov3-tiny
wget -P model_data https://pjreddie.com/media/files/yolov3-tiny.weights
``````


## Tracking

```
python object_tracker.py
````

## Result

### Tracking Person by Pre-trained model
![Alt text](tracking.gif?raw=true "video")

## References
1. Deep SORT Repository - https://github.com/nwojke/deep_sort

2. https://pylessons.com/ — Rokas Balsys






