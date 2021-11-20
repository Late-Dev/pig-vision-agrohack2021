FROM tensorflow/tensorflow:2.3.1-gpu-jupyter
WORKDIR /tracking
COPY . /tracking

RUN pip install -U pip
RUN pip install -U setuptools
RUN pip install -r requirements.txt
RUN pip install opencv-python
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev