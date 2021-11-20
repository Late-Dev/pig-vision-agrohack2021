FROM tensorflow/tensorflow:2.3.1-gpu-jupyter
WORKDIR /agro_tech
COPY . /agro_tech

RUN pip install -U pip
RUN pip install -U setuptools
# RUN pip install -r requirements.txt
RUN pip install opencv-python
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
CMD jupyter notebook --ip 0.0.0.0 --port 9988 --allow-root --NotebookApp.token=""