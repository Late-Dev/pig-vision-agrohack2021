import numpy as np
import redisai as rai

class ImageEncoder(object):
    def __init__(self, host, port):
        self.client = rai.Client(host=host, port=port)


    def __call__(self, image):
        image = image.astype(np.uint8)
        image = np.expand_dims(image, axis=0)

        self.client.tensorset("images", image)
        out = self.client.modelrun("image_encoder", "images", "out")
        result = self.client.tensorget('out')[0]

        return result
