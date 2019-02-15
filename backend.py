import os
import os.path
import pickle
import random
import sys

import numpy as np
import PIL.Image

import sys
sys.path.append(os.path.abspath('stylegan'))

import dnnlib
import dnnlib.tflib as tflib

latent_dims = 512

url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl

cache_dir = 'cache'

class NetworkWrapper(object):
    def __init__(self):
        with dnnlib.util.open_url(url, cache_dir) as f:
            _G, _D, Gs = pickle.load(f)
        self.Gs = Gs
        self.inputShape = self.Gs.input_shape[1]
        self.fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    def genImage(self, latents):
        #a = np.asanyarray(PIL.Image.open('images/-2_-2_-2_-2_0_0_0s.png'))
        a = self.Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=self.fmt)[0]
        return PIL.Image.fromarray(a, 'RGB')

    def genRandomImage(self):
        latents = np.random.randn(1, self.Gs.input_shape[1])
        return self.genImage(latents)
