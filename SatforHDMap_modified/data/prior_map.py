import os.path
import json
from math import sin, cos, tan, pi
import numpy as np
import random
from PIL import Image
from torchvision import transforms
import torchvision
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .coordinate_trans import local2global, params


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))


class PriorMap(object):
    def __init__(self,
                 prior_map_root,
                 img_size):
        super().__init__()
        self.data_root = prior_map_root
        f = open(os.path.join(self.data_root, 'map_prior.json'), 'r')
        content = f.read()
        self.config = json.loads(content)
        self.img_size = img_size

    # gete satellite map by sample token
    def get_prior_map(self, token):
        img_file = self.config[token]

        # values = list(self.config.values())
        # ran = random.randint(0, len(values)-1)
        # img_file = values[ran]

        img = Image.open(os.path.join(self.data_root, img_file))
        # print(img.size)
        img = img.resize(self.img_size)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # print(img.size)
        img_tensor = normalize_img(img)
        # print(img_tensor.shape)
        return img_tensor


if __name__ == '__main__':
    prior_map = PriorMap(prior_map_root='/opt/data/private/prior_map_dataset/prior_map_trainval', img_size=(400, 200))
    for key in prior_map.config.keys():
        img = prior_map.get_prior_map(key)

