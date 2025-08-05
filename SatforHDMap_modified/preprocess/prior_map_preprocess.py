import os
import numpy as np
import math
import json
from math import cos, sin
from PIL import Image
from data.dataset import HDMapNetSemanticDataset
from coordinatetrans.coordinate_trans import local2global, params
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt


def get_prior_map_tile(img, loc, translation, rotation, data_conf):
    tile_x, tile_y = local2global(translation[0], translation[1], loc)
    tile_x, tile_y = round(tile_x), round(tile_y)
    patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]
    patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]
    tile_h = round(patch_h * abs(params[loc][2]))
    tile_w = round(patch_w * abs(params[loc][0]))
    tile_box = (tile_x, tile_y, tile_h, tile_w)
    tile_angle = quaternion_yaw(Quaternion(rotation))  #radians

    tile = affine_transform(img, tile_box, tile_angle)
    return tile


# def affine_transform(img, tile_box, tile_angle):
#     tile_angle = torch.tensor(tile_angle)
#     cosp = torch.cos(tile_angle)
#     sinp = torch.sin(tile_angle)
#     tile_x, tile_y, tile_h, tile_w = tile_box
#     theta = torch.stack([
#         torch.stack([cosp, -sinp, tile_x-tile_x*cosp+tile_y*sinp], dim=-1),
#         torch.stack([sinp, cosp, tile_y-tile_y*cosp-tile_x*sinp], dim=-1)
#     ], dim=-2)
#     shape = img.shape
#
#     grids = F.affine_grid(theta.unsqueeze(0), torch.Size((1, shape[0], tile_h, tile_w)), align_corners=True)
#     grids = grids.to(torch.float32)
#     print(img.dtype)
#     print(grids.dtype)
#     cropped_features = F.grid_sample(img.unsqueeze(0), grids, align_corners=True)
#     return cropped_features


# img: H, W, C, numpy.ndarray
def affine_transform(img, tile_box, tile_angle):
    tile_x, tile_y, tile_h, tile_w = tile_box
    sinp = sin(tile_angle)
    cosp = cos(tile_angle)
    # plt.imshow(img)
    # plt.show()
    shape = img.shape
    tile = np.zeros((tile_h+1, tile_w+1, shape[2]))
    for y in range(tile_h):
        for x in range(tile_w):
            y_trans = tile_y - (tile_h / 2 - y) * cosp + (tile_w / 2 - x) * sinp
            x_trans = tile_x - (tile_h / 2 - y) * sinp - (tile_w / 2 - x) * cosp
            # y_trans = tile_y + (y-tile_h/2)*cosp + (x-tile_w/2)*sinp
            # x_trans = tile_x - (y-tile_h/2)*sinp + (x-tile_w/2)*cosp
            tile[y][x] = inter_linear(img, y_trans, x_trans)
    # plt.imshow(tile)
    # plt.show()
    # tile_img = Image.fromarray(tile)
    # tile_img.save("tile.jpg", m='RGB')
    return tile


def inter_linear(img, y, x):
    shape = img.shape
    if y < 0 or y >= shape[0] or x<0 or x>=shape[1]:
        return np.zeros(shape[2])
    y_0 = math.floor(y)
    y_1 = y_0+1
    x_0 = math.floor(x)
    x_1 = x_0+1
    value0 = img[y_0][x_0] + (img[y_0][x_1]-img[y_0][x_0])*(x-x_0)/(x_1-x_0)
    value1 = img[y_1][x_0] + (img[y_1][x_1]-img[y_1][x_0])*(x-x_0)/(x_1-x_0)
    value = value0 + (value1 - value0) * (y-y_0)/(y_1-y_0)
    return value


if __name__ == '__main__':
    data_conf = {
        'image_size': (512, 512),
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36
    }
    map_files = {'boston-seaport':'/home/wjgao/Documents/PriorMapNet/map/boston.png',
                 'singapore-hollandvillage':'/home/wjgao/Documents/PriorMapNet/map/HollandVillage.png',
                 'singapore-onenorth':'/home/wjgao/Documents/PriorMapNet/map/OneNorth.png',
                 'singapore-queenstown':'/home/wjgao/Documents/PriorMapNet/map/Queenstown.png'}
    # map_files = {'boston-seaport': '/home/wjgao/Documents/PriorMapNet/map/boston.png'}
    # map_files = {'singapore-onenorth':'/home/wjgao/Documents/PriorMapNet/map/OneNorth.png'}
    map_img = {}
    for key, value in map_files.items():
        img = Image.open(value)
        img = img.convert('RGB')
        img_torch = transforms.ToTensor()(img)
        img = img_torch.numpy().transpose(1, 2, 0)  # H, W, C
        # img = img_torch.numpy().transpose(1, 2, 0)
        map_img[key] = img

    data_root = '/home/wjgao/Documents/PriorMapNet/prior_map_val'
    dataset = HDMapNetSemanticDataset(version='v1.0-trainval', dataroot='/media/wjgao/Document/data/set/nuscenes', prior_map_root='map', data_conf=data_conf, is_train=False)
    # for idx in range(dataset.__len__()):
    #     pass
    prior_map_dict = {}
    # for i in range(1):
    for i in range(len(dataset.samples)):
        sample = dataset.samples[i]
        sample_data_record = dataset.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        ego_pose = dataset.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        # print(ego_pose)
        location = dataset.nusc.get('log', dataset.nusc.get('scene', sample['scene_token'])['log_token'])['location']
        tile = get_prior_map_tile(map_img[location], location, ego_pose['translation'], ego_pose['rotation'], data_conf)
        im_file = os.path.join(data_root, "{}_{}_{}_{}_{}.jpg".format(ego_pose['timestamp'],
                                                                      location,
                                                                   ego_pose['translation'][0],
                                                                   ego_pose['translation'][1],
                                                                   sample_data_record['ego_pose_token']))
        im = Image.fromarray((tile*255).astype(np.uint8))
        im.save(im_file)
        # print(im_file)
        prior_map_dict[sample_data_record['ego_pose_token']] = im_file
        if i % 100 == 0:
            print("{}/{}".format(i, len(dataset.samples)))
    with open(os.path.join(data_root, "prior_map.json"), "w") as f:
        f.write(json.dumps(prior_map_dict, ensure_ascii=False, indent=4, separators=(',', ':')))

    # print(dataset.samples[0])

    # tile = affine_transform(map_img['singapore-onenorth'], (3371, 3002, 1000, 2000), np.pi/6)
    # im = Image.fromarray((tile*255).astype(np.uint8))
    # im.save('tile.jpg')
    # plt.imshow(tile[0].numpy().transpose(1, 2, 0))
    # plt.show()
