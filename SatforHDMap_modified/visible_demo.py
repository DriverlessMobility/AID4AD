import os
import numpy
import torch
import tqdm

from data.dataset import HDMapNetSemanticDataset
from model.pointpillar import PointPillarEncoder


if __name__ == '__main__':
    data_conf = {
        'num_channels': 3 + 1,
        'image_size': [128, 352],
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [4.0, 45.0, 1.0],
        'thickness': 5,
        'angle_class': 36,
    }

    dataset = HDMapNetSemanticDataset('v1.0-trainval', '/media/wjgao/Document/data/set/nuscenes', 'prior_map_trainval', data_conf=data_conf, is_train=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)

    lidar_model = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
    lidar_model.cuda()

    num = 0
    for imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, prior_map in tqdm.tqdm(
        data_loader):
        print(lidar_data.shape, prior_map.shape)
        out = lidar_model(lidar_data.cuda(), lidar_mask.cuda())
        print(out.cpu().shape)
        num += 1
        if num >=1:
            break