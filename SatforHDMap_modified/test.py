import argparse
import tqdm
import logging
import sys
import os

import torch
import numpy as np
from data.dataset import semantic_dataset_test
from data.const import NUM_CLASSES
from evaluation.iou import get_batch_iou
from postprocess.vectorize import vectorize
from model import get_model


SAMPLED_RECALLS = torch.linspace(0.1, 1, 10)
THRESHOLDS = [2, 4, 6]


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def test_iou(model, test_loader):
    model.eval()
    total_intersects = 0
    total_union = 0
    with torch.no_grad():
        for imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, prior_map in tqdm.tqdm(test_loader):
            semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                                                lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(),
                                                prior_map.cuda())
            semantic_gt = semantic_gt.cuda().float()
            intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
            total_intersects += intersects
            total_union += union
    return total_intersects / (total_union + 1e-7)


def main(args):
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    logfile = os.path.join(args.logdir, "{}_{}_{}".format(args.fusion_mode, args.version.split('-')[-1], os.path.split(args.modelf)[-1].split('.')[0]))
    logging.basicConfig(filename=logfile,
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
    }

    test_loader = semantic_dataset_test(args.version, args.dataroot, args.prior_map_root, data_conf, args.bsz, args.nworkers, is_newsplit=args.is_newsplit)
    model = get_model(args.model, data_conf, args, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.modelf).items()})

    # model.load_state_dict(torch.load(args.modelf), strict=False)
    model.cuda()
    iou = test_iou(model, test_loader)
    logger.info(f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--is_newsplit", action='store_true')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fusion_mode", type=str, default='seg-masked-atten', choices=['attention', 'swin-atten', 'deform-atten', 'masked-atten', 'seg-masked-atten'])
    parser.add_argument("--align_fusion", action='store_true')
    parser.add_argument('--satellite_img_h', type=int, default=200)
    parser.add_argument('--satellite_img_w', type=int, default=400)
    parser.add_argument("--logdir", type=str, default='./test_log')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='../nuScenes')
    parser.add_argument('--prior_map_root', type=str, default='./satmap/satellite_map_trainval')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini', 'v1.0-test'])

    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_cam')

    # testing config
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=10)

    # model config
    parser.add_argument('--modelf', type=str, default='./runs/model28.pt')

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)

    args = parser.parse_args()
    main(args)


# CUDA_VISIBLE_DIVICES=4 python test.py --dataroot /data3/nuscenes/v1.0-test/ --instance_seg --embedding_dim --modelf ./runs/model29.pt 