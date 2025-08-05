import os
import numpy as np
import sys
import logging
from time import time
from tensorboardX import SummaryWriter
import argparse

import torch
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from loss import SimpleLoss, DiscriminativeLoss

from data.dataset import semantic_dataset, semantic_dataset_dist
from data.const import NUM_CLASSES
from evaluation.iou import get_batch_iou
from evaluation.angle_diff import calc_angle_diff
from model import get_model
from evaluate import onehot_encoding, eval_iou


def write_log(writer, ious, title, counter):
    writer.add_scalar(f'{title}/iou', torch.mean(ious[1:]), counter)

    for i, iou in enumerate(ious):
        writer.add_scalar(f'{title}/class_{i}/iou', iou, counter)


# def setup():
#     dist.init_process_group('nccl')


# def cleanup():
#     dist.destroy_process_group()


def train(args):
    rank = 0
    if args.multi_gpu:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        pid = os.getpid()
        print(f'current pid:{pid}')
        print(f'current rank:{rank}')
        device_id = rank % torch.cuda.device_count()

        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()

    if rank == 0:
        if not os.path.exists(args.logdir):
            os.mkdir(args.logdir)
        logging.basicConfig(filename=os.path.join(args.logdir, "results.log"),
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

    if args.multi_gpu:
        train_loader, val_loader, train_sampler, val_sampler = semantic_dataset_dist(args.version, args.dataroot, args.prior_map_root, data_conf, args.bsz, args.nworkers, args.multi_gpu, (args.satellite_img_w, args.satellite_img_h), is_newsplit=args.is_newsplit)
    else:
        train_loader, val_loader = semantic_dataset(args.version, args.dataroot, args.prior_map_root, data_conf, args.bsz,
                                                args.nworkers, args.multi_gpu, (args.satellite_map_w, args.satellite_map_h), is_newsplit=args.is_newsplit)

    model = get_model(args.model, data_conf, args, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)

    if args.model_root != '':
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.model_root).items()})
        # model.load_state_dict(torch.load(args.model_root))
    if args.backbone_model != '':
        model_dict = model.state_dict()
        dict = {k.replace('module.', ''): v for k, v in torch.load(args.backbone_model).items()}
        dict = {k : v for k, v in dict.items() if any(k.startswith(prefix) for prefix in ['seg', 'prior_map_encoder', 'view_fusion', 'camencode', 'bevencoder'])}
        model_dict.update(dict)
        # dict = {k : v for k, v in dict}
        model.load_state_dict(model_dict)

    if args.finetune:
        model.load_state_dict(torch.load(args.modelf), strict=False)
        for name, param in model.named_parameters():
            if 'bevencode.up' in name or 'bevencode.layer3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    if args.multi_gpu:
        model = model.to(device_id)
        model = DistributedDataParallel(model, device_ids=[device_id], find_unused_parameters=True)
    else:
        model.cuda()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = StepLR(opt, 10, 0.1)
    if rank==0:
        writer = SummaryWriter(logdir=args.logdir)

    loss_fn = SimpleLoss(args.pos_weight).to(device_id)
    embedded_loss_fn = DiscriminativeLoss(args.embedding_dim, args.delta_v, args.delta_d).to(device_id)
    direction_loss_fn = torch.nn.BCELoss(reduction='none').to(device_id)
    model.train()
    counter = 0
    last_idx = len(train_loader) - 1
    for epoch in range(args.nepochs):
        if args.multi_gpu:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans,
                     yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, prior_map) in enumerate(train_loader):
    
            t0 = time()
            opt.zero_grad()

            semantic, embedding, direction = model(imgs.to(device_id), trans.to(device_id), rots.to(device_id), intrins.to(device_id),
                                                   post_trans.to(device_id), post_rots.to(device_id), lidar_data.to(device_id),
                                                   lidar_mask.to(device_id), car_trans.to(device_id), yaw_pitch_roll.to(device_id),
                                                   prior_map.to(device_id))

            semantic_gt = semantic_gt.cuda().float()
            instance_gt = instance_gt.cuda()
            seg_loss = loss_fn(semantic, semantic_gt)
            if args.instance_seg:
                var_loss, dist_loss, reg_loss = embedded_loss_fn(embedding, instance_gt)
            else:
                var_loss = 0
                dist_loss = 0
                reg_loss = 0

            if args.direction_pred:
                direction_gt = direction_gt.cuda()
                lane_mask = (1 - direction_gt[:, 0]).unsqueeze(1)
                direction_loss = direction_loss_fn(torch.softmax(direction, 1), direction_gt)
                direction_loss = (direction_loss * lane_mask).sum() / (lane_mask.sum() * direction_loss.shape[1] + 1e-6)
                angle_diff = calc_angle_diff(direction, direction_gt, args.angle_class)
            else:
                direction_loss = 0
                angle_diff = 0

            final_loss = seg_loss * args.scale_seg + var_loss * args.scale_var + dist_loss * args.scale_dist + direction_loss * args.scale_direction
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0 and rank == 0:
                intersects, union = get_batch_iou(onehot_encoding(semantic), semantic_gt)
                iou = intersects / (union + 1e-7)
                logger.info(f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{last_idx}]    "
                            f"Time: {t1-t0:>7.4f}    "
                            f"Loss: {final_loss.item():>7.4f}    "
                            f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

                write_log(writer, iou, 'train', counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
                writer.add_scalar('train/seg_loss', seg_loss, counter)
                writer.add_scalar('train/var_loss', var_loss, counter)
                writer.add_scalar('train/dist_loss', dist_loss, counter)
                writer.add_scalar('train/reg_loss', reg_loss, counter)
                writer.add_scalar('train/direction_loss', direction_loss, counter)
                writer.add_scalar('train/final_loss', final_loss, counter)
                writer.add_scalar('train/angle_diff', angle_diff, counter)

        iou = eval_iou(model, val_loader)
        if rank==0:
            logger.info(f"EVAL[{epoch:>2d}]:    "
                        f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")

            write_log(writer, iou, 'eval', counter)
            model_name = os.path.join(args.logdir, f"model{epoch}.pt")
            torch.save(model.module.state_dict(), model_name)
            logger.info(f"{model_name} saved")
        model.train()
        sched.step()
    # cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet training.')

    parser.add_argument('--is_newsplit', action='store_true')
    # multi_gpu config
    parser.add_argument("--multi_gpu", type=bool, default=True)
    parser.add_argument("--local_rank", type=int, default=0)

    # fusion mode config
    # parser.add_argument("--fusion_mode", type=str, default="concat", choices=['concat', 'swin-atten', 'atten', 'new-atten', 'deform-atten', 'masked-atten', 'masked-atten-seg', 'new-atten-masked', 'new-atten-masked-seg', 'new-deform-t'])
    parser.add_argument("--fusion_mode", type=str, default='seg-masked-atten', choices=['attention', 'swin-atten', 'deform-atten', 'masked-atten', 'seg-masked-atten'])
    parser.add_argument('--align_fusion', action='store_true')

    # logging config
    parser.add_argument("--logdir", type=str, default='./runs/original')

    # nuScenes config
    #parser.add_argument('--dataroot', type=str, default='./dataset/nuScenes_mini')
    #parser.add_argument('--prior_map_root', type=str, default='./satmap/prior_map_mini')
    #parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--dataroot', type=str, default='../nuScenes')
    parser.add_argument('--prior_map_root', type=str, default='./satmap/satellite_map_trainval')
    parser.add_argument('--version', type=str, default='v1.0-trainval', choices=['v1.0-trainval', 'v1.0-mini'])

    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_cam')

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--nworkers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)

    # checkpoint config
    parser.add_argument('--model_root', type=str, default='')
    parser.add_argument('--backbone_model', type=str, default='')

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

    # map prior config
    parser.add_argument('--map_prior', action='store_true')
    parser.add_argument('--satellite_img_h', type=int, default=200)
    parser.add_argument('--satellite_img_w', type=int, default=400)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)

    args = parser.parse_args()
    train(args)
