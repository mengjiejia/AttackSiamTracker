# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import gc
import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from pysot.pysot.utils.lr_scheduler_apn import build_lr_scheduler
from pysot.pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.pysot.utils.distributed import dist_init, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from pysot.pysot.utils.model_load import load_pretrain, restore_from
from pysot.pysot.utils.average_meter import AverageMeter
from pysot.pysot.utils.misc import describe, commit
from pysot.pysot.models.model_builder_apn import ModelBuilderAPN

from pysot.pysot.datasets.dataset_apn import TrkDataset
from pysot.pysot.core.config_apn import cfg
from Setting import project_path
from attack_utils import adv_attack_search, get_clean_1, get_losses
from Model_config_test import *
from pysot.pysot.tracker.siamapn_tracker import SiamAPNTracker
import cv2
from Setting import *
from Model_config_test import *

model_name = opt.model
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)

# /home/mengjie/PycharmProjects/Ad2Attack/pysot/training_dataset
logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='SiamAPN tracking')
# parser.add_argument('--cfg', type=str, default='../experiments/config.yaml',
#                     help='configuration of tracking')
parser.add_argument('--cfg', type=str, default='../experiments/SiamAPN/config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# back up build_data_loader

# def build_data_loader():
#     logger.info("build train dataset")
#     # train_dataset
#     train_dataset = TrkDataset()
#     logger.info("build dataset done")
#
#     train_sampler = None
#     if get_world_size() > 1:
#         train_sampler = DistributedSampler(train_dataset)
#     train_loader = DataLoader(train_dataset,
#                               batch_size=cfg.TRAIN.BATCH_SIZE,
#                               num_workers=cfg.TRAIN.NUM_WORKERS,
#                               pin_memory=True,
#                               sampler=train_sampler)
#     return train_loader

def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset()
    logger.info("build dataset done")

    train_sampler = None
    # if get_world_size() > 1:
    #     train_sampler = DistributedSampler(train_dataset)
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=cfg.TRAIN.BATCH_SIZE,
    #                           num_workers=cfg.TRAIN.NUM_WORKERS,
    #                           pin_memory=True,
    #                           sampler=train_sampler)
    print('cfg.TRAIN.NUM_WORKERS,',cfg.TRAIN.NUM_WORKERS)
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler,
                              shuffle=False)
    print('train_loader', len(train_loader))
    return train_loader


def build_opt_lr(model, current_epoch=0):
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    else:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                            model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.new.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.grader.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]
    
    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    cur_lr = lr_scheduler.get_cur_lr()
    rank = get_rank()

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // \
        cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model.module)))
    end = time.time()
    for idx, data in enumerate(train_loader):
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch

            if get_rank() == 0:
                torch.save(
                        {'epoch': epoch,
                          'state_dict': model.module.state_dict(),
                          'optimizer': optimizer.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint00_e%d.pth' % (epoch))
               
            if epoch == cfg.TRAIN.EPOCH:
                
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch: {}'.format(epoch+1))

        tb_idx = idx
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch+1, pg['lr']))
                if rank == 0:
                    tb_writer.add_scalar('lr/group{}'.format(idx+1),
                                          pg['lr'], tb_idx)

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        outputs = model(data)
        loss = outputs['total_loss'].mean()
        if is_valid_number(loss.data.item()):
            optimizer.zero_grad()
            loss.backward()
            reduce_gradients(model)

            # clip gradient
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)
        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.mean().data.item())

        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, tb_idx)

            if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:
                info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                            epoch+1, (idx+1) % num_per_epoch,
                            num_per_epoch, cur_lr)
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                                getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                                getattr(average_meter, k))
                logger.info(info)
                print_speed(idx+1+start_epoch*num_per_epoch,
                            average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch)
        end = time.time()
        

def get_train_loss_attack(train_loader, model, tb_writer):
    rank = get_rank()

    average_meter = AverageMeter() \

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    # print('cfg.TRAIN.EPOCH', cfg.TRAIN.EPOCH)
    # print('cfg.TRAIN.BATCH_SIZE', cfg.TRAIN.BATCH_SIZE)
    # print('len(train_loader.dataset) // cfg.TRAIN.EPOCH', len(train_loader.dataset) // cfg.TRAIN.EPOCH)
    # print('len(train_loader.dataset)', len(train_loader.dataset))
    # num_per_epoch = len(train_loader.dataset) // \
    #     cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    num_per_epoch = len(train_loader.dataset)

    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    # logger.info("model\n{}".format(describe(model.module)))
    end = time.time()
    for idx, data in enumerate(train_loader):
        # print('epoch', epoch)
        # print('idx', idx)
        # print('idx // num_per_epoch + start_epoch:', idx // num_per_epoch + start_epoch)
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch
        print('epoch', epoch)
        print('data len', len(data))
        # print('template', len(data['template']))
        # print('search', len(data['search']))
        # print('video_name', data['video_name'])


        tb_idx = idx

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        # print('data size', len(data))
        # print('search shape',data['search'].shape)
        # print('template shape',data['template'].shape)
        # show the image

        # tem = data[0]['template'][0] / 255
        # tem = tem.permute(1, 2, 0)
        # tem = tem.cpu().detach().numpy()
        # cv2.imshow('search',tem)
        # cv2.waitKey()
        # stop
        # show search images
        # for i in range(len(data)):
        #     item = data[i]['search'][0]/255
        #     img = item.permute(1, 2, 0)
        #     img = img.cpu().detach().numpy()
        #     # print('img shape', img)
        #     cv2.imshow('search',img)
        #     cv2.waitKey(100)

        # get zhanbi and write to json file
        print('tem shape', data[0]['template'][0].shape)
        print('search shape', data[0]['search'][0].shape)
        stop



        with torch.no_grad():
            outputs = model.forward(data)
            stop

        x_crop, scale_z = model.get_x_crop(data['search'])
        '''Adversarial Attack'''
        zhanbi = zhanbi
        # print(zhanbi)
        # print("x_crop:",x_crop.shape)
        x_crop_adv = adv_attack_search(x_crop, zhanbi, GAN)

        loss = outputs['total_loss'].mean()

        return loss




        # batch_time = time.time() - end
        # batch_info = {}
        # batch_info['batch_time'] = average_reduce(batch_time)
        # batch_info['data_time'] = average_reduce(data_time)
        # for k, v in sorted(outputs.items()):
        #     batch_info[k] = average_reduce(v.mean().data.item())
        #
        # average_meter.update(**batch_info)
        #
        # end = time.time()

def get_train_loss(train_loader, model, tb_writer):
    rank = get_rank()

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // \
        cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    end = time.time()
    print(len(train_loader))
    for idx, data in enumerate(train_loader):
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch
        tb_idx = idx

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

        # to see the input data is frame by frame or video by video
        # print('data size', len(data))

        with torch.no_grad():
            outputs = model.forward(data)

        loss = outputs['total_loss'].mean()

        return loss

def get_zhanbi(train_loader, model, AdA):
    num_per_epoch = len(train_loader.dataset)

    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    for idx, data in enumerate(train_loader):
        if epoch != idx // num_per_epoch + start_epoch:
            epoch = idx // num_per_epoch + start_epoch
        print('epoch', epoch)
        print('data len', len(data))
        l = len(data)
        # for i in range(l):





        # get zhanbi and write to json file
        print('tem shape', data[0]['template'][0].shape)
        print('search shape', data[0]['search'][0].shape)
        stop



        with torch.no_grad():
            outputs = model.forward(data)
            stop

        x_crop, scale_z = model.get_x_crop(data['search'])
        '''Adversarial Attack'''
        zhanbi = zhanbi
        # print(zhanbi)
        # print("x_crop:",x_crop.shape)
        x_crop_adv = adv_attack_search(x_crop, zhanbi, GAN)

        loss = outputs['total_loss'].mean()

        return loss



def main_backup():
    rank, world_size = dist_init()
    # rank = 0
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                              os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                              logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = ModelBuilderAPN().train()
    dist_model = nn.DataParallel(model).cuda()

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../pretrained_models/', cfg.BACKBONE.PRETRAINED)
        load_pretrain(model.backbone, backbone_path)

    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build dataset loader
    train_loader = build_data_loader()

    # build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(dist_model.module,
                                            cfg.TRAIN.START_EPOCH)

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../pretrained_models/')
        model.backbone.load_state_dict(torch.load(backbone_path+cfg.TRAIN.PRETRAINED))

    
    dist_model = nn.DataParallel(model)
    
    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)

def main():
    rank, world_size = dist_init()
    # rank = 0
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)

    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                              os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                              logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = ModelBuilderAPN().train()
    dist_model = nn.DataParallel(model).cuda()

    cfg_file = os.path.join(project_path, 'pysot/experiments/SiamAPN/config.yaml') # replace siamapn with SiamAPN
    snapshot = os.path.join(project_path, 'pysot/experiments/SiamAPN/model.pth')
    # load config
    cfg.merge_from_file(cfg_file)
    # create model
    model = ModelBuilderAPN()# A Neural Network.(a torch.nn.Module)
    # load model
    model = load_pretrain(model, snapshot).cuda().eval()
    # become siam apn tracker to calculate zhanbi
    # model = SiamAPNTracker(model)

    # load pretrained backbone weights, it is the same as above
    # if cfg.BACKBONE.PRETRAINED:
    #     cur_path = os.path.dirname(os.path.realpath(__file__))
    #     backbone_path = os.path.join('/home/mengjie/PycharmProjects/Ad2Attack/pysot/pretrained_models', cfg.BACKBONE.PRETRAINED)
    #     # print('backbone_path', backbone_path)
    #     load_pretrain(model.backbone, backbone_path)

    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build dataset loader
    train_loader = build_data_loader()
    # build optimizer and lr_scheduler
    # optimizer, lr_scheduler = build_opt_lr(dist_model.module,
    #                                         cfg.TRAIN.START_EPOCH)

    # resume training
    # if cfg.TRAIN.RESUME:
    #     logger.info("resume from {}".format(cfg.TRAIN.RESUME))
    #     assert os.path.isfile(cfg.TRAIN.RESUME), \
    #         '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
    #     model, optimizer, cfg.TRAIN.START_EPOCH = \
    #         restore_from(model, optimizer, cfg.TRAIN.RESUME)

    # load pretrain, it is the same as above pretrain
    # if True:
    # # if cfg.TRAIN.PRETRAINED:
    #     cur_path = os.path.dirname(os.path.realpath(__file__))
    #     # backbone_path = os.path.join(cur_path, '../pretrained_models/')
    #     backbone_path = os.path.join('/home/mengjie/PycharmProjects/Ad2Attack/pysot/pretrained_models', cfg.BACKBONE.PRETRAINED)
    #     model.backbone.load_state_dict(torch.load(backbone_path+cfg.TRAIN.PRETRAINED))


    dist_model = nn.DataParallel(model)


    logger.info("model prepare done")

    # model in test mode to get zhanbi

    # get_zhanbi(train_loader, dist_model, AdA)


    # start training
    loss_normal = get_train_loss_attack(train_loader, dist_model, tb_writer)
    # loss = get_train_loss(train_loader, model, tb_writer)
    stop
    print('loss_normal', loss_normal)
    # loss attack
    loss_attack = get_train_loss_attack(train_loader, dist_model, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
