# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import cv2
import torch
import numpy as np
from scipy import ndimage
import json
from pysot.mypysot.models.model_builder_apn import ModelBuilderAPN
from pysot.mypysot.tracker.siamapn_tracker import SiamAPNTracker
from pysot.mypysot.utils.bbox import get_axis_aligned_bbox
from pysot.mypysot.utils.model_load import load_pretrain
from pysot.toolkit.datasets import DatasetFactory
from pysot.mypysot.core.config_apn import cfg
from data_utils import GOT10k_dataset, tensor2img
from statistics import mean
import matplotlib.pyplot as plt
from siamRPNPP import SiamRPNPP

parser = argparse.ArgumentParser(description='siamapn tracking')
parser.add_argument('--dataset', default='V4RFlight112', type=str,
                    help='datasets')
parser.add_argument('--snapshot', default='/home/mengjie/PycharmProjects/Ad2Attack/snapshot/general_model.pth',
                    type=str,
                    help='snapshot of models to eval')  # './snapshot/general_model.pth'
parser.add_argument('--config', default='../experiments/config.yaml', type=str,
                    help='config file')
parser.add_argument('--trackername', default='SiamAPN', type=str,
                    help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
                    help='eval one special video')
parser.add_argument('--vis', default='', action='store_true',
                    help='whether visualzie result')
parser.add_argument('--epoch', default=10, type=int,
                    help='training epoch for attack network')
args = parser.parse_args()
from Model_config_test import *
from Model_config_train import *

model_name = opt.model
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)


def squeeze_color(img, i):
    p = np.power(2, i) - 1
    image = img * p
    image = np.rint(image)
    image = image / p

    return image


def median_filter(img, s):
    image = ndimage.median_filter(img, size=s)

    return image


# training function for Ad2 attack networks
def train_AdA(epoch):
    # load config
    cfg.merge_from_file(args.config)
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    # train_rep = '/home/mengjie/Downloads/crop287'
    train_rep = os.path.normpath(os.path.join(os.getcwd(), '../training_dataset/got10k/crop287'))
    # create dataset
    # load got10k dataset
    dataset = GOT10k_dataset(train_rep)
    print('Start Training')
    loss_list = []
    for e in range(epoch):
        # loss for batches
        running_loss = 0.0
        # loss for epoch
        loss_e = []
        for v_idx, data in enumerate(dataset):
            AdA.set_input(data)
            AdA.optimize_parameters()
            running_loss += AdA.loss_A
            loss_e.append(AdA.loss_A.item())
            if v_idx % 200 == 199:  # print every 200 mini-batches
                print(f'[{e + 1}, {v_idx + 1:5d}] loss: {running_loss / 200:.3f}')

                running_loss = 0.0
        loss_list.append(mean(loss_e))
    AdA.save_networks(epoch)
    json.dump(loss_list, open(os.path.join(os.getcwd(), 'checkpoints/AdATTACK/loss.json'), 'w'))
    l = range(1, len(loss_list) + 1)
    plt.plot(l, loss_list, label='loss', marker='.', markersize=10)
    plt.xticks(range(1, len(loss_list) + 1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss plot')
    plt.savefig(os.path.join(os.getcwd(), 'checkpoints/AdATTACK/loss plot.jpg'), dpi=600)

    print('Finish Training')


def save_images():
    # load config
    # print(args.config)
    cfg.merge_from_file(args.config)
    model_normal = ModelBuilderAPN()
    model_attack = ModelBuilderAPN()

    # print(args.snapshot)
    model_normal = load_pretrain(model_normal, args.snapshot).cuda().eval()
    tracker_normal = SiamAPNTracker(model_normal)

    model_attack = load_pretrain(model_attack, args.snapshot).cuda().eval()
    tracker_attack = SiamAPNTracker(model_attack)

    # dataset_root = os.path.join(cur_dir, '/home/mengjie/UAV_Tracking/UAVTrack112', args.dataset)

    train_rep = '/media/mengjie/Data/Downloads/crop287_adv'
    # create dataset
    # load got10k dataset
    dataset = GOT10k_dataset(train_rep)
    print('Start Testing')
    save_path = os.path.join('/media/mengjie/Data/Downloads', 'crop287_adv(test)')

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for v_idx, video in enumerate(dataset):
        # video = (init_tensor, search_tensor, zhanbi, cur_folder)
        head_tail = os.path.split(video[3])

        img_dir = os.path.join(save_path, head_tail[1])

        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        with torch.no_grad():
            AdA.set_input(video)
            AdA.forward()

        tensor_adv = AdA.search_adv255

        for idx in range(len(tensor_adv)):
            img_adv = tensor2img(tensor_adv[idx])
            frame_id = idx + 1
            cv2.imwrite(os.path.join(img_dir, '%08d.jpg' % frame_id), img_adv)


# training function for recovery networks
# def train_AdA_R():
#     # load config
#     epoch = 50
#     cfg.merge_from_file(args.config)
#     cur_dir = os.path.dirname(os.path.realpath(__file__))
#     train_rep_clean = '/media/mengjie/Data/Downloads/crop287'
#     # train_rep_adv = '/media/mengjie/Data/Downloads/crop287_adv'
#     # train_rep_clean = '/home/mengjie/Downloads/crop287'
#     # train_rep_adv = '/home/mengjie/Downloads/crop287_adv'
#     # create dataset
#     # load got10k dataset
#     dataset_clean = GOT10k_dataset(train_rep_clean)
#     # dataset_adv = GOT10k_dataset(train_rep_adv)
#     print('Start Training')
#     # loss_list = []
#     for e in range(20, epoch, 1):
#         running_loss = 0.0
#         # loss_e = []
#         for v_idx, data_clean in enumerate(dataset_clean):
#             with torch.no_grad():
#                 AdA.set_input(data_clean)
#                 AdA.forward()
#
#             tensor_adv = AdA.search_adv255
#
#             AdA_R.set_input(data_clean)
#             AdA_R.set_input_adv(tensor_adv)
#             AdA_R.optimize_parameters_R()
#             running_loss += AdA_R.loss_R
#             if v_idx % 200 == 199:  # print every 200 mini-batches
#                 print(f'[{e + 1}, {v_idx + 1:5d}] loss: {running_loss / 200:.3f}')
#                 # loss_e.append(running_loss / 200)
#                 running_loss = 0.0
#         # loss_list.append(loss_e)
#         if e == 5:
#             AdA_R.save_networks(e)
#         elif e == 10:
#             AdA_R.save_networks(e)
#         elif e == 15:
#             AdA_R.save_networks(e)
#         elif e == 20:
#             AdA_R.save_networks(e)
#         elif e == 25:
#             AdA_R.save_networks(e)
#         elif e == 30:
#             AdA_R.save_networks(e)
#         elif e == 35:
#             AdA_R.save_networks(e)
#
#     AdA_R.save_networks(epoch)
#     # json.dump(loss_e, open('loss_R.json', 'w'))
#     print('Finished Training')


#
if __name__ == '__main__':

    train_AdA(args.epoch)
