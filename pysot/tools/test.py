# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import random
import argparse
import json
import os
import cv2
import torch
import numpy as np
from scipy import ndimage
from attack_utils import adv_attack_search
from data_utils import tensor2img, GOT10k_dataset, img2tensor
from pysot.mypysot.models.model_builder_apn import ModelBuilderAPN
from pysot.mypysot.tracker.siamapn_tracker import SiamAPNTracker
from pysot.mypysot.utils.bbox import get_axis_aligned_bbox
from pysot.mypysot.utils.model_load import load_pretrain
from pysot.toolkit.datasets import DatasetFactory
from pysot.mypysot.core.config_apn import cfg
from pysot.toolkit.utils.statistics import overlap_ratio
import sys
from data_utils import normalize

parser = argparse.ArgumentParser(description='siamapn tracking')
parser.add_argument('--dataset', default='V4RFlight112', type=str,
                    help='datasets')
# parser.add_argument('--dataset', default='UAV123_10fps', type=str,
#                   help='datasets')
# parser.add_argument('--dataset', default='UAVDT', type=str,
#                      help='datasets')

parser.add_argument('--snapshot', default='../experiments/SiamAPN/model.pth',
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
parser.add_argument('--img_show',  action="store_true",
                    help='whether visualzie img')
parser.add_argument('--original',  action="store_true",
                    help='original tracking demo')
parser.add_argument('--attack', action="store_true",
                    help='attack tracking demo')
parser.add_argument('--comparison', action="store_true",
                    help='draw original and attack prediction')
args = parser.parse_args()

# from Setting import *
from Model_config_test import *
from Model_config_test_R import *

model_name = opt.model
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)


def original_tracking(img_show=True):
    # load config
    cfg.merge_from_file(args.config)
    model = ModelBuilderAPN()

    model = load_pretrain(model, args.snapshot).cuda().eval()
    tracker = SiamAPNTracker(model)
    dataset_root = os.path.normpath(os.path.join(os.getcwd(), '../test_dataset', args.dataset))
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1)
    model_name = 'SiamAPN'
    model_path = os.path.join('original_results', model_name, args.dataset)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    # OPE tracking
    # video is a tuple (img, groundtruth bounding box)
    for v_idx, video in enumerate(dataset):

        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []

        bbox_path = os.path.join(model_path, 'bbox')

        # create bbox directory
        if not os.path.isdir(bbox_path):
            os.makedirs(bbox_path)

        img_dir = os.path.join(model_path, video.name)

        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                pred_bbox = gt_bbox_
                pred_img = img
                tracker.init(img, gt_bbox_)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
            else:

                # normal image prediction
                # before
                outputs = tracker.track_base(img)
                # from here
                # x_crop, scale_z = tracker.get_x_crop(img)
                # outputs = tracker_.track(img_filter, x_crop)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()

            if idx > 0:
                # draw normal prediction in yellow color
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]),
                              (0, 255, 255), 2)

                # draw ground truth in green color
                if not np.isnan(gt_bbox).any():
                    gt_bbox = list(map(int, gt_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 2)

                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # save image
                img_name = os.path.join(img_dir, '{}.jpg'.format(idx))
                cv2.imwrite(img_name, img)
                if img_show:
                    cv2.imshow(video.name, img)

                    cv2.waitKey(1)
        toc /= cv2.getTickFrequency()

        # save results

        result_path = os.path.join(bbox_path, '{}.txt'.format(video.name))

        # write normal prediction bounding box
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')

        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))


def Ad2test(start=1, end=-1, img_show=False):
    # load config
    cfg.merge_from_file(args.config)
    model_attack = ModelBuilderAPN()

    # model attack
    model_attack = load_pretrain(model_attack, args.snapshot).cuda().eval()
    tracker_attack = SiamAPNTracker(model_attack)

    dataset_root = os.path.normpath(os.path.join(os.getcwd(), '../test_dataset', args.dataset))
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1)
    model_name = 'SiamAPN'

    model_path = os.path.join('attack_results', model_name, args.dataset)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    output_path = os.path.join(model_path, 'output.txt')
    bbox_path = os.path.join(model_path, 'bbox')

    # create bbox directory
    if not os.path.isdir(bbox_path):
        os.makedirs(bbox_path)

    for v_idx, video in enumerate(dataset):
        if end == -1:
            end = len(video)

        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        # if video.name not in name_list:
        #     continue

        toc = 0
        pred_bboxes_adv = []
        scores_adv = []

        track_times = []

        img_dir = os.path.join(model_path, video.name)

        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        print('Attack to ', video.name, ' from frame ', start, ' to frame ', end)

        for idx, (img, gt_bbox) in enumerate(video):

            if idx % 200 == 0:
                print('Testing ', video.name, ' img ', idx)
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                pred_bbox_adv = gt_bbox_
                pred_img = img

                tracker_attack.init(img, gt_bbox_)

                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes_adv.append([1])
                else:
                    pred_bboxes_adv.append(gt_bbox_)

            else:

                adv_full_img = img.copy()
                # attack model
                # save previous center position and size first as reference
                cur_cp_adv = tracker_attack.center_pos.copy()
                cur_size_adv = tracker_attack.size.copy()

                zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])

                if start <= idx <= end:
                    # zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
                    # the last item from output is x_crop before attack
                    outputs_adv, x_crop_adv, context_2_adv, context_1_adv, org_patch_size_adv, x_crop = tracker_attack.track_adv(
                        img, zhanbi, AdA)

                    a = torch.squeeze(x_crop_adv)

                    a = torch.permute(a, (1, 2, 0))
                    x_crop_adv_con = a.detach().cpu().numpy()
                    x_crop_adv_con = cv2.resize(x_crop_adv_con, (org_patch_size_adv[0], org_patch_size_adv[1]))

                    adv_full_img[context_1_adv[0]:context_1_adv[1], context_1_adv[2]:context_1_adv[3],
                    :] = x_crop_adv_con[
                         context_2_adv[
                             0]:
                         context_2_adv[
                             1],
                         context_2_adv[
                             2]:
                         context_2_adv[
                             3],
                         :]
                else:
                    x_crop_adv, scale_z, context_2_adv, context_1_adv, org_patch_size_adv = tracker_attack.get_x_crop(
                        img)
                    outputs_adv = tracker_attack.track(img, x_crop_adv)

                pred_bbox_adv = outputs_adv['bbox']
                pred_bboxes_adv.append(pred_bbox_adv)
                # save score for detection
                scores_adv.append(outputs_adv['best_score'])

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())

            if start <= idx <= end:
                # draw attack prediction in red color
                pred_bbox_adv = list(map(int, pred_bbox_adv))
                cv2.rectangle(img, (pred_bbox_adv[0], pred_bbox_adv[1]),
                              (pred_bbox_adv[0] + pred_bbox_adv[2], pred_bbox_adv[1] + pred_bbox_adv[3]), (0, 0, 255),
                              2)

                # draw ground truth in green color
                if not np.isnan(gt_bbox).any():
                    gt_bbox = list(map(int, gt_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 2)
                #
                cv2.putText(img, '#' + str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # save image
                img_name = os.path.join(img_dir, '{}.jpg'.format(idx))
                cv2.imwrite(img_name, img)
                if img_show:
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)

        toc /= cv2.getTickFrequency()

        tmp = sys.stdout
        sys.stdout = open(output_path, 'a')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))

        sys.stdout.close()  # <-comment out to see the output
        sys.stdout = tmp

        # save bbox
        result_path_adv = os.path.join(bbox_path, '{}.txt'.format(video.name))

        # write attack prediction bounding box
        with open(result_path_adv, 'w') as f:
            for x in pred_bboxes_adv:
                f.write(','.join([str(i) for i in x]) + '\n')


def compare_prediction(img_show=False):
    dataset_root = os.path.normpath(os.path.join(os.getcwd(), '../test_dataset', args.dataset))
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    model_name = 'SiamAPN'
    original_path = os.path.join('original_results', args.dataset, model_name)
    original_bbox_path = os.path.join(original_path, 'bbox')
    bbox_files = os.listdir(original_bbox_path)

    attack_path = os.path.join('attack_results', model_name, args.dataset)
    attack_bbox_path = os.path.join(attack_path, 'bbox')

    model_name = 'SiamAPN'

    model_path = os.path.join('Evaluation', model_name, args.dataset)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    for v_idx, video in enumerate(dataset):

        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        # if video.name not in name_list:
        #     continue
        v = video + '.txt'
        if v not in bbox_files:
            continue

        img_dir = os.path.join(model_path, video.name)

        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        # get original box
        o_bbox_list = []
        original_bbox_file = os.path.join(original_bbox_path, video + '.txt')
        with open(original_bbox_file) as f:
            original_bboxes = f.read().splitlines()

        for idx, line in enumerate(original_bboxes):
            o_bbox = line.strip().split(',')
            o_bbox_ = [int(float(o_bbox[0])), int(float(o_bbox[1])), int(float(o_bbox[2])), int(float(o_bbox[3]))]
            o_bbox_list.append(o_bbox_)

        # get attack box
        a_bbox_list = []
        attack_bbox_file = os.path.join(attack_bbox_path, video + '.txt')
        with open(attack_bbox_file) as f:
            attack_bboxes = f.read().splitlines()

        for idx, line in enumerate(attack_bboxes):
            a_bbox = line.strip().split(',')
            a_bbox_ = [int(float(a_bbox[0])), int(float(a_bbox[1])), int(float(a_bbox[2])), int(float(a_bbox[3]))]
            a_bbox_list.append(a_bbox_)

        for idx, (img, gt_bbox) in enumerate(video):
            if idx == 0:
                continue

            # draw attack prediction in red color
            cv2.rectangle(img, (a_bbox_list[idx][0], a_bbox_list[idx][1]),
                              (a_bbox_list[idx][0] + a_bbox_list[idx][2], a_bbox_list[idx][1] + a_bbox_list[idx][3]), (0, 0, 255),
                              2)
            # draw original prediction in blue color
            cv2.rectangle(img, (o_bbox_list[idx][0], o_bbox_list[idx][1]),
                              (o_bbox_list[idx][0] + o_bbox_list[idx][2], o_bbox_list[idx][1] + o_bbox_list[idx][3]), (255, 0, 0),
                              2)

            # draw ground truth in green color
            if not np.isnan(gt_bbox).any():
                    gt_bbox = list(map(int, gt_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 2)

            cv2.putText(img, '#' + str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            # save image
            img_name = os.path.join(img_dir, '{}.jpg'.format(idx))
            cv2.imwrite(img_name, img)
            if img_show:
                cv2.imshow(video.name, img)
                cv2.waitKey(1)


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

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '/home/mengjie/UAV_Tracking/UAVTrack112', args.dataset)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1)
    for v_idx, video in enumerate(dataset):

        # if v_idx == 4:
        #    break
        # if video.name=='uav1_1':
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes_adv = []
        pred_bboxes = []
        scores_adv = []
        scores = []
        track_times = []
        iou_v = []

        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        img_dir = os.path.join(model_path, video.name)

        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        for idx, (img, gt_bbox) in enumerate(video):
            # squeeze color
            # img = squeeze_color(img, 4)

            # median_filter
            # img_filter = median_filter(img, 3)
            # remember to calculate the running time separately for normal and attack model
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                pred_bbox = gt_bbox_
                pred_bbox_adv = gt_bbox_
                pred_img = img
                tracker_normal.init(img, gt_bbox_)
                tracker_attack.init(img, gt_bbox_)
                iou_v.append(None)

                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
                    pred_bboxes_adv.append(pred_bbox)
            else:

                # normal image prediction
                # before
                # outputs = tracker.track_base(img)
                # from here
                x_crop, scale_z, context_2_normal, contex_1_normal, org_patch_size_normal = tracker_normal.get_x_crop(
                    img)
                outputs = tracker_normal.track(img, x_crop)

                pred_bbox_normal = outputs['bbox']
                pred_bboxes.append(pred_bbox_normal)
                scores.append(outputs['best_score'])

                # attack prediction
                zhanbi = (pred_bbox_adv[2] * pred_bbox_adv[3]) / (pred_img.shape[0] * pred_img.shape[1])
                # print(idx, 'zhanbi', zhanbi)

                # loss = tracker.check_loss(img, zhanbi, AdA)
                # print('loss', loss)
                # cv2.imshow(video.name, img_clean1)

                # outputs_adv, x_crop_adv = tracker_attack.track_adv(img, zhanbi, AdA)
                outputs_adv, x_crop_adv, context_2_adv, context_1_adv, org_patch_size_adv = tracker_attack.track_adv(
                    img, zhanbi, AdA)
                # pred_bbox = outputs_adv['bbox']
                pred_bbox_adv = outputs_adv['bbox']
                pred_bboxes_adv.append(pred_bbox_adv)
                scores_adv.append(outputs_adv['best_score'])

                bbox_normal = np.array(pred_bbox_normal)
                bbox_normal = np.expand_dims(bbox_normal, axis=0)
                bbox_adv = np.array(pred_bbox_adv)
                bbox_adv = np.expand_dims(bbox_adv, axis=0)

                iou_img = overlap_ratio(bbox_normal, bbox_adv)
                iou_v.append(iou_img[0])

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            # if args.vis and idx > 0:
            if idx > 0:
                # save image
                tracker_normal.save_img(x_crop, x_crop_adv, img_dir, idx)

        toc /= cv2.getTickFrequency()
        # save results

        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))


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

    for v_idx, video in enumerate(dataset):
        # video = (init_tensor, search_tensor, zhanbi, cur_folder)

        save_path = os.path.join('/media/mengjie/Data/Downloads', 'crop287_adv(test)')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        img_dir = os.path.join(save_path, video[3])

        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)

        AdA.set_input(video)
        AdA.forward()

        tensor_adv = AdA.search_adv255

        for idx in range(len(tensor_adv)):
            img_adv = tensor2img(tensor_adv[idx])
            frame_id = idx + 1
            cv2.imwrite(os.path.join(img_dir, '%08d.jpg' % frame_id), img_adv)


if __name__ == '__main__':

    if args.original:
        original_tracking(img_show=args.img_show)
    if args.attack:
        Ad2test(img_show=args.img_show)
    if args.comparison:
        compare_prediction(img_show=args.img_show)
