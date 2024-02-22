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
from pysot.pysot.models.model_builder_apn import ModelBuilderAPN
from pysot.pysot.tracker.siamapn_tracker import SiamAPNTracker
from pysot.pysot.utils.bbox import get_axis_aligned_bbox
from pysot.pysot.utils.model_load import load_pretrain
from pysot.toolkit.datasets import DatasetFactory
from pysot.pysot.core.config_apn import cfg

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
args = parser.parse_args()
from Setting import *
from Model_config_test import *

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


def main():
    # load config
    # print(args.config)
    cfg.merge_from_file(args.config)
    model = ModelBuilderAPN()

    # print(args.snapshot)
    model = load_pretrain(model, args.snapshot).cuda().eval()
    tracker = SiamAPNTracker(model)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '/home/mengjie/UAV_Tracking/UAVTrack112', args.dataset)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0] + str(cfg.TRACK.w1)
    # OPE tracking
    # print('video name', len(dataset['air conditioning box1'][0]))
    # stop
    # video is a tuple (img, groundtruth bounding box)
    for v_idx, video in enumerate(dataset):
        # if v_idx == 20:
        #     break
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
            # img = median_filter(img, 4)
            # remember to calculate the running time separately for normal and attack model
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_img = img
                scores.append(None)
                if 'VOT2018-LT' == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
                    pred_bboxes_adv.append(pred_bbox)
            else:

                # normal image prediction
                # before
                # outputs = tracker.track_base(img)
                x_crop, scale_z = tracker.get_x_crop(img)
                outputs = tracker.track(img, x_crop)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])

                # attack prediction
                zhanbi = (pred_bbox[2] * pred_bbox[3]) / (pred_img.shape[0] * pred_img.shape[1])
                outputs_adv = tracker.track_adv(img, zhanbi, AdA)
                pred_bbox_adv = outputs_adv['bbox']
                pred_bboxes_adv.append(pred_bbox_adv)
                scores_adv.append(outputs_adv['best_score'])


            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            # if args.vis and idx > 0:
            if idx > 0:
                # draw normal prediction in yellow color
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)

                # draw attack prediction in red color
                pred_bbox_adv = list(map(int, pred_bbox_adv))
                cv2.rectangle(img, (pred_bbox_adv[0], pred_bbox_adv[1]),
                              (pred_bbox_adv[0] + pred_bbox_adv[2], pred_bbox_adv[1] + pred_bbox_adv[3]), (0, 0, 255), 3)


                # draw ground truth in green color
                if not np.isnan(gt_bbox).any():
                    gt_bbox = list(map(int, gt_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)

                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # save image
                # img_name = os.path.join(img_dir, '{}.jpg'.format(idx))
                # cv2.imwrite(img_name, img)
                #
                # cv2.imshow(video.name, img)
                if video.name == 'basketball player1' and idx == 225:
                    cv2.waitKey(100)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        # save results

        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        normal_img_dir = os.path.join(model_path, 'normal')
        adv_img_dir = os.path.join(model_path, 'attack')

        if not os.path.isdir(normal_img_dir):
            os.makedirs(normal_img_dir)
        if not os.path.isdir(adv_img_dir):
            os.makedirs(adv_img_dir)
        result_path_normal = os.path.join(normal_img_dir, '{}.txt'.format(video.name))
        result_path_adv = os.path.join(adv_img_dir, '{}.txt'.format(video.name))

        # write normal prediction bounding box
        with open(result_path_normal, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')

        # write attack prediction bounding box
        with open(result_path_adv, 'w') as f:
            for x in pred_bboxes_adv:
                f.write(','.join([str(i) for i in x]) + '\n')

        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx + 1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
