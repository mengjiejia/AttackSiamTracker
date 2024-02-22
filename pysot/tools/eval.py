import os
import sys
import time
import argparse
import json
import functools

import numpy as np

sys.path.append("./")

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from pysot.toolkit.datasets import UAV10Dataset, UAV20Dataset, VISDRONED2018Dataset, V4RDataset, UAVDTDataset
from pysot.toolkit.evaluation import OPEBenchmark
from pysot.toolkit.visualization import draw_success_precision

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    parser.add_argument('--dataset_dir', default='', type=str, help='dataset root directory')
    parser.add_argument('--dataset', default='V4RFlight112',type=str, help='dataset name')
    #parser.add_argument('--dataset', default='UAV123_10fps',type=str, help='dataset name')
    #parser.add_argument('--dataset', default='UAVDT', type=str, help='dataset name')
    parser.add_argument('--tracker_result_dir', default='', type=str, help='tracker result root')
    parser.add_argument('--trackers', default='attack_model', nargs='+')
    parser.add_argument('--vis', default='', dest='vis', action='store_true')
    parser.add_argument('--show_video_level', default=' ', dest='show_video_level', action='store_true')
    parser.add_argument('--num', default=1, type=int, help='number of processes to eval')
    parser.add_argument('--tracker_path_original',
                        default='original_results/SiamAPN/V4RFlight112/bbox',
                        type=str, help='tracker path')
    parser.add_argument('--tracker_path_attack',
                        default='attack_results/SiamAPN/V4RFlight112/bbox',
                        type=str, help='tracker path')
    parser.add_argument('--original',  action="store_true",
                    help='original tracking evaluation')
    parser.add_argument('--attack', action="store_true",
                    help='attack tracking evaluation')

    args = parser.parse_args()

    if args.original:
        tracker_dir = args.tracker_path_original
        trackers = ['original']
    elif args.attack:
        tracker_dir = args.tracker_path_attack
        trackers = ['attack']
    tracker_dir = args.tracker_path_attack
    trackers = ['attack']
    #tracker_dir = os.path.join(os.getcwd(), args.tracker_path_attack)
    # tracker_dir = args.tracker_path_attack

    # anno root for UAV112
    # root = os.path.join('/home/mengjie/UAV_Tracking/UAVTrack112', args.dataset)
    root = os.path.normpath(os.path.join(os.getcwd(), '../test_dataset', args.dataset))

    # anno root for UAV123
    #root = os.path.join('/media/mengjie/Data/Downloads', args.dataset)

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    # dump success and precision to json file, define json name first
    # my success
    # success_json = tracker_dir + '/' + trackers[0] + '/' + 'my_success.json'
    # precision_json = tracker_dir + '/' + trackers[0] + '/' + 'my_precision.json'

    # Ad2A success
    save_dir = os.path.join(os.getcwd(),'Evaluation' ,trackers[0])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    success_json = os.path.join(save_dir, 'success.json')
    precision_json = os.path.join(save_dir, 'precision.json')

    # if 'UAV10' in args.dataset:
    if 'UAV123_10fps' in args.dataset:
        dataset = UAV10Dataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=18):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=18):
                precision_ret.update(ret)
        # benchmark.show_result(success_ret, precision_ret,
        #                       show_video_level=args.show_video_level)

        s = []
        p = []
        tracker_name = trackers[0]
        for v in success_ret[tracker_name].keys():
            s1 = success_ret[tracker_name][v].tolist()
            s.append(s1)
            p1 = precision_ret[tracker_name][v].tolist()
            p.append(p1)

        # print(type(success_ret))
        with open(success_json, 'w') as file:
            json.dump(s, file, indent=1)

        with open(precision_json, 'w') as file:
            json.dump(p, file, indent=1)

        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)


    elif 'UAVDT' in args.dataset:
        dataset = UAVDTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=18):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=18):
                precision_ret.update(ret)
        # benchmark.show_result(success_ret, precision_ret,
        #                       show_video_level=args.show_video_level)

        s = []
        p = []
        tracker_name = trackers[0]


        for v in success_ret[tracker_name].keys():
            s1 = success_ret[tracker_name][v].tolist()
            s.append(s1)
            p1 = precision_ret[tracker_name][v].tolist()
            p.append(p1)


        # print(type(success_ret))
        with open(success_json, 'w') as file:
            json.dump(s, file, indent=1)

        with open(precision_json, 'w') as file:
            json.dump(p, file, indent=1)

        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)


    elif 'UAV20' in args.dataset:
        dataset = UAV20Dataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=18):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=18):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)


    elif 'VISDRONED' in args.dataset:
        dataset = VISDRONED2018Dataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=18):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=18):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)


    elif 'V4RFlight112' in args.dataset:

        dataset = V4RDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                                                trackers), desc='eval success', total=len(trackers), ncols=18):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                                                trackers), desc='eval precision', total=len(trackers), ncols=18):
                precision_ret.update(ret)
        # show results
        benchmark.show_result(success_ret, precision_ret,
                               show_video_level=args.show_video_level)
        s = []
        p = []
        tracker_name = trackers[0]
        for v in success_ret[tracker_name].keys():
            s1 = success_ret[tracker_name][v].tolist()
            s.append(s1)
            p1 = precision_ret[tracker_name][v].tolist()
            p.append(p1)

        # print(type(success_ret))
        with open(success_json, 'w') as file:
            json.dump(s, file, indent=1)

        with open(precision_json, 'w') as file:
            json.dump(p, file, indent=1)

        # print(dataset.attr.items())

        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)
