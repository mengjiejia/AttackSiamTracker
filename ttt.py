import glob
import os
import re
import sys
import numpy as np
import cv2
import shutil
import json
from statistics import mean
import matplotlib.pyplot as plt


# rename the file name of prediction bbox file
def re_file():
    folder_name = 'Original'
    path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/Experiments/Pred_bbox'
    name_list = os.listdir(path + '/' + folder_name)
    name_list.sort()
    name_list.remove('attack')
    name_list.remove('normal')

    a = len(name_list)
    sour = path + '/' + folder_name + '/'
    adv_dest = path + '/' + folder_name + '/attack/'
    normal_dest = path + '/' + folder_name + '/normal/'

    for jj in range(a):
        b = str(name_list[jj])
        print(b)
        current_file_name = path + '/' + folder_name + '/' + b
        if 'adv' in b:
            new_file_name = path + '/' + folder_name + '/' + b.replace('_adv', '')
            os.rename(current_file_name, new_file_name)
            shutil.move(new_file_name, adv_dest)
        else:
            new_file_name = path + '/' + folder_name + '/' + b.replace('_normal', '')
            os.rename(current_file_name, new_file_name)
            shutil.move(new_file_name, normal_dest)


# draw comparison of success and precision in different tracker

# get frame index
def get_frame_index(success, precision, threshold_s, threshold_p):
    n_videos = len(success)
    frame_index = []
    for i in range(n_videos):
        w = []
        for j in range(len(success[i])):
            if success[i][j] >= threshold_s and precision[i][j] <= threshold_p:
                w.append(j)
        frame_index.append(w)

    return frame_index


# check drift of success and precision
def check_drift(success_1, precision_1, success_2, precision_2, frame_index):
    n_videos = len(success_1)
    drift_s = []
    drift_p = []
    for i in range(n_videos):
        ds = []
        dp = []
        for j in frame_index[i]:
            ds.append(abs(success_1[i][j] - success_2[i][j]))
            dp.append(abs(precision_1[i][j] - precision_2[i][j]))
        drift_s.append(ds)
        drift_p.append(dp)

    return drift_s, drift_p


# write frame index to json file
def frame_index2json(tracker):
    path_normal = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/Experiments/Pred_bbox/' + tracker + '/normal'
    path_attack = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/Experiments/Pred_bbox/' + tracker + '/attack'

    # normal
    s_json_file_normal = path_normal + '/my_success.json'
    p_json_file_normal = path_normal + '/my_precision.json'
    index_file_normal = path_normal + '/frame_index.json'

    # attack
    s_json_file_attack = path_attack + '/my_success.json'
    p_json_file_attack = path_attack + '/my_precision.json'
    index_file_attack = path_attack + '/frame_index.json'

    with open(s_json_file_normal) as f:
        success_normal = json.load(f)

    with open(p_json_file_normal) as f:
        precision_normal = json.load(f)

    with open(s_json_file_attack) as f:
        success_attack = json.load(f)

    with open(p_json_file_attack) as f:
        precision_attack = json.load(f)

    threshold_s = 0.6
    threshold_p = 25  # before is 15

    frame_index_normal = get_frame_index(success_normal, precision_normal, threshold_s, threshold_p)
    with open(index_file_normal, 'w') as file:
        json.dump(frame_index_normal, file, indent=1)

    frame_index_attack = get_frame_index(success_attack, precision_attack, threshold_s, threshold_p)
    with open(index_file_attack, 'w') as file:
        json.dump(frame_index_attack, file, indent=1)


# read success and precision results from json files
def read_data(tracker1, tracker2, attack):
    # the frame index is from tracker 1 by default
    if attack:
        path_1 = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/Experiments/Pred_bbox/' + tracker1 + '/attack'
        path_2 = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/Experiments/Pred_bbox/' + tracker2 + '/attack'
    else:
        path_1 = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/Experiments/Pred_bbox/' + tracker1 + '/normal'
        path_2 = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/Experiments/Pred_bbox/' + tracker2 + '/normal'

    # data1
    s_json_file_1 = path_1 + '/success.json'
    p_json_file_1 = path_1 + '/precision.json'
    index_file = path_1 + '/frame_index.json'

    # data2
    s_json_file_2 = path_2 + '/success.json'
    p_json_file_2 = path_2 + '/precision.json'

    with open(s_json_file_1) as f:
        success_1 = json.load(f)

    with open(p_json_file_1) as f:
        precision_1 = json.load(f)

    with open(s_json_file_2) as f:
        success_2 = json.load(f)

    with open(p_json_file_2) as f:
        precision_2 = json.load(f)

    with open(index_file) as f:
        frame_index = json.load(f)

    return success_1, precision_1, success_2, precision_2, frame_index


def read_my_eval_data(tracker1):
    # the frame index is from tracker 1 by default

    path_1 = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/Experiments/Pred_bbox/' + tracker1 + '/normal'
    path_2 = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/Experiments/Pred_bbox/' + tracker1 + '/attack'

    # data1 is normal
    s_json_file_1 = path_1 + '/my_success.json'
    p_json_file_1 = path_1 + '/my_precision.json'
    index_file_1 = path_1 + '/frame_index.json'

    # data2 is under attack
    s_json_file_2 = path_2 + '/my_success.json'
    p_json_file_2 = path_2 + '/my_precision.json'
    index_file_2 = path_2 + '/frame_index.json'

    with open(s_json_file_1) as f:
        success_1 = json.load(f)

    with open(p_json_file_1) as f:
        precision_1 = json.load(f)

    with open(s_json_file_2) as f:
        success_2 = json.load(f)

    with open(p_json_file_2) as f:
        precision_2 = json.load(f)

    with open(index_file_1) as f:
        frame_index_1 = json.load(f)

    with open(index_file_2) as f:
        frame_index_2 = json.load(f)

    return success_1, precision_1, success_2, precision_2, frame_index_1, frame_index_2


def plot_evl_n_a(success_1, precision_1, success_2, precision_2):
    l = range(len(success_1))
    # print(l)
    success_1 = np.array(success_1)
    success_2 = np.array(success_2)
    precision_1 = np.array(precision_1)
    precision_2 = np.array(precision_2)
    s1 = []
    s2 = []
    p1 = []
    p2 = []
    for i in range(len(success_1)):
        s1.append(np.mean(success_1[i]))
        s2.append(np.mean(success_2[i]))
        p1.append(np.mean(precision_1[i]))
        p2.append(np.mean(precision_2[i]))

    c_s = []
    c_p = []
    for i in range(len(success_1)):
        c_s.append(abs(s1[i] - s2[i]))
        c_p.append(abs(p1[i] - p2[i]))

    plt.plot(l, s1, label='normal success rate')
    plt.plot(l, s2, label='attack success rate')
    plt.xlabel('video index')
    plt.ylabel('value')
    plt.title('success rate comparison')
    plt.savefig('success rate comparison.jpg', dpi=600)
    plt.show()

    plt.plot(l, p1, label='normal precision')
    plt.plot(l, p2, label='attack precision')
    plt.xlabel('video index')
    plt.ylabel('value')
    plt.title('precision comparison')
    plt.savefig('precision comparison.jpg', dpi=600)
    plt.show()

    plt.plot(l, c_s, label='success rate difference')
    plt.xlabel('video index')
    plt.ylabel('value')
    plt.title('success rate difference')
    plt.savefig('success rate difference.jpg', dpi=600)
    plt.show()

    plt.plot(l, c_p, label='precision difference')
    plt.xlabel('video index')
    plt.ylabel('value')
    plt.title('precision difference')
    plt.savefig('precision difference.jpg', dpi=600)
    plt.show()


def cal_pos_rate(iou, score, threshold_iou, threshold_score):
    # in easy test case, all frames are under attack
    positive = 0
    positive_list = []
    for idx, item in enumerate(iou):
        # if item < threshold_iou and abs(score[idx]) > threshold_score:
        if abs(score[idx]) > threshold_score:
            positive = positive + 1
            positive_list.append(idx)

    positive_rate = positive / len(iou)

    return positive_rate


def cal_false_pos_rate(iou, score, threshold_iou, threshold_score):
    # in easy test case, all frames are under attack
    f_positive = 0
    f_positive_list = []
    for idx, item in enumerate(iou):
        # if item < threshold_iou and abs(score[idx]) > threshold_score:
        if abs(score[idx]) > threshold_score:
            f_positive = f_positive + 1
            f_positive_list.append(idx)

    f_positive_rate = f_positive / len(iou)

    return f_positive_rate


def read_iou(folder, attack, threshold_iou, threshold_score):
    root_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/validation'
    test_path = os.path.join(root_path, folder)
    video_ls = os.listdir(test_path)
    video_ls.remove('output.txt')
    video_ls.remove('bbox')

    if attack:
        tpr_f_a = []
        for video in video_ls:
            if video == 'normal_idx':
                continue
            v_path = os.path.join(test_path, video)
            filter_attack = os.path.join(v_path, 'filter_attack')
            # normal_attack = os.path.join(v_path, 'normal_attack')
            iou_f_a_path = os.path.join(filter_attack, 'iou_f_a.json')
            score_diff_f_a_path = os.path.join(filter_attack, 'score_diff_f_a.json')
            # iou_n_a_path = os.path.join(normal_attack, 'iou_n_a.json')
            iou_f_a_file = open(iou_f_a_path)
            iou_f_a = json.load(iou_f_a_file)

            score_diff_f_a_file = open(score_diff_f_a_path)
            score_diff_f_a = json.load(score_diff_f_a_file)


            f_a = cal_pos_rate(iou_f_a, score_diff_f_a, threshold_iou, threshold_score)
            tpr_f_a.append(f_a)

        avrg_tpr_f_a = np.average(tpr_f_a)

        return avrg_tpr_f_a, tpr_f_a



    # normal test
    else:
        fpr_f_n = []
        for video in video_ls:
            v_path = os.path.join(test_path, video)
            filter_normal = os.path.join(v_path, 'filter_normal')
            iou_f_n_path = os.path.join(filter_normal, 'iou_f_n.json')
            iou_f_n_file = open(iou_f_n_path)
            iou_f_n = json.load(iou_f_n_file)

            score_diff_f_n_path = os.path.join(filter_normal, 'score_diff_f_n.json')
            score_diff_f_n_file = open(score_diff_f_n_path)
            score_diff_f_n = json.load(score_diff_f_n_file)

            f_n = cal_false_pos_rate(iou_f_n, score_diff_f_n, threshold_iou, threshold_score)
            fpr_f_n.append(f_n)

        # json.dump(fpr_f_n, open(fpr_path, 'w'))
        avrg_fpr_f_n = np.average(fpr_f_n)

        return avrg_fpr_f_n, fpr_f_n


def cal_accuracy(filter_size):
    folder_attack = 'val_filter' + str(filter_size) + '_attack'
    folder_normal = 'val_filter' + str(filter_size) + '_normal'

    folder_tpr = 'filter' + str(filter_size) + '_tpr'
    folder_fpr = 'filter' + str(filter_size) + '_fpr'
    folder_accuracy = 'filter' + str(filter_size) + '_accuracy'
    folder = 'new_val_filter' + str(filter_size)

    threshold_ious = np.arange(0.6, 1.01, 0.01)
    threshold_scores = np.arange(0, 0.5, 0.01)

    accuracy = {}
    root_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/validation/Accuracy'
    folder_path = os.path.join(root_path, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    img_file = 'filter_' + str(filter_size) + '_accuracy comparison.jpg'
    img_path = os.path.join(folder_path, img_file)

    acc_file = folder_accuracy + '.json'
    acc_path = os.path.join(folder_path, acc_file)

    tpr_file = folder_tpr + '.json'
    fpr_file = folder_fpr + '.json'
    tpr_path = os.path.join(folder_path, tpr_file)
    fpr_path = os.path.join(folder_path, fpr_file)

    threshold_file = 'filter' + str(filter_size) + '_thresholds' + '.json'
    threshold_path = os.path.join(folder_path, threshold_file)

    # key a[1], a[4]
    tpr = []
    fpr = []

    tpr_save = []
    fpr_save = []

    thresholds = []
    for t_i in threshold_ious:
        for t_s in threshold_scores:
            thresholds.append([round(t_i, 2), round(t_s, 2)])
            avrg_tpr_f_a, tpr_f_a = read_iou(folder_attack, True, t_i, t_s)
            avrg_fpr_f_n, fpr_f_n = read_iou(folder_normal, False, t_i, t_s)

            results = [avrg_tpr_f_a, avrg_fpr_f_n]
            tpr.append(avrg_tpr_f_a)
            fpr.append(avrg_fpr_f_n)

            tpr_save.append(tpr_f_a)
            fpr_save.append(fpr_f_n)

            key = [round(t_i, 2), round(t_s, 2)]
            key = str(key)
            accuracy[key] = results

    json.dump(accuracy, open(acc_path, 'w'))
    json.dump(thresholds, open(threshold_path, 'w'))
    json.dump(tpr_save, open(tpr_path, 'w'))
    json.dump(fpr_save, open(fpr_path, 'w'))

    l = len(tpr)
    index = range(l)

    plt.figure(figsize=(18, 10))
    plt.plot(index, tpr, label='True positive rate')
    plt.plot(index, fpr, label='False positive rate')
    plt.grid(True)
    plt.legend()
    plt.xlabel('threshold index')
    plt.ylabel('value')
    plt.title('filter ' + str(filter_size) + ' accuracy comparison')
    plt.savefig(img_path, dpi=600)

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    for idx, i in enumerate(range(0, len(threshold_ious) * len(threshold_scores), len(threshold_scores))):

        s = i
        e = i + len(threshold_scores)

        if ix in range(s, e, 1):
            plt.figure(figsize=(18, 10))
            plt.scatter(fpr[ix], tpr[ix], marker='.', color='red', s=350, label='Best')
            plt.plot(fpr[s:e], tpr[s:e], marker='.', markersize=10)
            plt.text(fpr[ix], tpr[ix] - 0.01, '({:.2f}, {:.2f})'.format(fpr[ix], tpr[ix]), color='red', fontsize=15)
            plt.text(fpr[ix], tpr[ix] - 0.03, 'threshold={:.2f} '.format(thresholds[ix][1]), color='red',
                     fontsize=15)
            plt.grid(True)
            plt.legend(fontsize=15)
            plt.xlabel('False positive rate', fontsize=15)
            plt.ylabel('True positive rate', fontsize=15)
            plt.title('Filter ' + str(filter_size) + ' ' + ' ROC', fontsize=15)
            img_roc = 'Filter ' + str(filter_size) + ' ' + ' ROC.jpg'
            img_path_roc = os.path.join(folder_path, img_roc)
            plt.savefig(img_path_roc, dpi=600)


def cal_accuracy_combine(filter_size):
    folder_attack1 = 'val_filter' + str(filter_size) + '_attack'
    folder_attack2 = 'val_filter' + str(filter_size) + '_attack_end'
    folder_attack3 = 'val_filter' + str(filter_size) + '_attack_twice'
    folder_normal = 'val_filter' + str(filter_size) + '_normal'

    folder_tpr = 'filter' + str(filter_size) + '_tpr'
    folder_fpr = 'filter' + str(filter_size) + '_fpr'
    folder_accuracy = 'filter' + str(filter_size) + '_accuracy'
    folder = 'val_filter' + str(filter_size) + '_combine'

    threshold_ious = np.arange(0.6, 1.01, 0.01)
    threshold_scores = np.arange(0, 0.5, 0.01)

    accuracy = {}
    root_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/Accuracy'
    folder_path = os.path.join(root_path, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    img_file = 'filter_' + str(filter_size) + '_accuracy comparison.jpg'
    img_path = os.path.join(folder_path, img_file)

    acc_file = folder_accuracy + '.json'
    acc_path = os.path.join(folder_path, acc_file)

    tpr_file = folder_tpr + '.json'
    fpr_file = folder_fpr + '.json'
    tpr_path = os.path.join(folder_path, tpr_file)
    fpr_path = os.path.join(folder_path, fpr_file)

    threshold_file = 'filter' + str(filter_size) + '_thresholds' + '.json'
    threshold_path = os.path.join(folder_path, threshold_file)

    # key a[1], a[4]
    tpr1 = []
    fpr1 = []

    thresholds = []
    for t_i in threshold_ious:
        for t_s in threshold_scores:
            thresholds.append([round(t_i, 2), round(t_s, 2)])
            avrg_tpr_f_a, tpr_f_a = read_iou(folder_attack1, True, t_i, t_s)
            avrg_fpr_f_n, fpr_f_n = read_iou(folder_normal, False, t_i, t_s)

            results = [avrg_tpr_f_a, avrg_fpr_f_n]
            tpr1.append(avrg_tpr_f_a)
            fpr1.append(avrg_fpr_f_n)

    tpr1 = np.array(tpr1)
    fpr1 = np.array(fpr1)

    tpr2 = []
    fpr2 = []

    thresholds = []
    for t_i in threshold_ious:
        for t_s in threshold_scores:
            thresholds.append([round(t_i, 2), round(t_s, 2)])
            avrg_tpr_f_a, tpr_f_a = read_iou(folder_attack2, True, t_i, t_s)
            avrg_fpr_f_n, fpr_f_n = read_iou(folder_normal, False, t_i, t_s)

            results = [avrg_tpr_f_a, avrg_fpr_f_n]
            tpr2.append(avrg_tpr_f_a)
            fpr2.append(avrg_fpr_f_n)

    tpr2 = np.array(tpr2)
    fpr2 = np.array(fpr2)

    tpr3 = []
    fpr3 = []

    thresholds = []
    for t_i in threshold_ious:
        for t_s in threshold_scores:
            thresholds.append([round(t_i, 2), round(t_s, 2)])
            avrg_tpr_f_a, tpr_f_a = read_iou(folder_attack2, True, t_i, t_s)
            avrg_fpr_f_n, fpr_f_n = read_iou(folder_normal, False, t_i, t_s)

            results = [avrg_tpr_f_a, avrg_fpr_f_n]
            tpr3.append(avrg_tpr_f_a)
            fpr3.append(avrg_fpr_f_n)

    tpr3 = np.array(tpr3)
    fpr3 = np.array(fpr3)

    tpr = (tpr1 + tpr2 + tpr3) / 3
    fpr = (fpr1 + fpr2 + fpr3) / 3

    l = len(tpr)

    index = range(l)

    plt.figure(figsize=(18, 10))
    plt.plot(index, tpr, label='true positive rate')
    plt.plot(index, fpr, label='false positive rate')
    plt.grid(True)
    plt.legend()
    plt.xlabel('threshold index')
    plt.ylabel('value')
    plt.title('filter ' + str(filter_size) + ' accuracy comparison')
    plt.savefig(img_path, dpi=600)

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    for idx, i in enumerate(range(0, len(threshold_ious) * len(threshold_scores), len(threshold_scores))):

        s = i
        e = i + len(threshold_scores)

        if ix in range(s, e, 1):
            plt.figure(figsize=(18, 10))
            plt.plot(fpr[s:e], tpr[s:e], marker='.', markersize=10)
            plt.scatter(fpr[ix], tpr[ix], marker='.', color='red', s=350, label='Best')
            plt.text(fpr[ix], tpr[ix] - 0.01, '({:.2f}, {:.2f})'.format(fpr[ix], tpr[ix]), color='red', fontsize=15)
            plt.text(fpr[ix], tpr[ix] - 0.03, 'threshold_iou={:.2f} '.format(thresholds[ix][0]), color='red',
                     fontsize=15)
            plt.text(fpr[ix], tpr[ix] - 0.05, 'threshold_score={:.2f} '.format(thresholds[ix][1]), color='red',
                     fontsize=15)
            plt.grid(True)
            plt.legend(fontsize=15)
            plt.xlabel('false positive rate', fontsize=15)
            plt.ylabel('true positive rate', fontsize=15)
            plt.title('filter ' + str(filter_size) + ' ' + ' ROC', fontsize=15)
            img_roc = 'filter ' + str(filter_size) + ' ' + ' ROC.jpg'
            img_path_roc = os.path.join(folder_path, img_roc)
            plt.savefig(img_path_roc, dpi=600)


def print_accuracy(filter_size, tpr, fpr, avrg, video_ls):
    l = len(tpr)
    index = range(l)
    root_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/Accuracy'
    folder = 'filter' + str(filter_size)
    folder_path = os.path.join(root_path, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    img_file = 'filter_' + str(filter_size) + ' video comparison.jpg'
    img_path = os.path.join(folder_path, img_file)

    plt.figure(figsize=(18, 10))
    plt.plot(index, tpr, marker='.', markersize=10, label='true positive rate')
    plt.plot(index, fpr, marker='.', markersize=10, label='false positive rate')
    plt.axhline(y=avrg[0], color='r', linestyle='--')
    plt.axhline(y=avrg[1], color='r', linestyle='--')

    plt.grid(True)
    plt.legend(fontsize=15)
    plt.xlabel('video index', fontsize=15)
    plt.ylabel('value', fontsize=15)
    plt.title('filter ' + str(filter_size) + ' video comparison', fontsize=15)
    plt.savefig(img_path, dpi=600)

    output_path = os.path.join(folder_path, 'output.txt')
    tmp = sys.stdout
    sys.stdout = open(output_path, 'a')

    print('Videos with tpr below average {:.4f}'.format(avrg[0]))

    print(f"{'Count' :<10}{'Video' :<25}{'TPR' :<10}{'FPR' :<10}")
    count = 0
    for idx, t in enumerate(tpr):
        if t < avrg[0]:
            count = count + 1
            tt = '{:.4f}'.format(t)
            ff = '{:.4f}'.format(fpr[idx])
            print(f"{count :<10}{video_ls[idx] :<25}{tt :<10}{ff :<10}")
            # print(idx+1, '\t', video_ls[idx], '\t{:.4f}'.format(t), '\t{:.4f}'.format(fpr[idx]))

    print('\n')
    print('Videos with fpr above average {:.4f}'.format(avrg[1]))
    print(f"{'Count' :<10}{'Video' :<25}{'TPR' :<10}{'FPR' :<10}")
    count = 0
    for idx, f in enumerate(fpr):
        if f > avrg[1]:
            count = count + 1
            tt = '{:.4f}'.format(tpr[idx])
            ff = '{:.4f}'.format(f)
            print(f"{count :<10}{video_ls[idx] :<25}{tt :<10}{ff :<10}")

    sys.stdout.close()
    sys.stdout = tmp


def load_tpr_fpr(filter_size, threshold):
    root_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/Accuracy'
    folder = 'filter' + str(filter_size)
    folder_path = os.path.join(root_path, folder)

    # load thresholds
    thresholds_path = os.path.join(folder_path, 'filter' + str(filter_size) + '_thresholds.json')
    thresholds_file = open(thresholds_path)
    thresholds = json.load(thresholds_file)
    idx = thresholds.index(threshold)

    # load tpr_l
    tpr_path = os.path.join(folder_path, 'filter' + str(filter_size) + '_tpr.json')
    tpr_file = open(tpr_path)
    tpr_l = json.load(tpr_file)
    tpr = tpr_l[idx]

    # load fpr_l
    fpr_path = os.path.join(folder_path, 'filter' + str(filter_size) + '_fpr.json')
    fpr_file = open(fpr_path)
    fpr_l = json.load(fpr_file)
    fpr = fpr_l[idx]

    # load average fpr, fpr
    acc_path = os.path.join(folder_path, 'filter' + str(filter_size) + '_accuracy.json')
    acc_file = open(acc_path)
    acc_dic = json.load(acc_file)
    key = str(threshold)
    avrg = acc_dic[key]

    return tpr, fpr, avrg


# calculate threshold
def AdA_accuracy(folder):
    # original_path = '/home/mengjie/UAV_Tracking/UAVTrack112/V4RFlight112/data_seq'
    original_path = '/media/mengjie/Data/Downloads/UAV123_10fps/data_seq'
    root_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/{}/normal_idx'.format(folder)
    name_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/{}'.format(folder)
    name_list = os.listdir(name_path)
    name_list.remove('bbox')
    name_list.remove('normal_idx')
    name_list.remove('output.txt')
    name_list = sorted(name_list)

    total_tpr = []
    total_fpr = []

    for video in name_list:
        index_path = os.path.join(root_path, video + '.json')
        start_path = os.path.join(root_path, video + '_start.json')

        index_file = open(index_path)
        index = json.load(index_file)

        start_file = open(start_path)
        start = json.load(start_file)

        video_path = os.path.join(original_path, video)
        imgs = os.listdir(video_path)
        total_imgs = len(imgs)

        total_p = range(start, total_imgs)
        total_p = list(total_p)

        total_n = range(1, start)
        total_n = list(total_n)

        t_n = 0
        t_p = 0
        f_n = 0
        f_p = 0

        for i in index:
            if i < start:
                t_n += 1
            else:
                f_n += 1

        t_p = len(total_p) - f_n
        f_p = len(total_n) - t_n

        if (t_p + f_n) == 0:
            print(video)
            print(total_imgs)
            print(len(total_p))
            print(len(total_n))
            print(t_n)
            print(t_p)
            print(f_n)
            print(f_p)

        tpr = t_p / (t_p + f_n)
        fpr = f_p / (f_p + t_n)

        total_tpr.append(tpr)
        total_fpr.append(fpr)

    return total_tpr, total_fpr


def AdA_accuracy_all(folder):
    original_path = '/home/mengjie/UAV_Tracking/UAVTrack112/V4RFlight112/data_seq'
    #original_path = '/media/mengjie/Data/Downloads/UAV123_10fps/data_seq'
    root_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/{}/normal_idx'.format(folder)
    name_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/{}'.format(folder)
    name_list = os.listdir(name_path)
    name_list.remove('bbox')
    name_list.remove('normal_idx')
    name_list.remove('output.txt')
    name_list = sorted(name_list)

    f_p_list_path = os.path.join(name_path, 'f_p_list.json')
    f_n_list_path = os.path.join(name_path, 'f_n_list.json')

    if os.path.exists(f_p_list_path):
        name_list.remove('f_p_list.json')

    if os.path.exists(f_n_list_path):
        name_list.remove('f_n_list.json')

    total_tpr = []
    total_fpr = []

    total_fpr_list = []
    total_fnr_list = []

    for video in name_list:
        index_path = os.path.join(root_path, video + '.json')
        start_path = os.path.join(root_path, video + '_start.json')
        end_path = os.path.join(root_path, video + '_end.json')

        index_file = open(index_path)
        index = json.load(index_file)

        start_file = open(start_path)
        start = json.load(start_file)

        end_file = open(end_path)
        end = json.load(end_file)

        video_path = os.path.join(original_path, video)
        imgs = os.listdir(video_path)
        total_imgs = len(imgs)

        total_p = range(start, end + 1)
        total_p = list(total_p)

        total_n_1 = range(1, start)
        total_n_1 = list(total_n_1)

        total_n_2 = range(end, total_imgs)
        total_n_2 = list(total_n_2)

        total_n = total_n_1 + total_n_2

        t_n = 0
        t_p = 0
        f_n = 0
        f_p = 0

        t_n_list = []
        f_n_list = []

        for i in index:
            if i in total_n:
                t_n += 1
                t_n_list.append(i)

            else:
                f_n += 1
                f_n_list.append(i)

        t_p = len(total_p) - f_n
        f_p = len(total_n) - t_n

        if (t_p + f_n) == 0:
            print(video)
            print(total_imgs)
            print(len(total_p))
            print(len(total_n))
            print(t_n)
            print(t_p)
            print(f_n)
            print(f_p)

        tpr = t_p / (t_p + f_n)
        # fpr = f_p / (f_p + t_n)
        fpr = 0

        total_tpr.append(tpr)
        total_fpr.append(fpr)

        f_p_list = sorted(list(set(total_n) - set(t_n_list)))
        f_n_list = sorted(f_n_list)

        f_p_list_1 = [[start, end], f_p_list]
        f_n_list_1 = [[start, end], f_n_list]

        total_fpr_list.append(f_p_list_1)
        total_fnr_list.append(f_n_list_1)

    json.dump(total_fpr_list, open(f_p_list_path, 'w'))

    json.dump(total_fnr_list, open(f_n_list_path, 'w'))

    return total_tpr, total_fpr


def AdA_accuracy_end(folder):
    original_path = '/home/mengjie/UAV_Tracking/UAVTrack112/V4RFlight112/data_seq'
    #original_path = '/media/mengjie/Data/Downloads/UAV123_10fps/data_seq'
    root_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/{}/normal_idx'.format(folder)
    name_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/{}'.format(folder)
    name_list = os.listdir(name_path)
    name_list.remove('bbox')
    name_list.remove('normal_idx')
    name_list.remove('output.txt')
    f_p_list_path = os.path.join(name_path, 'f_p_list.json')
    f_n_list_path = os.path.join(name_path, 'f_n_list.json')

    if os.path.exists(f_p_list_path):
        name_list.remove('f_p_list.json')

    if os.path.exists(f_n_list_path):
        name_list.remove('f_n_list.json')

    name_list = sorted(name_list)

    total_tpr = []
    total_fpr = []

    total_fpr_list = []
    total_fnr_list = []

    for video in name_list:
        index_path = os.path.join(root_path, video + '.json')
        start_path = os.path.join(root_path, video + '_start.json')
        end_path = os.path.join(root_path, video + '_end.json')

        index_file = open(index_path)
        index = json.load(index_file)

        start_file = open(start_path)
        start = json.load(start_file)

        end_file = open(end_path)
        end = json.load(end_file)

        video_path = os.path.join(original_path, video)
        imgs = os.listdir(video_path)
        total_imgs = len(imgs)

        total_p = range(start, end + 1)
        total_p = list(total_p)

        total_n_1 = range(1, start)
        total_n_1 = list(total_n_1)

        total_n_2 = range(end + 1, total_imgs)
        total_n_2 = list(total_n_2)

        total_n = total_n_1 + total_n_2

        t_n = 0
        t_p = 0
        f_n = 0
        f_p = 0

        t_n_list = []
        f_n_list = []

        for i in index:
            if i in total_n:
                t_n += 1
                t_n_list.append(i)
            else:
                f_n += 1
                f_n_list.append(i)

        t_p = len(total_p) - f_n
        f_p = len(total_n) - t_n

        f_p_list = sorted(list(set(total_n) - set(t_n_list)))
        f_n_list = sorted(f_n_list)

        if (t_p + f_n) == 0:
            print(video)
            print(total_imgs)
            print(len(total_p))
            print(len(total_n))
            print(t_n)
            print(t_p)
            print(f_n)
            print(f_p)

        tpr = t_p / (t_p + f_n)
        fpr = f_p / (f_p + t_n)

        total_tpr.append(tpr)
        total_fpr.append(fpr)

        f_p_list_1 = [[start, end], f_p_list]
        f_n_list_1 = [[start, end], f_n_list]

        total_fpr_list.append(f_p_list_1)
        total_fnr_list.append(f_n_list_1)

    json.dump(total_fpr_list, open(f_p_list_path, 'w'))

    json.dump(total_fnr_list, open(f_n_list_path, 'w'))

    return total_tpr, total_fpr


def AdA_accuracy_twice(folder):
    original_path = '/home/mengjie/UAV_Tracking/UAVTrack112/V4RFlight112/data_seq'
    #original_path = '/media/mengjie/Data/Downloads/UAV123_10fps/data_seq'
    root_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/{}/normal_idx'.format(folder)
    name_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/{}'.format(folder)
    name_list = os.listdir(name_path)
    name_list.remove('bbox')
    name_list.remove('normal_idx')
    name_list.remove('output.txt')
    f_p_list_path = os.path.join(name_path, 'f_p_list.json')
    f_n_list_path = os.path.join(name_path, 'f_n_list.json')

    if os.path.exists(f_p_list_path):
        name_list.remove('f_p_list.json')

    if os.path.exists(f_n_list_path):
        name_list.remove('f_n_list.json')

    name_list = sorted(name_list)

    total_tpr = []
    total_fpr = []

    total_fpr_list = []
    total_fnr_list = []

    for video in name_list:
        index_path = os.path.join(root_path, video + '.json')
        start_path1 = os.path.join(root_path, video + '_start1.json')
        end_path1 = os.path.join(root_path, video + '_end1.json')

        start_path2 = os.path.join(root_path, video + '_start2.json')
        end_path2 = os.path.join(root_path, video + '_end2.json')

        index_file = open(index_path)
        index = json.load(index_file)

        start_file1 = open(start_path1)
        start1 = json.load(start_file1)

        end_file1 = open(end_path1)
        end1 = json.load(end_file1)

        start_file2 = open(start_path2)
        start2 = json.load(start_file2)

        end_file2 = open(end_path2)
        end2 = json.load(end_file2)

        video_path = os.path.join(original_path, video)
        imgs = os.listdir(video_path)
        total_imgs = len(imgs) - 1

        total_p1 = range(start1, end1 + 1)
        total_p1 = list(total_p1)

        total_p2 = range(start2, end2 + 1)
        total_p2 = list(total_p2)

        total_p = total_p1 + total_p2

        total_n_1 = range(1, start1)
        total_n_1 = list(total_n_1)

        total_n_2 = range(end1 + 1, start2)
        total_n_2 = list(total_n_2)

        total_n = total_n_1 + total_n_2

        t_n = 0
        t_p = 0
        f_n = 0
        f_p = 0

        t_n_list = []
        f_n_list = []

        for i in index:
            if i in total_n:
                t_n += 1
                t_n_list.append(i)
            else:
                f_n += 1
                f_n_list.append(i)

        t_p = len(total_p) - f_n
        f_p = len(total_n) - t_n

        f_p_list = sorted(list(set(total_n) - set(t_n_list)))
        f_n_list = sorted(f_n_list)

        if (t_p + f_n) == 0:
            print(video)
            print(total_imgs)
            print(len(total_p))
            print(len(total_n))
            print(t_n)
            print(t_p)
            print(f_n)
            print(f_p)

        tpr = t_p / (t_p + f_n)
        fpr = f_p / (f_p + t_n)

        total_tpr.append(tpr)
        total_fpr.append(fpr)

        f_p_list_1 = [[start1, end1], [start2, end2], f_p_list]
        f_n_list_1 = [[start1, end1], [start2, end2], f_n_list]

        total_fpr_list.append(f_p_list_1)
        total_fnr_list.append(f_n_list_1)

    json.dump(total_fpr_list, open(f_p_list_path, 'w'))

    json.dump(total_fnr_list, open(f_n_list_path, 'w'))

    return total_tpr, total_fpr


def normal_idx_15(folder):
    original_path = '/home/mengjie/UAV_Tracking/UAVTrack112/V4RFlight112/data_seq'
    root_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/{}/normal_idx'.format(folder)
    name_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/V4RFlight112/{}'.format(folder)
    name_list = os.listdir(name_path)
    name_list.remove('bbox')
    name_list.remove('normal_idx')
    name_list.remove('output.txt')
    name_list = sorted(name_list)

    head_tail = os.path.split(root_path)

    total_tpr = []
    total_fpr = []

    for video in name_list:
        # index_path = os.path.join(root_path, video + '.json')
        start_path = os.path.join(root_path, video + '_start.json')

        score_path = os.path.join(head_tail[0], video + '/filter_attack/score_diff_f_a.json')
        score_file = open(score_path)
        score_diff = json.load(score_file)

        # index_file = open(index_path)
        # index = json.load(index_file)

        start_file = open(start_path)
        start = json.load(start_file)

        video_path = os.path.join(original_path, video)
        imgs = os.listdir(video_path)
        total_imgs = len(imgs)

        total_p = range(start, total_imgs)
        total_p = list(total_p)

        total_n = range(1, start)
        total_n = list(total_n)

        t_n = 0
        t_p = 0
        f_n = 0
        f_p = 0

        index = []

        for i, d in enumerate(score_diff):
            if abs(d) <= 0.15:
                index.append(i + 1)

        for i in index:
            if i < start:
                t_n += 1
            else:
                f_n += 1

        t_p = len(total_p) - f_n
        f_p = len(total_n) - t_n

        if (t_p + f_n) == 0:
            print(video)
            print(total_imgs)
            print(len(total_p))
            print(len(total_n))
            print(t_n)
            print(t_p)
            print(f_n)
            print(f_p)

        tpr = t_p / (t_p + f_n)
        fpr = f_p / (f_p + t_n)

        total_tpr.append(tpr)
        total_fpr.append(fpr)

    return total_tpr, total_fpr


def UAV10():
    path = '/media/mengjie/Data/Downloads/UAV123_10fps'
    name_list = os.listdir(path + '/data_seq')
    name_list.sort()
    txt = path + '/img_index.txt'
    f = open(txt)
    file = f.readlines()
    for i in range(len(file)):

        video_name = re.search('name(.*)path', file[i])
        video_name = video_name.group(1)
        video_name = re.search(",'(.*)',", video_name)
        video_name = video_name.group(1)

        start = re.search('startFrame(.*)end', file[i])
        start = start.group(1)
        start = re.search(',(.*),', start)
        start = start.group(1)
        start = int(start)

        end = re.search('endFrame(.*)nz', file[i])
        end = end.group(1)
        end = re.search(',(.*),', end)
        end = end.group(1)
        end = int(end)

        og_path = re.search('path(.*)start', file[i])
        og_path = og_path.group(1)
        og_path = re.search(',(.*),', og_path)
        og_path = og_path.group(1)
        og_path = re.search('D:(.*)', og_path)
        og_path = og_path.group(1)
        og_path = og_path.replace("'", '')
        og_path = og_path.replace('\\', '/')
        og_path = og_path.replace('/UAV123_10fps', '')
        og_path = path + og_path

        vn = re.search('data_seq/(.*)/', og_path)
        vn = vn.group(1)

        des_path = og_path.replace('data_seq', 'data_seq(1)')
        des_path = des_path.replace(vn, video_name)
        if not os.path.isdir(des_path):
            os.makedirs(des_path)

        for j in range(start, end + 1):
            og_img = os.path.join(og_path, '{:06d}.jpg'.format(j))
            des_img = os.path.join(des_path, '{:06d}.jpg'.format(j))
            shutil.copyfile(og_img, des_img)

# plot result comparison figures
def success_plot(dataset):
    # normal
    normal_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/{}/Experiments/Pred_bbox/{}/normal'.format(
        dataset, 'filter3_normal')
    p_normal_path = os.path.join(normal_path, 'precision.json')
    s_normal_path = os.path.join(normal_path, 'success.json')
    p_normal = json.load(open(p_normal_path))
    s_normal = json.load(open(s_normal_path))
    x_p = np.arange(0, 51, 1)
    x_s = np.arange(0, 1.05, 0.05)
    dataset_l = len(p_normal)

    y_p_normal = np.zeros(51)
    y_s_normal = np.zeros(21)

    for i in range(len(p_normal)):
        y_p_normal += p_normal[i]

    for i in range(len(s_normal)):
        y_s_normal += s_normal[i]

    y_p_normal = np.round(y_p_normal / dataset_l, 3)
    y_s_normal = np.round(y_s_normal / dataset_l, 3)

    # all, Ad2 attack
    folder1 = dataset + '_50_whole_19(1)'
    Ad2_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/{}/Experiments/Pred_bbox/{}/recovery(1)'.format(
        dataset, folder1)
    p_ad2_path = os.path.join(Ad2_path, 'precision.json')
    s_ad2_path = os.path.join(Ad2_path, 'success.json')
    p_ad2 = json.load(open(p_ad2_path))
    s_ad2 = json.load(open(s_ad2_path))

    y_p_ad2 = np.zeros(51)
    y_s_ad2 = np.zeros(21)

    for i in range(len(p_ad2)):
        y_p_ad2 += p_ad2[i]

    for i in range(len(s_ad2)):
        y_s_ad2 += s_ad2[i]

    y_p_ad2 = np.round(y_p_ad2 / dataset_l, 3)
    y_s_ad2 = np.round(y_s_ad2 / dataset_l, 3)

    folder1_1 = dataset + '_50_whole_19(attack)'
    Ad2_path_a = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/{}/Experiments/Pred_bbox/{}/attack'.format(
        dataset, folder1_1)
    p_ad2_path_a = os.path.join(Ad2_path_a, 'precision.json')
    s_ad2_path_a = os.path.join(Ad2_path_a, 'success.json')
    p_ad2_a = json.load(open(p_ad2_path_a))
    s_ad2_a = json.load(open(s_ad2_path_a))

    y_p_ad2_a = np.zeros(51)
    y_s_ad2_a = np.zeros(21)

    for i in range(len(p_ad2_a)):
        y_p_ad2_a += p_ad2_a[i]

    for i in range(len(s_ad2_a)):
        y_s_ad2_a += s_ad2_a[i]

    y_p_ad2_a = np.round(y_p_ad2_a / dataset_l, 3)
    y_s_ad2_a = np.round(y_s_ad2_a / dataset_l, 3)

    # end, duration
    folder2 = dataset + '_50_end_19(1)'
    end_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/{}/Experiments/Pred_bbox/{}/recovery'.format(
        dataset, folder2)
    p_end_path = os.path.join(end_path, 'precision.json')
    s_end_path = os.path.join(end_path, 'success.json')
    p_end = json.load(open(p_end_path))
    s_end = json.load(open(s_end_path))

    y_p_end = np.zeros(51)
    y_s_end = np.zeros(21)

    for i in range(len(p_end)):
        y_p_end += p_end[i]

    for i in range(len(s_end)):
        y_s_end += s_end[i]

    y_p_end = np.round(y_p_end / dataset_l, 3)
    y_s_end = np.round(y_s_end / dataset_l, 3)

    folder2_2 = dataset + '_50_end_19(attack)'
    end_path_a = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/{}/Experiments/Pred_bbox/{}/attack'.format(
        dataset, folder2_2)
    p_end_path_a = os.path.join(end_path_a, 'precision.json')
    s_end_path_a = os.path.join(end_path_a, 'success.json')
    p_end_a = json.load(open(p_end_path_a))
    s_end_a = json.load(open(s_end_path_a))

    y_p_end_a = np.zeros(51)
    y_s_end_a = np.zeros(21)

    for i in range(len(p_end_a)):
        y_p_end_a += p_end_a[i]

    for i in range(len(s_end_a)):
        y_s_end_a += s_end_a[i]

    y_p_end_a = np.round(y_p_end_a / dataset_l, 3)
    y_s_end_a = np.round(y_s_end_a / dataset_l, 3)

    # twice attack
    folder3 = dataset + '_50_twice_19(1)'
    twice_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/{}/Experiments/Pred_bbox/{}/recovery'.format(
        dataset, folder3)
    p_twice_path = os.path.join(twice_path, 'precision.json')
    s_twice_path = os.path.join(twice_path, 'success.json')
    p_twice = json.load(open(p_twice_path))
    s_twice = json.load(open(s_twice_path))

    y_p_twice = np.zeros(51)
    y_s_twice = np.zeros(21)

    for i in range(len(p_twice)):
        y_p_twice += p_twice[i]

    for i in range(len(s_twice)):
        y_s_twice += s_twice[i]

    y_p_twice = np.round(y_p_twice / dataset_l, 3)
    y_s_twice = np.round(y_s_twice / dataset_l, 3)

    folder3_3 = dataset + '_50_twice_19(attack)'
    twice_path_a = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/{}/Experiments/Pred_bbox/{}/attack'.format(
        dataset, folder3_3)
    p_twice_path_a = os.path.join(twice_path_a, 'precision.json')
    s_twice_path_a = os.path.join(twice_path_a, 'success.json')
    p_twice_a = json.load(open(p_twice_path_a))
    s_twice_a = json.load(open(s_twice_path_a))

    y_p_twice_a = np.zeros(51)
    y_s_twice_a = np.zeros(21)

    for i in range(len(p_twice_a)):
        y_p_twice_a += p_twice_a[i]

    for i in range(len(s_twice_a)):
        y_s_twice_a += s_twice_a[i]

    y_p_twice_a = np.round(y_p_twice_a / dataset_l, 3)
    y_s_twice_a = np.round(y_s_twice_a / dataset_l, 3)

    plt.figure(figsize=(12, 8))
    plt.plot(x_p, y_p_normal, label='[0.811] Normal', color='green')
    plt.plot(x_p, y_p_ad2_a, label='[0.315] Ad2 Attack', linestyle='--', color='orange')
    plt.plot(x_p, y_p_ad2, label='[0.746] Ad2 Recovery', color='orange')

    plt.plot(x_p, y_p_end_a, label='[0.410] One-period Attack', linestyle='--', color='red')
    plt.plot(x_p, y_p_end, label='[0.785] One-period Recovery', color='red')

    plt.plot(x_p, y_p_twice_a, label='[0.414] Two-period Attack', linestyle='--', color='blue')
    plt.plot(x_p, y_p_twice, label='[0.764] Two-period Recovery', color='blue')
    plt.xlim(0, 50)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.xlabel('Location error threshold', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    plt.title('Precision plots on UAVTrack112', fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize='12', )
    plt.tight_layout()
    plt.savefig('/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/Figures/Precision/' + 'Precision 1.jpg',
                dpi=600)

    plt.figure(figsize=(12, 8))
    plt.plot(x_s, y_s_normal, label='[0.617] Normal', color='green')
    plt.plot(x_s, y_s_ad2_a, label='[0.135] Ad2 Attack', linestyle='--', color='orange')
    plt.plot(x_s, y_s_ad2, label='[0.545] Ad2 Recovery', color='orange')

    plt.plot(x_s, y_s_end_a, label='[0.232] One-period Attack', linestyle='--', color='red')
    plt.plot(x_s, y_s_end, label='[0.587] One-period Recovery', color='red')

    plt.plot(x_s, y_s_twice_a, label='[0.230] Two-period Attack', linestyle='--', color='blue')
    plt.plot(x_s, y_s_twice, label='[0.565] Two-period Recovery', color='blue')

    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.xlabel('Overlap threshold', fontsize=15)
    plt.ylabel('Success rate', fontsize=15)
    plt.title('Success plots on UAVTrack112', fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize='12', )
    plt.tight_layout()
    plt.savefig('/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/Figures/Success/' + 'Success 1.jpg',
                dpi=600)


def get_normal_success_index(dataset, folder, threshold):
    # normal
    normal_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/{}/Experiments/Pred_bbox/{}/normal'.format(
        dataset, 'filter3_normal')
    save_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/{}/{}'.format(dataset, folder)
    # p_normal_path = os.path.join(normal_path, 'my_precision.json')
    s_normal_path = os.path.join(normal_path, 'my_success.json')
    # p_normal = json.load(open(p_normal_path))
    s_normal = json.load(open(s_normal_path))
    # x_p = np.arange(0, 51, 1)
    # x_s = np.arange(0, 1.05, 0.05)
    dataset_l = len(s_normal)
    normal_success_path = os.path.join(save_path, 'normal_success.json')
    normal_fail_path = os.path.join(save_path, 'normal_fail.json')

    normal_success_index_all = []
    normal_fail_index_all = []

    for v in range(dataset_l):
        v_frame = s_normal[v]
        normal_success_index = []
        normal_fail_index = []
        print('len(v_frame)', len(v_frame))
        for k in range(len(v_frame)):
            if v_frame[k] >= threshold:
                normal_success_index.append(k)
            else:
                normal_fail_index.append(k)

        normal_success_index_all.append(sorted(normal_success_index))
        normal_fail_index_all.append(sorted(normal_fail_index))

    with open(normal_success_path, 'w') as file:
        json.dump(normal_success_index_all, file, indent=1)

    with open(normal_fail_path, 'w') as file:
        json.dump(normal_fail_index_all, file, indent=1)


def get_fail_attack_index(dataset, folder, threshold):
    # normal
    root_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/{}/Experiments/Pred_bbox/{}/attack'.format(
        dataset, folder)
    save_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/{}/{}'.format(dataset, folder)
    fail_attack_index_path = os.path.join(save_path, 'fail_attack_index.json')
    unreasonable_attack_index_path = os.path.join(save_path, 'unreasonable_attack_index.json')
    real_f_n_index_path = os.path.join(save_path, 'real_f_n_index.json')
    correct_result_path = os.path.join(save_path, 'correct_result.json')
    average_result_path = os.path.join(save_path, 'average_result.json')

    s_attack_path = os.path.join(root_path, 'my_success_attack.json')
    s_attack = json.load(open(s_attack_path))

    f_n_list_path = os.path.join(save_path, 'f_n_list.json')
    f_n_list = json.load(open(f_n_list_path))

    normal_success_path = os.path.join(save_path, 'normal_success.json')
    normal_success = json.load(open(normal_success_path))

    normal_fail_path = os.path.join(save_path, 'normal_fail.json')
    normal_fail = json.load(open(normal_fail_path))

    real_f_n_index_all = []
    unreasonable_attack_index_all = []
    fail_attack_index_all = []
    correct_result_all = []  # result = [fnr, fail_attack_rate, real_f_n_rate, unreasonable_attack_rate]
    if 'twice' in folder:
        # i is the video index
        for i in range(len(f_n_list)):
            fail_attack_index = []
            real_f_n_index = []
            unreasonable_attack_index = []
            correct_result = []
            p1 = f_n_list[i][0]
            p2 = f_n_list[i][1]

            l1 = p1[1] - p1[0] + 1
            l2 = p2[1] - p2[0] + 1

            l = l1 + l2

            f_n_index = f_n_list[i][2]
            s_attack_list = s_attack[i]
            normal_success_index = normal_success[i]
            normal_fail_index = normal_fail[i]

            # j is the frame index
            for j in range(len(s_attack_list)):
                # reasonable attack
                if j in f_n_index and j in normal_success_index:
                    # unsuccessful attack
                    if s_attack_list[j] < threshold:
                        fail_attack_index.append(j)

                    else:
                        # real false negatives
                        real_f_n_index.append(j)

                # unreasonable attack
                elif j in f_n_index and j in normal_fail_index:
                    unreasonable_attack_index.append(j)

            l = l - len(unreasonable_attack_index) - len(fail_attack_index)
            if len(f_n_index) == 0:
                rate = 0
                fnr = 0
                fail_attack_index_all.append([rate, p1, p2, fail_attack_index])
                real_f_n_index_all.append([rate, real_f_n_index])
                unreasonable_attack_index_all.append([rate, unreasonable_attack_index])
                correct_result = [fnr, rate, rate, rate]
                correct_result_all.append(correct_result)
            else:
                fnr = np.round(len(real_f_n_index) / l, 3)
                rate1 = np.round(len(fail_attack_index) / len(f_n_index), 3)
                fail_attack_index_all.append([rate1, p1, p2, fail_attack_index])

                rate2 = np.round(len(real_f_n_index) / len(f_n_index), 3)
                real_f_n_index_all.append([rate2, real_f_n_index])

                rate3 = np.round(len(unreasonable_attack_index) / len(f_n_index), 3)
                unreasonable_attack_index_all.append([rate3, unreasonable_attack_index])

                correct_result = [fnr, rate1, rate2, rate3]
                correct_result_all.append(correct_result)


    else:
        # i is the video index
        for i in range(len(f_n_list)):
            fail_attack_index = []
            real_f_n_index = []
            unreasonable_attack_index = []
            correct_result = []  # result = [fnr, fail_attack, real_f_n, unreasonable_attack]

            p1 = f_n_list[i][0]

            l = p1[1] - p1[0] + 1

            f_n_index = f_n_list[i][1]

            s_attack_list = s_attack[i]
            normal_success_index = normal_success[i]
            normal_fail_index = normal_fail[i]

            # j is the frame index
            for j in range(len(s_attack_list)):
                # reasonable attack
                if j in f_n_index and j in normal_success_index:
                    # unsuccessful attack
                    if s_attack_list[j] < threshold:
                        fail_attack_index.append(j)

                    else:
                        # real false negatives
                        real_f_n_index.append(j)

                # unreasonable attack
                elif j in f_n_index and j in normal_fail_index:
                    unreasonable_attack_index.append(j)

            l = l - len(unreasonable_attack_index) - len(fail_attack_index)

            if len(f_n_index) == 0:
                rate = 0
                fnr = 0
                fail_attack_index_all.append([rate, p1, fail_attack_index])
                real_f_n_index_all.append([rate, real_f_n_index])
                unreasonable_attack_index_all.append([rate, unreasonable_attack_index])
                correct_result = [fnr, rate, rate, rate]
                correct_result_all.append(correct_result)
            else:
                fnr = np.round(len(real_f_n_index) / l, 3)
                rate1 = np.round(len(fail_attack_index) / len(f_n_index), 3)
                fail_attack_index_all.append([rate1, p1, fail_attack_index])

                rate2 = np.round(len(real_f_n_index) / len(f_n_index), 3)
                real_f_n_index_all.append([rate2, real_f_n_index])

                rate3 = np.round(len(unreasonable_attack_index) / len(f_n_index), 3)
                unreasonable_attack_index_all.append([rate3, unreasonable_attack_index])

                correct_result = [fnr, rate1, rate2, rate3]
                correct_result_all.append(correct_result)

    r = np.array(correct_result_all)
    #mask = (r[:, 0] != 0)
    # r1 = r[~np.all(r == 0, axis=1)]
    # # r1 = np.delete(r, np.where(r[])[0], axis=0)
    z_index = []
    for i in range(len(r)):
        if np.count_nonzero(r[i]) == 0:
            print(r[i])
            z_index.append(i)
    r1 = np.delete(r, z_index, axis=0)
    print(r.shape)
    print(np.mean(r, axis=0))
    print(r1.shape)
    print(np.mean(r1, axis=0))

    average_result = np.mean(r1, axis=0).tolist()
    with open(average_result_path, 'w') as file:
        json.dump(average_result, file, indent=1)
    with open(fail_attack_index_path, 'w') as file:
        json.dump(fail_attack_index_all, file, indent=1)
    with open(real_f_n_index_path, 'w') as file:
        json.dump(real_f_n_index_all, file, indent=1)
    with open(unreasonable_attack_index_path, 'w') as file:
        json.dump(unreasonable_attack_index_all, file, indent=1)
    with open(correct_result_path, 'w') as file:
        json.dump(correct_result_all, file, indent=1)


def check_f_n_index(dataset, folder):
    root_path = '/home/mengjie/PycharmProjects/Ad2Attack/pysot/tools/results/{}/{}'.format(dataset, folder)
    f_n_path = os.path.join(root_path, 'f_n_list.json')
    f_n = json.load(open(f_n_path))

    # x_p = np.arange(0, 51, 1)
    # x_s = np.arange(0, 1.05, 0.05)
    dataset_l = len(s_normal)
    normal_success_path = os.path.join(normal_path, 'normal_success.json')
    normal_fail_path = os.path.join(normal_path, 'normal_fail.json')

    normal_success_index_all = []
    normal_fail_index_all = []

    for v in range(dataset_l):
        v_frame = s_normal[v]
        normal_success_index = []
        normal_fail_index = []
        print('len(v_frame)', len(v_frame))
        for k in range(len(v_frame)):
            if v_frame[k] >= threshold:
                normal_success_index.append(k)
            else:
                normal_fail_index.append(k)

        normal_success_index_all.append(sorted(normal_success_index))
        normal_fail_index_all.append(sorted(normal_fail_index))

    with open(normal_success_path, 'w') as file:
        json.dump(normal_success_index_all, file, indent=1)

    with open(normal_fail_path, 'w') as file:
        json.dump(normal_fail_index_all, file, indent=1)


if __name__ == '__main__':
    # # filter2: [0.95, 0.12]; filter3: [0.97, 0.15]; filter4: [0.94,0.22]
    #cal_accuracy(3)
    #stop
    dataset = 'V4RFlight112'
    success_plot(dataset)
    stop
    folder_end = 'V4RFlight112_50_end_19(1)'
    folder_twice = 'V4RFlight112_50_twice_19(1)'
    folder_whole = 'V4RFlight112_50_whole_19(1)'
    #
    threshold = 0.4
    # stop
    get_normal_success_index(dataset, folder_end, threshold)
    get_normal_success_index(dataset, folder_twice, threshold)
    get_normal_success_index(dataset, folder_whole, threshold)
    # # stop
    for f in [folder_whole, folder_twice, folder_end]:
        get_fail_attack_index(dataset, f, threshold)
    # stop

    # filter_size_range = [2, 3, 4, 5]
    #
    # for filter_size in filter_size_range:
    #     cal_accuracy(filter_size)
    # stop
    # # cal_accuracy(3)

    # cal_accuracy_combine(3)
    # stop

    # high_resolution()
    # dataset = 'V4RFlight112'
    # success_plot(dataset)
    # stop
    #
    # folder_end = 'UAVDT_50_end_19(1)'
    # folder_twice = 'UAVDT_50_twice_19(1)'
    # folder_whole = 'UAVDT_50_whole_19(1)'
    stop
    total_tpr, total_fpr = AdA_accuracy_end(folder_end)
    total_tpr = np.array(total_tpr)
    total_fpr = np.array(total_fpr)

    tpr = mean(total_tpr)
    fpr = mean(total_fpr)

    print('end average tpr: ', tpr)
    print('end average fpr: ', fpr)

    total_tpr, total_fpr = AdA_accuracy_twice(folder_twice)
    total_tpr = np.array(total_tpr)
    total_fpr = np.array(total_fpr)

    tpr = mean(total_tpr)
    fpr = mean(total_fpr)

    print('twice average tpr: ', tpr)
    print('twice average fpr: ', fpr)
    # total_tpr, total_fpr = normal_idx_15(folder)

    total_tpr, total_fpr = AdA_accuracy_all(folder_whole)
    total_tpr = np.array(total_tpr)
    total_fpr = np.array(total_fpr)

    tpr = mean(total_tpr)
    fpr = mean(total_fpr)

    print('whole average tpr: ', tpr)
    print('whole average fpr: ', fpr)
    # total_tpr = np.array(total_tpr)
    # total_fpr = np.array(total_fpr)
    #
    # tpr = mean(total_tpr)
    # fpr = mean(total_fpr)
    #
    # print('average tpr: ', tpr)
    # print('average fpr: ', fpr)

    # combine_imgs()

    # tpr, fpr, avrg = load_tpr_fpr(filter_size, [0.94,0.22])
    # print_accuracy(filter_size, tpr, fpr, avrg, video_ls)
    #
    # # cal_accuracy(filter_size)
    # stop

    # AdA accuracy

    # filter_sizes = [5]
    #
    # for s in filter_sizes:
    #     cal_accuracy(s)
