import json

import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import cv2
import numpy as np
from Setting import train_set_path as dataset_dir
from PIL import Image
from statistics import mean
# from pysot.pysot.datasets.augmentation import Augmentation
import shutil

def img2tensor(img_arr):
    '''float64 ndarray (H,W,3) ---> float32 torch tensor (1,3,H,W)'''
    img_arr = img_arr.astype(np.float32)
    img_arr = img_arr.transpose(2, 0, 1)  # channel first
    img_arr = img_arr[np.newaxis, :, :, :]
    init_tensor = torch.from_numpy(img_arr)  # (1,3,H,W)
    return init_tensor


def normalize(im_tensor):
    '''(0,255) ---> (-1,1)'''
    im_tensor = im_tensor / 255.0
    im_tensor = im_tensor - 0.5
    im_tensor = im_tensor / 0.5
    return im_tensor

# original backup
# def tensor2img(tensor):
#     '''(0,255) tensor ---> (0,255) img'''
#     '''(1,3,H,W) ---> (H,W,3)'''
#     tensor = tensor.squeeze(0).permute(1, 2, 0)
#     img = tensor.cpu().numpy().clip(0, 255).astype(np.uint8)
#     return img

def tensor2img(tensor):
    '''(0,255) tensor ---> (0,255) img'''
    '''(1,3,H,W) ---> (H,W,3)'''
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    img = tensor.detach().cpu().numpy()
    return img

# original function
# class GOT10k_dataset(Dataset):
#    def __init__(self, max_num=15):
#        folders = sorted(os.listdir(dataset_dir))
#        folders.remove('init_gt.txt')
#        self.folders_list = [os.path.join(dataset_dir, folder) for folder in folders]
#        self.max_num = max_num

#    def __getitem__(self, index):
#        cur_folder = self.folders_list[index]
#        img_paths = sorted(glob.glob(os.path.join(cur_folder, '*.jpg')))
#        zhanbi_path = os.path.join(cur_folder, 'zhanbi.txt')
#        file = open(zhanbi_path, 'r')
#        zhanbi = float(file.readline())
#        '''get init frame tensor'''
#        init_frame_path = img_paths[0]
#        init_frame_arr = cv2.imread(init_frame_path)
#        init_tensor = img2tensor(init_frame_arr)
#        '''get search regions' tensor'''
#        search_region_paths = img_paths[1:self.max_num + 1]  # to avoid being out of GPU memory
#        num_search = len(search_region_paths)
#        search_tensor = torch.zeros((num_search, 3, 255, 255), dtype=torch.float32)
#        for i in range(num_search):
#            search_arr = cv2.imread(search_region_paths[i])
#            search_tensor[i] = img2tensor(search_arr)
#        '''Note: we don't normalize these tensors here,
#        but leave normalization to training process'''
#        return (init_tensor, search_tensor, zhanbi)

#    def __len__(self):
#        return len(self.folders_list)

# define my dataset
class GOT10k_dataset(Dataset):
    def __init__(self, train_rep, max_num=15):
        folders = sorted(os.listdir(train_rep))
        # folders.remove('init_gt.txt')
        self.folders_list = [os.path.join(train_rep, folder) for folder in folders]
        self.max_num = max_num

    def __getitem__(self, index):
        cur_folder = self.folders_list[index]
        img_paths = sorted(glob.glob(os.path.join(cur_folder, '*.jpg')))
        zhanbi_path = os.path.join(cur_folder, 'zhanbi.txt')
        file = open(zhanbi_path, 'r')
        zhanbi = float(file.readline())
        '''get init frame tensor'''
        init_frame_path = img_paths[0]
        init_frame_arr = cv2.imread(init_frame_path)
        init_tensor = img2tensor(init_frame_arr)
        '''get search regions' tensor'''
        search_region_paths = img_paths[1:self.max_num + 1]  # to avoid being out of GPU memory
        num_search = len(search_region_paths)
        # before is (num_search, 3, 255, 255)
        search_tensor = torch.zeros((num_search, 3, 287, 287), dtype=torch.float32)
        for i in range(num_search):
            search_arr = cv2.imread(search_region_paths[i])
            search_tensor[i] = img2tensor(search_arr)
        '''Note: we don't normalize these tensors here, 
        but leave normalization to training process'''
        return (init_tensor, search_tensor, zhanbi, cur_folder)

    def __len__(self):
        return len(self.folders_list)


# calculate zhanbi and write it to a json file

def cal_zhanbi(data_dir):
    folders = sorted(os.listdir(data_dir))
    folders.remove('list.txt')
    video_list = [os.path.join(data_dir, folder) for folder in folders]
    for video in video_list:
        img_paths = sorted(glob.glob(os.path.join(video, '*.jpg')))
        json_path = os.path.join(video, 'zhanbi.json')
        gt_path = os.path.join(video, 'groundtruth.txt')
        with open(gt_path) as f:
            gt_video = f.read().splitlines()
        zhanbi_list = []
        for idx, gt_line in enumerate(gt_video):
            gt_img = gt_line.strip().split(',')
            gt_w, gt_h = [int(float(gt_img[2])), int(float(gt_img[3]))]
            img = cv2.imread(img_paths[idx])
            img_h, img_w, img_c = img.shape
            zhanbi = (gt_w * gt_h) / (img_h * img_w)
            zhanbi_list.append(zhanbi)

        json.dump(zhanbi_list, open(json_path, 'w'))
        # f = open(json_path)
        # data = json.load(f)
        head_tail = os.path.split(video)
        print('Finish Video ', head_tail[1])

    print('Finish All Videos')


def get_GOT10K_data(data_dir, video_index, max_num=15):
    folders = sorted(os.listdir(data_dir))
    video_list = [os.path.join(data_dir, folder) for folder in folders]
    for video in video_list:
        img_paths = sorted(glob.glob(os.path.join(video, '*.jpg')))
        zhanbi_path = os.path.join(video, 'zhanbi.txt')
        file = open(zhanbi_path, 'r')
        zhanbi = float(file.readline())
        # gt_path = os.path.join(video, 'groundtruth.txt')
        init_frame_path = img_paths[0]
        init_frame_arr = cv2.imread(init_frame_path)
        init_tensor = img2tensor(init_frame_arr)
        '''get search regions' tensor'''
        search_region_paths = img_paths[1:self.max_num + 1]  # to avoid being out of GPU memory
        num_search = len(search_region_paths)
        search_tensor = torch.zeros((num_search, 3, 255, 255), dtype=torch.float32)
        for i in range(num_search):
            search_arr = cv2.imread(search_region_paths[i])
            search_tensor[i] = img2tensor(search_arr)
        '''Note: we don't normalize these tensors here, 
        but leave normalization to training process'''
        return (init_tensor, search_tensor, zhanbi)

        gt_first_img = gt_video[0].strip().split(',')
        # bbox: x_left, y_left, w, h
        bbox_first = [int(float(gt_first_img[0])), int(float(gt_first_img[1])),
                      int(float(gt_first_img[2])), int(float(gt_first_img[3]))]
        zhanbi_list = []

        for idx, gt_line in enumerate(gt_video):
            gt_img = gt_line.strip().split(',')
            gt_w, gt_h = [int(float(gt_img[2])), int(float(gt_img[3]))]
            img = cv2.imread(img_paths[idx])
            img_h, img_w, img_c = img.shape
            zhanbi = (gt_w * gt_h) / (img_h * img_w)
            zhanbi_list.append(zhanbi)

        json.dump(zhanbi_list, open(json_path, 'w'))
        # f = open(json_path)
        # data = json.load(f)
        head_tail = os.path.split(video)
        print('Finish Video ', head_tail[1])

    print('Finish All Videos')


def resize_img(data_dir, train_rep, max_num=15):
    folders = sorted(os.listdir(data_dir))
    folders.remove('list.txt')
    # create reproduce path
    for folder in folders:
        re_video_path = os.path.join(train_rep, folder)
        isExist = os.path.exists(re_video_path)
        if not isExist:
            os.makedirs(re_video_path)

    # reproduce video paths
    rep_video_list = [os.path.join(train_rep, folder) for folder in folders]

    # original video paths
    video_list = [os.path.join(data_dir, folder) for folder in folders]
    for v_idx, video in enumerate(video_list):
        img_paths = sorted(glob.glob(os.path.join(video, '*.jpg')))
        img_paths = img_paths[0:max_num + 1]
        # reproduce image path
        rep_img_paths = []
        for item in img_paths:
            head_tail = os.path.split(item)
            rep_img_paths.append(os.path.join(rep_video_list[v_idx], head_tail[1]))

        # get first frame ground truth
        gt_path = os.path.join(video, 'groundtruth.txt')
        with open(gt_path) as f:
            gt_video = f.read().splitlines()
        first_gt = gt_video[0:16]
        f = open(os.path.join(rep_video_list[v_idx], 'init_gt.txt'), 'w')
        f.write(first_gt[0])
        f.close()

        f = open(os.path.join(rep_video_list[v_idx], 'init_gt.txt'), 'w')
        f.write(first_gt[0])
        f.close()


        # resize image
        # for idx, img_path in enumerate(img_paths):
        #     # using Image PIL resize img
        #     # img = Image.open(img_path)
        #     # img = img.resize((255, 255), resample=2)
        #     # img.save(rep_img_paths[idx])
        #
        #     # using interpolation
        #     img = cv2.imread(img_path)
        #     img = img2tensor(img)
        #     img = torch.nn.functional.interpolate(img, size=(255, 255), mode='bilinear')
        #     img = tensor2img(img)
        #     cv2.imwrite(rep_img_paths[idx], img)

        # read zhanbi json
        json_path = os.path.join(video, 'zhanbi.json')
        f = open(json_path)
        zhanbi_video = json.load(f)
        zhanbi_video = zhanbi_video[0:max_num + 1]
        zhanbi_arg = mean(zhanbi_video)
        # write zhanbi_avrg
        f = open(os.path.join(rep_video_list[v_idx], 'zhanbi.txt'), 'w')
        f.write(str(zhanbi_arg))
        f.close()

        head_tail = os.path.split(video)
        print('Finish Video ', head_tail[1])

    print('Finish All Videos')


if __name__ == '__main__':
    # data_dir = '/media/mengjie/Data/Downloads/train'
    # train_rep = '/media/mengjie/Data/Downloads/train_reproduce'
    # crop = '/media/mengjie/Data/Downloads/crop287'
    # crop_wrong = '/media/mengjie/Data/Downloads/crop287(wrong)'
    # crop_adv = '/media/mengjie/Data/Downloads/crop287_adv'

    # calculate zhanbi
    # data_dir = '/media/mengjie/Data/Downloads/val'
    # cal_zhanbi(data_dir)
    data_dir = '/media/mengjie/Data/Downloads/val'
    val_crop = '/media/mengjie/Data/Downloads/val_crop287'
    resize_img(data_dir, val_crop, 15)

    stop

    # resize image
    # resize_img(data_dir, train_rep, 15)
    # load got10k dataset
    # folders = sorted(os.listdir(train_rep))
    #     # folders.remove('init_gt.txt')
    # folders_list = [os.path.join(train_rep, folder) for folder in folders]
    # des_folderlist = [os.path.join(crop, folder) for folder in folders]
    # for v_idx, video in enumerate(folders_list):
    #     s_zhanbi = os.path.join(video, 'init_gt.txt')
    #     d_zhanbi = os.path.join(des_folderlist[v_idx], 'init_gt.txt')
    #     shutil.copy(s_zhanbi,d_zhanbi)

    folders = sorted(os.listdir(crop))

        # folders.remove('init_gt.txt')
    folders_list = [os.path.join(crop, folder) for folder in folders]
    des_folderlist = [os.path.join(crop_adv, folder) for folder in folders]

    for v_idx, video in enumerate(folders_list):
        s_zhanbi = os.path.join(video, 'zhanbi.txt')
        d_zhanbi = os.path.join(des_folderlist[v_idx], 'zhanbi.txt')
        shutil.copy(s_zhanbi,d_zhanbi)

        s_init_gt = os.path.join(video, 'init_gt.txt')
        d_init_gt = os.path.join(des_folderlist[v_idx], 'init_gt.txt')
        shutil.copy(s_init_gt,d_init_gt)

        s_examplar = os.path.join(video, '00000001.jpg')
        d_examplar = os.path.join(des_folderlist[v_idx], '00000000.jpg')
        shutil.copy(s_examplar,d_examplar)





    stop
    dataset = GOT10k_dataset(train_rep)
    tem = dataset[100]
    print(len(tem[1]))
