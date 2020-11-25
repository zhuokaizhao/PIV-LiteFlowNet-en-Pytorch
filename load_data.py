import glob
import json
import torch
import numpy as np
import flowiz as fz
from PIL import Image


def read_all(data_path):
    # Read the whole dataset
    try:
        img1_name_list = json.load(
            open(data_path + "/img1_name_list.json", 'r'))
        img2_name_list = json.load(
            open(data_path + "/img2_name_list.json", 'r'))
        gt_name_list = json.load(open(data_path + "/gt_name_list.json", 'r'))
        print('images/ground_truths json name lists already exist\n')
    except:
        gt_name_list = []
        img1_name_list = []
        img2_name_list = []

        gt_name_list.extend(glob.glob(data_path + '/*flow.flo'))
        img1_name_list.extend(glob.glob(data_path + '/*img1.tif'))
        img2_name_list.extend(glob.glob(data_path + '/*img2.tif'))
        gt_name_list.sort()
        img1_name_list.sort()
        img2_name_list.sort()
        assert (len(gt_name_list) == len(img1_name_list))
        assert (len(img2_name_list) == len(img1_name_list))

        # Serialize data into file:
        json.dump(img1_name_list, open(data_path + "/img1_name_list.json",
                                       'w'))
        json.dump(img2_name_list, open(data_path + "/img2_name_list.json",
                                       'w'))
        json.dump(gt_name_list, open(data_path + "/gt_name_list.json", 'w'))

    return img1_name_list, img2_name_list, gt_name_list


def construct_dataset(img1_name_list,
                      img2_name_list,
                      gt_name_list):

    amount = len(gt_name_list)
    total_data_index = np.arange(0, amount, 1)

    # load images and labels
    img1_name_list = np.asarray(img1_name_list)
    img2_name_list = np.asarray(img2_name_list)
    label_name_list = np.asarray(gt_name_list)

    # load the first sample to determine the image size
    temp = np.asarray(Image.open(img1_name_list[0])) * 1.0 / 255.0
    img_height, img_width = temp.shape
    # print(f'Input image has size ({img_height}, {img_width})')
    # load first image, second image and labels
    # concatenate image pair on channel axis
    image_pairs = np.zeros((amount, img_height, img_width, 2))
    labels = np.zeros((amount, img_height, img_width, 2))
    for i in range(amount):
        image_pairs[i, :, :, 0:1] = np.asarray(Image.open(img1_name_list[i])).reshape(img_height, img_width, 1) * 1.0/255.0
        image_pairs[i, :, :, 1:] = np.asarray(Image.open(img2_name_list[i])).reshape(img_height, img_width, 1) * 1.0/255.0
        labels[i] = fz.read_flow(label_name_list[i])

    # prepare pytorch training data (channel first)
    image_pairs = torch.from_numpy(image_pairs).float().permute(0, 3, 1, 2)
    labels = torch.from_numpy(labels).float().permute(0, 3, 1, 2)

    return image_pairs, labels


