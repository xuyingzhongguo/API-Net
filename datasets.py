import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from PIL import Image
import numpy as np
import cv2
import sys
import os
from glob import glob


def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB', (224, 224), 'white')
    return img


# b: CIELab, v: Hsv, cb: ycrcb
def bvcb_loader(path):
    try:
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        h, w = img[:, :, 0].shape
        new_arr = np.zeros((h, w, 3))
        new_arr[:, :, 0] = lab_img[:, :, 2]
        new_arr[:, :, 1] = hsv_img[:, :, 2]
        new_arr[:, :, 2] = ycrcb_img[:, :, 2]
        new_arr = new_arr.astype('uint8')
        img = Image.fromarray(new_arr)
    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB', (224, 224), 'white')
    return img


def hsv_loader(path):
    try:
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = Image.fromarray(hsv_img)
    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB', (224, 224), 'white')
    return img


def lab_loader(path):
    try:
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        img = Image.fromarray(lab_img)
    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB', (224, 224), 'white')
    return img


def ycbcr_loader(path):
    try:
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img = Image.fromarray(ycrcb_img)
    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB', (224, 224), 'white')
    return img


def single_channels_loader(path):
    try:
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img[:, :, 0].shape
        new_arr = np.zeros((h, w, 3))

        # new_arr[:, :, 0] = lab_img[:, :, 0]
        # new_arr[:, :, 1] = lab_img[:, :, 0]
        # new_arr[:, :, 2] = lab_img[:, :, 0]

        new_arr[:, :, 0] = hsv_img[:, :, 2]
        new_arr[:, :, 1] = hsv_img[:, :, 2]
        new_arr[:, :, 2] = hsv_img[:, :, 2]

        new_arr = new_arr.astype('uint8')
        img = Image.fromarray(new_arr)

    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        new_arr = np.zeros((224, 224, 9))
        new_arr = new_arr.astype('uint8')
        img = Image.fromarray(new_arr)
        print('error')
        print(img.size)
        return img
    return img


def nine_channels_loader(path):
    try:
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        h, w = img[:, :, 0].shape
        new_arr = np.zeros((h, w, 9))
        # new_arr[:, :, 0] = lab_img[:, :, 0]
        # new_arr[:, :, 1] = lab_img[:, :, 1]
        # new_arr[:, :, 2] = lab_img[:, :, 2]
        # new_arr[:, :, 3] = hsv_img[:, :, 0]
        # new_arr[:, :, 4] = hsv_img[:, :, 1]
        # new_arr[:, :, 5] = hsv_img[:, :, 2]
        # new_arr[:, :, 6] = ycrcb_img[:, :, 0]
        # new_arr[:, :, 7] = ycrcb_img[:, :, 1]
        # new_arr[:, :, 8] = ycrcb_img[:, :, 2]

        # rgb+lab+hsv
        # new_arr[:, :, 0] = img[:, :, 0]
        # new_arr[:, :, 1] = img[:, :, 1]
        # new_arr[:, :, 2] = img[:, :, 2]
        # new_arr[:, :, 3] = lab_img[:, :, 0]
        # new_arr[:, :, 4] = lab_img[:, :, 1]
        # new_arr[:, :, 5] = lab_img[:, :, 2]
        # new_arr[:, :, 6] = hsv_img[:, :, 0]
        # new_arr[:, :, 7] = hsv_img[:, :, 1]
        # new_arr[:, :, 8] = hsv_img[:, :, 2]

        # rgb+lab+ycbcr
        new_arr[:, :, 0] = img[:, :, 0]
        new_arr[:, :, 1] = img[:, :, 1]
        new_arr[:, :, 2] = img[:, :, 2]
        new_arr[:, :, 3] = lab_img[:, :, 0]
        new_arr[:, :, 4] = lab_img[:, :, 1]
        new_arr[:, :, 5] = lab_img[:, :, 2]
        new_arr[:, :, 6] = ycrcb_img[:, :, 0]
        new_arr[:, :, 7] = ycrcb_img[:, :, 1]
        new_arr[:, :, 8] = ycrcb_img[:, :, 2]

        new_arr = new_arr.astype('uint8')
        img = new_arr

    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        new_arr = np.zeros((224, 224, 9))
        new_arr = new_arr.astype('uint8')
        img = Image.fromarray(new_arr)
        print('error')
        print(img.size)
        return img
    return img


def rgb_hsv_loader(path):
    try:
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img[:, :, 0].shape
        new_arr = np.zeros((h, w, 6))

        # rgb+hsv
        new_arr[:, :, 0] = img[:, :, 0]
        new_arr[:, :, 1] = img[:, :, 1]
        new_arr[:, :, 2] = img[:, :, 2]
        new_arr[:, :, 3] = hsv_img[:, :, 0]
        new_arr[:, :, 4] = hsv_img[:, :, 1]
        new_arr[:, :, 5] = hsv_img[:, :, 2]

        new_arr = new_arr.astype('uint8')
        img = new_arr

    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        new_arr = np.zeros((224, 224, 6))
        new_arr = new_arr.astype('uint8')
        img = Image.fromarray(new_arr)
        print('error')
        print(img.size)
        return img
    return img


def rgb_lab_loader(path):
    try:
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        h, w = img[:, :, 0].shape
        new_arr = np.zeros((h, w, 6))

        # rgb+lab
        new_arr[:, :, 0] = img[:, :, 0]
        new_arr[:, :, 1] = img[:, :, 1]
        new_arr[:, :, 2] = img[:, :, 2]
        new_arr[:, :, 3] = lab_img[:, :, 0]
        new_arr[:, :, 4] = lab_img[:, :, 1]
        new_arr[:, :, 5] = lab_img[:, :, 2]

        new_arr = new_arr.astype('uint8')
        img = new_arr

    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        new_arr = np.zeros((224, 224, 6))
        new_arr = new_arr.astype('uint8')
        img = Image.fromarray(new_arr)
        print('error')
        print(img.size)
        return img
    return img


def rgb_ycbcr_loader(path):
    try:
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        h, w = img[:, :, 0].shape
        new_arr = np.zeros((h, w, 6))

        # lab+ycbcr
        new_arr[:, :, 0] = img[:, :, 0]
        new_arr[:, :, 1] = img[:, :, 1]
        new_arr[:, :, 2] = img[:, :, 2]
        new_arr[:, :, 3] = ycrcb_img[:, :, 0]
        new_arr[:, :, 4] = ycrcb_img[:, :, 1]
        new_arr[:, :, 5] = ycrcb_img[:, :, 2]

        new_arr = new_arr.astype('uint8')
        img = new_arr

    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        new_arr = np.zeros((224, 224, 6))
        new_arr = new_arr.astype('uint8')
        img = Image.fromarray(new_arr)
        print('error')
        print(img.size)
        return img
    return img


def temporal_9(path):
    try:
        img_0 = cv2.imread(path, cv2.COLOR_BGR2RGB)
        path_split = path.split('/')
        frame_num = path_split[-1].split('_')[0][5:8]

        next_frame = int(frame_num) + 10
        next_frame_path = os.path.join('/', *path_split[:-1], 'frame' + str(next_frame) + '*.png')
        path_2 = glob(next_frame_path)[0]

        next2_frame = int(frame_num) + 20
        next2_frame_path = os.path.join('/', *path_split[:-1], 'frame' + str(next2_frame) + '*.png')
        path_3 = glob(next2_frame_path)[0]

        img_1 = cv2.imread(path_2, cv2.COLOR_BGR2RGB)
        img_2 = cv2.imread(path_3, cv2.COLOR_BGR2RGB)

        print('-----------------temporal1-----------------')
        print(path)
        print(path_2)
        print(path_3)
        print('-----------------temporal2-----------------')

        h, w = img_0[:, :, 0].shape
        new_arr = np.zeros((h, w, 9))

        new_arr[:, :, 0] = img_0[:, :, 0]
        new_arr[:, :, 1] = img_0[:, :, 1]
        new_arr[:, :, 2] = img_0[:, :, 2]
        new_arr[:, :, 3] = img_1[:, :, 0]
        new_arr[:, :, 4] = img_1[:, :, 1]
        new_arr[:, :, 5] = img_1[:, :, 2]
        new_arr[:, :, 6] = img_2[:, :, 0]
        new_arr[:, :, 7] = img_2[:, :, 1]
        new_arr[:, :, 8] = img_2[:, :, 2]

        new_arr = new_arr.astype('uint8')
        img = new_arr

    except:
        with open('read_error.txt', 'a') as fid:
            fid.write(path+'\n')
        new_arr = np.zeros((224, 224, 9))
        new_arr = new_arr.astype('uint8')
        img = Image.fromarray(new_arr)
        print('error')
        print(img.size)
        return img
    return img


def label_setting(label, image_path):
    if label != 0:
        if 'Celeb-DF' in image_path:
            label = 1
        elif any(item in image_path for item in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter']):
        # if any(item in image_path for item in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter']):
            label = 2
        elif 'DeepFakeDetection' in image_path:
            label = 3
        # elif 'DeeperF-1.0' in image_path:
        #     label = 3
        elif 'DFDC' in image_path:
            label = 4
        else:
            print('Seems like you have wrong paths or labels baby')

    return label


class RandomDataset(Dataset):
    def __init__(self, val_list, loader, struc_label, transform=None):
        self.transform = transform
        if loader == 'default_loader':
            print('self.dataloader = default_loader')
            self.dataloader = default_loader
        elif loader == 'bvcb_loader':
            print('self.dataloader = bvcb_loader')
            self.dataloader = bvcb_loader
        elif loader == 'nine_channels':
            print('self.dataloader = nine_channels_loader')
            self.dataloader = nine_channels_loader
        elif loader == 'rgb_lab':
            print('self.dataloader = rgb_lab_loader')
            self.dataloader = rgb_lab_loader
        elif loader == 'rgb_hsv':
            print('self.dataloader = rgb_hsv_loader')
            self.dataloader = rgb_hsv_loader
        elif loader == 'rgb_ycbcr':
            print('self.dataloader = rgb_ycbcr_loader')
            self.dataloader = rgb_ycbcr_loader
        elif loader == 'single_channel':
            print('self.dataloader = single_channels_loader')
            self.dataloader = single_channels_loader
        elif loader == 'hsv_loader':
            print('self.dataloader = hsv_loader')
            self.dataloader = hsv_loader
        elif loader == 'lab_loader':
            print('self.dataloader = lab_loader')
            self.dataloader = lab_loader
        elif loader == 'ycbcr_loader':
            print('self.dataloader = ycbcr_loader')
            self.dataloader = ycbcr_loader
        elif loader == 'temporal_9':
            print('self.dataloader = temporal_9')
            self.dataloader = temporal_9
        else:
            sys.exit('wrong image loader baby')

        # with open(val_list, 'r') as fid:
        #     self.imglist = fid.readlines()
        self.imglist = val_list
        self.struc_label = struc_label

    def __getitem__(self, index):
        image_name, label = self.imglist[index].strip().split()
        image_path = image_name
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        if self.struc_label == 'cross':
            label = label_setting(label, image_path)
        label = torch.LongTensor([label])

        return [img, label]

    def __len__(self):
        return len(self.imglist)


class BatchDataset(Dataset):
    def __init__(self, train_list, loader, struc_label, transform=None):
        self.transform = transform
        if loader == 'default_loader':
            self.dataloader = default_loader
        elif loader == 'bvcb_loader':
            self.dataloader = bvcb_loader
        elif loader == 'nine_channels':
            print('self.dataloader = nine_channels_loader')
            self.dataloader = nine_channels_loader
        elif loader == 'rgb_lab':
            print('self.dataloader = six_channels_loader')
            self.dataloader = rgb_lab_loader
        elif loader == 'rgb_hsv':
            print('self.dataloader = six_channels_loader')
            self.dataloader = rgb_hsv_loader
        elif loader == 'rgb_ycbcr':
            print('self.dataloader = six_channels_loader')
            self.dataloader = rgb_ycbcr_loader
        elif loader == 'single_channel':
            print('self.dataloader = single_channels_loader')
            self.dataloader = single_channels_loader
        elif loader == 'hsv_loader':
            print('self.dataloader = hsv_loader')
            self.dataloader = hsv_loader
        elif loader == 'lab_loader':
            print('self.dataloader = lab_loader')
            self.dataloader = lab_loader
        elif loader == 'ycbcr_loader':
            print('self.dataloader = ycbcr_loader')
            self.dataloader = ycbcr_loader
        else:
            sys.exit('wrong image loader baby')

        self.struc_label = struc_label

        # with open(train_list, 'r') as fid:
        #     self.imglist = fid.readlines()
        self.imglist = train_list

        self.labels = []
        for line in self.imglist:
            image_path, label = line.strip().split()
            label = int(label)
            if self.struc_label == 'cross':
                label = label_setting(label, image_path)
            self.labels.append(int(label))
        self.labels = np.array(self.labels)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, index):
        image_name, label = self.imglist[index].strip().split()
        image_path = image_name
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        if self.struc_label == 'cross':
            label = label_setting(label, image_path)
        label = torch.LongTensor([label])

        return [img, label]


    def __len__(self):
        return len(self.imglist)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            # print(f'self.labels_set, self.n_classes {self.labels_set, self.n_classes}')
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


class RandomDataset_test(Dataset):
    def __init__(self, val_list, loader, transform=None):
        self.transform = transform
        if loader == 'default_loader':
            self.dataloader = default_loader
        elif loader == 'bvcb_loader':
            self.dataloader = bvcb_loader
        elif loader == 'nine_channels':
            print('self.dataloader = nine_channels_loader')
            self.dataloader = nine_channels_loader
        elif loader == 'rgb_lab':
            print('self.dataloader = six_channels_loader')
            self.dataloader = rgb_lab_loader
        elif loader == 'rgb_hsv':
            print('self.dataloader = six_channels_loader')
            self.dataloader = rgb_hsv_loader
        elif loader == 'rgb_ycbcr':
            print('self.dataloader = six_channels_loader')
            self.dataloader = rgb_ycbcr_loader
        elif loader == 'single_channel':
            print('self.dataloader = single_channels_loader')
            self.dataloader = single_channels_loader
        elif loader == 'hsv_loader':
            print('self.dataloader = hsv_loader')
            self.dataloader = hsv_loader
        elif loader == 'lab_loader':
            print('self.dataloader = lab_loader')
            self.dataloader = lab_loader
        elif loader == 'ycbcr_loader':
            print('self.dataloader = ycbcr_loader')
            self.dataloader = ycbcr_loader
        else:
            sys.exit('wrong image loader baby')

        with open(val_list, 'r') as fid:
            self.imglist = fid.readlines()

    def __getitem__(self, index):
        split = self.imglist[index].strip().split()
        if len(split) == 3:
            split[0] = split[0] + ' ' + split[1]
            split[1] = split[2]
            split.pop(2)
        image_name, label = split
        image_path = image_name
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        # label_setting(label, image_path)
        label = torch.LongTensor([label])

        return [img, label, image_name]


    def __len__(self):
        return len(self.imglist)
