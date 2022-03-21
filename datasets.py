import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from PIL import Image
import numpy as np
import cv2
import sys


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


def nine_channels_loader(path):
    try:
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        h, w = img[:, :, 0].shape
        new_arr = np.zeros((h, w, 9))
        new_arr[:, :, 0] = lab_img[:, :, 0]
        new_arr[:, :, 1] = lab_img[:, :, 1]
        new_arr[:, :, 2] = lab_img[:, :, 2]
        new_arr[:, :, 3] = hsv_img[:, :, 0]
        new_arr[:, :, 4] = hsv_img[:, :, 1]
        new_arr[:, :, 5] = hsv_img[:, :, 2]
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
        else:
            sys.exit('wrong image loader baby')

        with open(val_list, 'r') as fid:
            self.imglist = fid.readlines()

    def __getitem__(self, index):
        image_name, label = self.imglist[index].strip().split()
        image_path = image_name
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)
        # label_setting(label, image_path)
        label = torch.LongTensor([label])

        return [img, label, image_name]


    def __len__(self):
        return len(self.imglist)
