import numpy as np
import cv2
import sys

sys.path.append('..')

import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import DataLoader
from PIL import Image


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels

class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """
    def __call__(self, sample):
        image, landmarks, facecls = sample['image'], sample['landmarks'], sample['facecls']
        image = channel_norm(image)
        return {'image': image, 'landmarks': landmarks, 'facecls':facecls}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    def __call__(self, sample):
        image, landmarks, facecls = sample['image'], sample['landmarks'], sample['facecls']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks), 'facecls':torch.from_numpy(facecls)}

def load_data(file_list):
    if 'Train' in file_list or 'train' in file_list:
        tsfm = transforms.Compose([
            Normalize(),                # do channel normalization
            ToTensor()                 # convert to torch type: NxCxHxW
        ])
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = MyDatasets(file_list, transform=tsfm)
    return data_set

class MyDatasets(data.Dataset):
    def __init__(self, file_list, transform=None):
        self.line = None
        self.landmarks = None
        self.transform = transform
        self.sample = None
        self.facecls = None
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        # self.img = cv2.imread(self.line[0])
        self.facecls = np.asarray(self.line[-1], dtype=np.float32)
        self.img = np.asarray(Image.open(self.line[0]).convert('L'), dtype=np.float32)
        if self.facecls == 1:
            self.landmark = np.asarray(self.line[1:43], dtype=np.float32)

        elif self.facecls == 0:
            self.landmark = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32)

        self.sample = {'image': self.img, 'landmarks': self.landmark, 'facecls': self.facecls}
        # 数据预处理已经做了数据增强了，所以这里只将数据转成tensor
        if self.transform:
            self.sample = self.transform(self.sample)
        return self.sample

    def __len__(self):
        return len(self.lines)


if __name__ == '__main__':
    file_list = '../Data/ODATA/TestData/labels.txt'
    # file_list = '../Data/ODATA/TrainData/labels.txt'
    mydataset = load_data(file_list)
    dataloader = DataLoader(mydataset, batch_size=5, shuffle=True, num_workers=0, drop_last=False)
    img_size = 512
    for samples in dataloader:
        for i in range(len(samples['image'])):
            img = samples['image'][i]
            landmarks = samples['landmarks'][i]
            facecls = samples['facecls'][i]
            img = img.numpy() #Tensor to array
            img = np.squeeze(img) #delete the empty asix
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (img_size, img_size))
            if facecls == 1:
                landmarks = landmarks.reshape(21, 2)
                for keypoint in landmarks:
                    cv2.circle(img, (keypoint[0]* img_size, keypoint[1]* img_size),  1, (0, 0, 255), 3)
            cv2.imshow('result', img)
            key = cv2.waitKey()
            cv2.destroyAllWindows()

