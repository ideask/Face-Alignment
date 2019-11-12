import numpy as np
import cv2
import sys

sys.path.append('..')

from torch.utils import data
from torch.utils.data import DataLoader


class MyDatasets(data.Dataset):
    def __init__(self, file_list, transforms=None):
        self.line = None
        self.landmarks = None
        self.euler_angle = None
        self.transforms = transforms
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.img = cv2.imread(self.line[0])
        self.landmark = np.asarray(self.line[1:43], dtype=np.float32)
        self.euler_angle = np.asarray(self.line[43:], dtype=np.float32)
        # 数据预处理已经做了数据增强了，所以这里只将数据转成tensor
        if self.transforms:
            self.img = self.transforms(self.img)
        return (self.img, self.landmark, self.euler_angle)

    def __len__(self):
        return len(self.lines)


if __name__ == '__main__':
    file_list = '../Data/ODATA/TestData/labels.txt'
    mydataset = MyDatasets(file_list)
    dataloader = DataLoader(mydataset, batch_size=256, shuffle=True, num_workers=0, drop_last=False)
    img_size = 512
    for images, landmarks, euler_angles in dataloader:
        # print("img shape", img.shape)
        # print("landmark size", landmark.size())
        # print("euler_angle", euler_angle.size())
        for i in range(len(images)):
            image = images[i].numpy()
            image = cv2.resize(image, (img_size, img_size))
            landmark = landmarks[i].numpy()
            landmark = landmark.reshape(-1, 2)
            euler_angle = euler_angles[i].numpy()
            pitch = euler_angle[0]
            yaw = euler_angle[1]
            roll = euler_angle[2]
            count = 0
            for j in landmark:
                cv2.putText(image, str(count), (int(j[0] * img_size), int(j[1] * img_size)), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 255), 1, 1)
                cv2.circle(image, (int(j[0] * img_size), int(j[1] * img_size)), 1, (0, 255, 0), 3)
                count += 1
            cv2.putText(image, 'pitch:{}'.format(pitch),  (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, 1)
            cv2.putText(image, 'yaw:{}'.format(yaw), (0, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, 1)
            cv2.putText(image, 'roll:{}'.format(roll), (0, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, 1)
            cv2.imshow('result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

