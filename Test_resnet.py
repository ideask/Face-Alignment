import cv2
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from DataLoader.resnet import MyDatasets
from Models.resnet import resnet18


def show_result(images, show_size=(1024, 1024), blank_size=2, window_name="merge"):
    small_h, small_w = images[0].shape[:2]
    column = int(show_size[1] / (small_w + blank_size))
    row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    merge_img = np.zeros(tuple(shape), images[0].dtype)

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size)
                t_w_start = j * (small_w + blank_size)
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]
                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    if count < max_count:
        print("Total pictures ： %s" % (max_count - count))
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, merge_img)
    cv2.waitKey(0)


def validate(wlfw_val_dataloader, resnet_backbone):
    resnet_backbone.eval()

    with torch.no_grad():
        losses = []
        losses_ION = []
        img_show = []
        for img, landmark_gt in wlfw_val_dataloader:
            img.requires_grad = False
            img = img.cuda(non_blocking=True)

            landmark_gt.requires_grad = False
            landmark_gt = landmark_gt.cuda(non_blocking=True)

            resnet_backbone = resnet_backbone.cuda()

            landmarks = resnet_backbone(img)

            loss = torch.mean(
                torch.sqrt(torch.sum((landmark_gt - landmarks) ** 2, axis=1))
            )

            landmarks = landmarks.cpu().numpy()
            landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy()
            error_diff = np.sum(np.sqrt(np.sum((landmark_gt - landmarks) ** 2, axis=2)), axis=1)
            interocular_distance = np.sqrt(np.sum((landmarks[:, 6, :] - landmarks[:, 9, :]) ** 2, axis=1))
            # interpupil_distance = np.sqrt(np.sum((landmarks[:, 60, :] - landmarks[:, 72, :]) ** 2, axis=1))
            error_norm = np.mean(error_diff / interocular_distance)

            # show result
            show_img = np.array(np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
            show_img = (show_img * 256).astype(np.uint8)
            np.clip(show_img, 0, 255)

            cv2.imwrite("xxx.jpg", show_img)
            img_clone = cv2.imread("xxx.jpg")
            img_clone = cv2.resize(img_clone, (112, 112))

            pre_landmark = landmarks[0] * [112, 112]
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2)
            pre_landmark_gt = landmark_gt[0] * [112, 112]

            for (x, y) in pre_landmark.astype(np.int32):
                print("x:{0:}, y:{1:}".format(x, y))
                cv2.circle(img_clone, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in pre_landmark_gt.astype(np.int32):
                print("x:{0:}, y:{1:}".format(x, y))
                cv2.circle(img_clone, (x, y), 2, (0, 0, 255), -1)
            img_show.append(img_clone)

        show_result(img_show)

        losses.append(loss.cpu().numpy())
        losses_ION.append(error_norm)

        print("NME", np.mean(losses))
        print("ION", np.mean(losses_ION))


def main(args):
    checkpoint = torch.load(args.model_path)
    resnet_backbone = resnet18(num_classes=42).cuda()

    resnet_backbone.load_state_dict(checkpoint['resnet_backbone'])

    transform = transforms.Compose([transforms.ToTensor()])

    my_val_dataset = MyDatasets(args.test_dataset, transform)
    my_val_dataloader = DataLoader(
        my_val_dataset, batch_size=5, shuffle=True, num_workers=0)

    validate(my_val_dataloader, resnet_backbone)


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path', default="./CheckPoints/snapshot_resnet/checkpoint.pth.tar", type=str)
    parser.add_argument('--test_dataset', default='./Data/ODATA/TestData/labels.txt', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
