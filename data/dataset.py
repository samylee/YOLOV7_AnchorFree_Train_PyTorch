import torch
import numpy as np
import cv2
import os
import random
from torch.utils.data import Dataset

from utils.utils import xywhn2xyxy, xyxy2xywhn, random_perspective, letterbox, augment_hsv


class VOCDataset(Dataset):
    def __init__(self, label_list, net_size=640):
        super(VOCDataset, self).__init__()
        self.mosaic = True
        self.net_size = net_size
        self.mosaic_border = [-self.net_size // 2, -self.net_size // 2]
        self.albumentations = Albumentations(size=self.net_size)

        with open(label_list, 'r') as f:
            image_path_lines = f.readlines()

        self.images_path = []
        self.labels = []
        for image_path_line in image_path_lines:
            image_path = image_path_line.strip().split()[0]
            label_path = image_path.replace('JPEGImages', 'labels').replace('jpg', 'txt')
            if not os.path.exists(label_path):
                continue

            self.images_path.append(image_path)
            with open(label_path, 'r') as f:
                label_lines = f.readlines()

            labels_tmp = np.empty((len(label_lines), 5), dtype=np.float32)
            for i, label_line in enumerate(label_lines):
                labels_tmp[i] = [float(x) for x in label_line.strip().split()]
            self.labels.append(labels_tmp)

        assert len(self.images_path) == len(self.labels), 'images_path\'s length dont match labels\'s length'
        self.indices = range(len(self.images_path))

    def __getitem__(self, idx):
        # mosaic data augment
        if self.mosaic:
            image, labels = self.load_mosaic(idx)
            # mixup augmentation
            if random.random() < 0.1:
                image, labels = self.mixup(image, labels, *self.load_mosaic(random.randint(0, len(self.images_path) - 1)))
        else:
            image, labels = self.load_origin(idx)

        nL = len(labels)
        if nL:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], image.shape[1], image.shape[0], clip=True, eps=1E-3)

        # Albumentations
        image, labels = self.albumentations(image, labels)
        nL = len(labels)  # update after albumentations

        # Augment hsv
        augment_hsv(image, hgain=0.015, sgain=0.7, vgain=0.4)

        # Augment flip
        if random.random() < 0.5:
            image = np.fliplr(image)
            if nL:
                labels[:, 1] = 1 - labels[:, 1]

        # labels to torch
        targets = torch.zeros((nL, 6))
        if nL:
            targets[:, 1:] = torch.from_numpy(labels)

        # images to torch
        image = np.ascontiguousarray(image.transpose((2, 0, 1))[::-1])
        inputs = torch.from_numpy(image).float().div(255)

        return inputs, targets

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        image, label = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(image, 0), torch.cat(label, 0)

    def load_image(self, index):
        path = self.images_path[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.net_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        return img, img.shape[:2]

    def load_origin(self, index):
        img, (h, w) = self.load_image(index)

        # Letterbox
        img, ratio, pad = letterbox(img, self.net_size, auto=False, scaleup=True)

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        img, labels = random_perspective(img, labels,
                                         degrees=0,
                                         translate=0.1,
                                         scale=0.9,
                                         shear=0,
                                         perspective=0)

        return img, labels

    def load_mosaic(self, index):
        # loads images in a 4-mosaic
        labels4 = []
        s = self.net_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)
        # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
            labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in labels4[:, 1:]:
            np.clip(x, 0, 2 * s, out=x)

        # Augment
        img4, labels4 = random_perspective(img4, labels4,
                                           degrees=0,
                                           translate=0.1,
                                           scale=0.9,
                                           shear=0,
                                           perspective=0,
                                           border=self.mosaic_border)  # border to remove

        return img4, labels4

    def mixup(self, im, labels, im2, labels2):
        # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        im = (im * r + im2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((labels, labels2), 0)
        return im, labels


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=640):
        self.transform = None
        try:
            import albumentations as A
            T = [
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        except ImportError:  # package not installed, skip
            print('can not import albumentations')
            pass
        except Exception as e:
            print(e)

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels