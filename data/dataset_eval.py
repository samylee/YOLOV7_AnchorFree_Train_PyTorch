import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset

from utils.utils import xywhn2xyxy, xyxy2xywhn, letterbox


class VOCDataset(Dataset):
    def __init__(self, label_list, net_size=640, batch_size=8, stride=32, pad=0.5):
        super(VOCDataset, self).__init__()
        self.net_size = net_size

        with open(label_list, 'r') as f:
            image_path_lines = f.readlines()

        self.images_path = []
        self.labels = []
        self.shapes = []
        for image_path_line in image_path_lines:
            image_path, w, h = image_path_line.strip().split()
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
            self.shapes.append([int(w), int(h)])

        self.shapes = np.array(self.shapes)
        n = len(self.shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Sort by aspect ratio
        s = self.shapes  # wh
        ar = s[:, 1] / s[:, 0]  # aspect ratio
        irect = ar.argsort()
        self.images_path = [self.images_path[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        self.shapes = s[irect]  # wh
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * net_size / stride + pad).astype(int) * stride

        assert len(self.images_path) == len(self.labels), 'images_path\'s length dont match labels\'s length'

    def __getitem__(self, idx):
        idx = self.indices[idx]
        # Load image
        image, (h0, w0), (h, w) = self.load_image(idx)

        # Letterbox
        shape = self.batch_shapes[self.batch[idx]]
        image, ratio, pad = letterbox(image, shape, auto=False, scaleup=False)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[idx].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=image.shape[1], h=image.shape[0], clip=True, eps=1E-3)

        targets = torch.zeros((nl, 6))
        if nl:
            targets[:, 1:] = torch.from_numpy(labels)

        # images to torch
        image = np.ascontiguousarray(image.transpose((2, 0, 1))[::-1])
        inputs = torch.from_numpy(image).float().div(255)

        return inputs, targets, shapes

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        image, label, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(image, 0), torch.cat(label, 0), shapes

    def load_image(self, index):
        path = self.images_path[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.net_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        return img, (h0, w0), img.shape[:2]