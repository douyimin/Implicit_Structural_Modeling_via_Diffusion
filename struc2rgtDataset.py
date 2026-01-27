import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import random
import math
from torchvision import transforms
import torch.nn.functional as F
import copy

import torchvision.transforms.functional as transF


# transforms.Normalize
def RandomHorizontalFlipCoord2D(rgt, fault, p=0.5):
    if random.random() < p:
        return cv2.flip(rgt, flipCode=1), cv2.flip(fault, flipCode=1)
    return rgt, fault


def RandomVerticalFlipCoord2D(rgt, fault, p=0.5):
    if random.random() < p:
        return 1. - cv2.flip(rgt, flipCode=-1), cv2.flip(fault, flipCode=-1)
    return rgt, fault


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range + 1e-6)


def z_score_clip(data, clp_s=3.):
    z = (data - np.mean(data)) / np.std(data)
    return normalization(np.clip(z, a_min=-clp_s, a_max=clp_s))


def RotateBound(data, angle, inter):
    scale = math.sin(abs(angle) * math.pi / 180) + math.cos(abs(angle) * math.pi / 180)
    (h, w) = data.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(data, M, (nW, nH), flags=inter)


def RandomRotateAg2D(rgt, fault, p=0.5):
    if random.random() < p:
        return rgt, fault
    cube_size, _ = rgt.shape
    angle = random.randint(-30, 30)
    rgt = RotateBound(rgt, angle, inter=cv2.INTER_LINEAR)
    fault = RotateBound(fault, angle, inter=cv2.INTER_NEAREST)

    l = int((rgt.shape[0] - cube_size) / 2)
    rgt = rgt[l:l + cube_size, l:l + cube_size]
    fault = fault[l:l + cube_size, l:l + cube_size]
    return normalization(rgt), fault


def randomHorizMask(horiz):
    if random.random() <= 0.1: return horiz
    h, w = horiz.shape
    mask = np.zeros_like(horiz)
    mask_len = random.uniform(0.1, 0.9)
    start = random.uniform(0, 1. - mask_len)
    mask[:, int(start * w):int((start + mask_len) * w)] = 1
    horiz = horiz * mask
    return horiz


def getHoriz2D(rgt, num=8, bin=0.01):
    horiz_rgt = rgt
    horiz_map = np.zeros_like(horiz_rgt)
    mask = np.zeros_like(horiz_map)

    # horiz_v = list(range(1, num + 1))
    # random.shuffle(horiz_v)
    for i in range(num):
        start = random.uniform(0, 1 - bin)
        horiz_mask = ((horiz_rgt > start) == (horiz_rgt <= start + bin)).astype(np.float32)
        if np.sum(horiz_mask) == 0: continue
        horiz_value = np.sum(horiz_rgt * horiz_mask) / (np.sum(horiz_mask))

        horiz_map = np.where(horiz_mask, horiz_value, horiz_map)
        mask = np.where(horiz_mask, 1., mask)

    return horiz_map, mask


def randomCropRGT(fault, rgt, size):
    shape = fault.shape
    size = np.array(size)
    lim = shape - size
    w = random.randint(0, lim[0])
    h = random.randint(0, lim[1])
    return fault[w:w + size[0], h:h + size[1]], rgt[w:w + size[0], h:h + size[1]]


def RandomPerspectiveTransform2D(rgt, fault, org_point=np.float32([(0, 0), (0, 512), (512, 0), (512, 512)]), p=0.5):
    if random.random() < p:
        return rgt, fault

    def transformCVPoint(point):
        tmp = np.zeros_like(point)
        tmp[:, 0] = point[:, 1]
        tmp[:, 1] = point[:, 0]
        return tmp

    H, W = rgt.shape
    max_len = int((H + W) / 3)
    target_point = copy.deepcopy(org_point)
    if random.randint(0, 1):
        target_point[0, 0] = target_point[0, 0] - random.randint(0, max_len)
        target_point[2, 0] = target_point[2, 0] + random.randint(0, max_len)
    else:
        target_point[1, 0] = target_point[1, 0] - random.randint(0, max_len)
        target_point[3, 0] = target_point[3, 0] + random.randint(0, max_len)

    if random.randint(0, 1):
        target_point[0, 1] = target_point[0, 1] - random.randint(0, max_len)
        target_point[2, 1] = target_point[2, 1] - random.randint(0, max_len)
    else:
        target_point[1, 1] = target_point[1, 1] + random.randint(0, max_len)
        target_point[3, 1] = target_point[3, 1] + random.randint(0, max_len)

    org_point = transformCVPoint(org_point)
    target_point = transformCVPoint(target_point)

    M = cv2.getPerspectiveTransform(org_point, target_point)

    rgt = cv2.warpPerspective(rgt, M, (H, W), flags=cv2.INTER_LINEAR)
    fault = (cv2.warpPerspective(fault, M, (H, W), flags=cv2.INTER_AREA) > 0.1).astype(np.float32)

    return rgt, fault


def RandomRotateRGT(fault, RGT, p=0.5):
    if random.random() < p:
        return fault, RGT
    return fault.transpose((0, 2, 1)), RGT.transpose((0, 2, 1))


def RandomVerticalFlipCoord(*aug_list, p=0.5):
    if random.random() < p:
        return [data for data in aug_list]
    return [transF.vflip(torch.from_numpy(data)[None])[0].numpy() for data in aug_list]


def RandomHorizontalFlipCoord(*aug_list, p=0.5):
    if random.random() < p:
        return [data for data in aug_list]
    return [transF.hflip(torch.from_numpy(data)[None])[0].numpy() for data in aug_list]


def RandomRotateAgSynTlineRGT(RGT, fault, p=0.5):
    if random.random() < p:
        return RGT, fault
    cube_size, _ = fault.shape
    angle = random.randint(-25, 25)
    fault = RotateBound(fault, angle, inter=cv2.INTER_NEAREST)
    RGT = RotateBound(RGT, angle, inter=cv2.INTER_LINEAR)
    l = int((fault.shape[1] - cube_size) / 2)
    fault = fault[l:l + cube_size, l:l + cube_size]
    RGT = RGT[l:l + cube_size, l:l + cube_size]
    return RGT, fault


def RandomTimeflipCoord(fault, rgt, p=0.5):
    if random.random() < p:
        return fault, rgt
    fault = transF.vflip(torch.from_numpy(fault)[None].permute((0, 2, 1, 3))).permute((0, 2, 1, 3))
    rgt = 1. - transF.vflip(torch.from_numpy(rgt)[None].permute((0, 2, 1, 3))).permute((0, 2, 1, 3))
    return fault[0].numpy(), rgt[0].numpy()


def getTrainFrame3D(rgt, fault, target_f=32):
    rgt = rgt.transpose(2, 0, 1)
    fault = fault.transpose(2, 0, 1)
    f, _, _ = rgt.shape
    start_f = random.randint(0, f - target_f)
    return rgt[start_f:start_f + target_f], fault[start_f:start_f + target_f]


def get_frame_mask(frame):
    sample_mask = np.zeros_like(frame)
    f, _, _ = sample_mask.shape
    sample_f = random.sample(range(f), random.randint(1, f))
    for i in sample_f: sample_mask[i] = 1.
    return sample_mask


def getHoriz3D(rgt, fault, horiz_num=24, bin=0.0075):
    horiz_rgt = normalization(rgt)
    horiz_map = np.zeros_like(horiz_rgt)
    merge_horiz_mask = np.zeros_like(horiz_map)
    for i in range(horiz_num):
        start = random.uniform(0, 1 - bin)
        horiz_mask = ((horiz_rgt > start) == (horiz_rgt <= start + bin)).astype(np.float32)
        horiz_value = np.sum(horiz_rgt * horiz_mask) / (np.sum(horiz_mask) + 1e-6)
        horiz_map = np.where(horiz_mask, horiz_value, horiz_map)
        merge_horiz_mask = np.where(horiz_mask, 1., merge_horiz_mask)
    frame_mask = get_frame_mask(rgt)

    horiz_map = np.where(merge_horiz_mask, horiz_map * 2. - 1., horiz_map)
    fault = fault * 2. - 1.

    merge_horiz_mask = merge_horiz_mask * frame_mask
    horiz_map = horiz_map * frame_mask
    fault = fault * frame_mask

    return horiz_map, merge_horiz_mask, fault, frame_mask


class seisDataset(Dataset):
    def __init__(self, data_path):
        self.data_list2D = [os.path.join(data_path, fie_name) \
                            for fie_name in os.listdir(data_path)]
        self.aug_list2D = [
            RandomHorizontalFlipCoord2D,
            RandomVerticalFlipCoord2D,
            RandomRotateAg2D,
            RandomPerspectiveTransform2D,
            RandomRotateAgSynTlineRGT
        ]

    def __len__(self):
        return 500000

    def __getitem__(self, index):
        # if random.randint(0, 1):
        idx = random.randint(0, len(self.data_list2D) - 1)
        data = np.load(self.data_list2D[idx])
        fault = data['fault'].astype(np.float32)
        rgt = normalization(data['rgt'].astype(np.float32))
        if rgt.shape[0] != 512:
            rgt = cv2.resize(rgt, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
            fault = cv2.resize(fault, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)

        random.shuffle(self.aug_list2D)
        for aug_func in self.aug_list2D:
            rgt, fault = aug_func(rgt, fault)

        rgt = normalization(rgt)
        rgt = cv2.resize(rgt, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        fault = cv2.resize(fault, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        fault = (fault > 0.).astype(np.float32)

        horiz, mask = getHoriz2D(rgt, num=random.randint(1, 24))
        rgt = torch.repeat_interleave(torch.from_numpy(rgt)[None], 3, dim=0) * 2. - 1.
        fault = torch.from_numpy(fault)[None] * 2. - 1.
        horiz = torch.from_numpy(horiz)[None] * 2. - 1.

        return dict(jpg=rgt, fault=fault, horiz=horiz)
