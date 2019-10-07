import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
from tqdm import tqdm

import os
import os.path as osp

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(video_dir, image_list, start, num):
  frames = []
  for i in range(start, start+num):
    img = cv2.imread(image_list[i])
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def load_flow_frames(video_dir, image_list, start, num):
  frames = []
  for i in range(start, start+num):
    img = cv2.imread(image_list[i])
    imgx = img[:, :, 0]
    imgy = img[:, :, 1]

    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)

    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=157):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid)):
            continue
        num_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == 'flow':
            num_frames = num_frames//2

        label = np.zeros((num_classes,num_frames), np.float32)

        fps = num_frames/data[vid]['duration']
        for ann in data[vid]['actions']:
            for fr in range(0,num_frames,1):
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[ann[0], fr] = 1 # binary classification
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1

    return dataset


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def list_directories(root, full_path = False):
    if full_path:
        return [osp.join(root, d) for d in os.listdir(root) if osp.exists(osp.join(root, d))]
    else:
        return [d for d in os.listdir(root) if osp.exists(osp.join(root, d))]

def list_images(root, full_path = True, ext = None):
    if ext is None:
        ext = IMG_EXTENSIONS

    if full_path:
        return [osp.join(root, i) for i in os.listdir(root) \
                if osp.exists(osp.join(root, i)) and \
                has_file_allowed_extension(osp.join(root, i), ext)]
    else:
        return [i for i in os.listdir(root) if osp.exists(osp.join(root, i)) \
                and has_file_allowed_extension(osp.join(root, i), ext)]

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
VIDEO_EXTENSION = ['.mp4', '.mov']

class TVQA(data_utl.Dataset):
    def __init__(self, root, mode, transforms = None, save_dir = None):
        self.root = root
        self.mode = mode
        self.save_dir = save_dir
        self.transforms = transforms

        self.num_classes = 157

        self.loc = osp.join('./preprocessed/lists', 'tvqa-{}'.format(self.mode))
        if not osp.isdir(self.loc):
            os.makedirs(self.loc)

        self.make_dataset()

    def make_dataset(self):
        video_file = osp.join(self.loc, 'video_list.npy')
        video_len_file = osp.join(self.loc, 'video_lengths.npy')
        video_image_file = osp.join(self.loc, 'image_list.npy')

        if osp.exists(video_file):
            self.video_list = np.load(video_file, allow_pickle = True)
            self.video_lengths = np.load(video_len_file, allow_pickle = True)
            self.image_list = np.load(video_image_file, allow_pickle = True)
            return

        self.video_list = []
        self.video_lengths = []
        self.image_list = []
        tvshows = list_directories(self.root, full_path = True)
        for show in tqdm(tvshows):
            clips = list_directories(show, full_path = True)
            for clip in clips:
                self.video_list.append(clip)
                if self.mode == 'rgb':
                    image_list = list_images(osp.join(self.root, show, clip), full_path = True)
                else:
                    image_list = list_images(osp.join(self.root, show, clip), full_path = True, ext = ['.flo'])

                image_list = sorted(image_list)
                self.image_list.append(image_list)
                self.video_lengths.append(len(image_list))

        np.save(video_file, self.video_list)
        np.save(video_len_file, self.video_lengths)
        np.save(video_image_file, self.image_list)

    def _integrate_lengths(self):
        # Save video_lengths as an integral array for faster lookups
        total = 0
        for idx in range(len(self.video_lengths)):
            total += self.video_lengths[idx]
            self.video_lengths[idx] = total

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_path = self.video_list[index]
        num_frames = len(self.image_list[index])

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, self.image_list[index], 0, num_frames)
        else:
            imgs = load_flow_frames(self.root, self.image_list[index], 0, num_frames)

        label = np.zeros((self.num_classes, num_frames), np.float32)

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label), video_path
