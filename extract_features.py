import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

from tqdm import tqdm
import numpy as np

from pytorch_i3d import InceptionI3d

#from charades_dataset_full import Charades as Dataset
from tvqa_dataset_full import TVQA as Dataset


def run(max_steps=64e3, mode='rgb', root='/ssd2/charades/Charades_v1_rgb', batch_size=1, load_model='', save_dir=''):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(root, mode, test_transforms, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    #val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
    #val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    dataloaders = {'train': dataloader}
    datasets = {'train': dataset}


    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for phase in ['train']:
        i3d.train(False)  # Set model to evaluate mode

        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0

        # Iterate over data.
        for data in tqdm(dataloaders[phase]):
            # get the inputs
            inputs, labels, name = data
            name = name[0].split('frames_hq/')[1:]

            parent_dir = os.path.join(save_dir, os.path.dirname(name[0]))
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)

            #if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
            #    continue

            b,c,t,h,w = inputs.shape
            if t > 128:
                features = []
                for start in range(0, t, 128):
                    end = min(t, start + 128)
                    if (end - start < 8):
                        start = start - 8
                    with torch.no_grad():
                        ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda())
                    features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
            else:
                # wrap them in Variable
                inputs = Variable(inputs.cuda(), volatile=True)
                features = i3d.extract_features(inputs)
                np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
