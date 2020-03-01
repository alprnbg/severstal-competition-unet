# pylint: disable=W,C,R

import argparse

import numpy as np
import cv2
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch

from model import *


def test_transform(img):
    to_pil = transforms.ToPILImage()
    img = to_pil(img)
    resize = transforms.Resize(64)
    img = resize(img)
    img = TF.to_tensor(img)
    normalize = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    img = normalize(img)
    return img.view((-1,3,64,400))

def test(args):
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

    net = UNet(3, 4).to(device).float()
    net.eval()

    if args.model_path:
        net.load_state_dict(torch.load(args.model_path))

    img = cv2.imread(args.image)
    img = test_transform(img).to(device)

    with torch.set_grad_enabled(False):
        prediction = np.round(np.array(net(img).cpu()))
        cv2.imwrite("label0"+ args.image.split("/")[-1], prediction[0,0,:,:]*255)
        cv2.imwrite("label1"+ args.image.split("/")[-1], prediction[0,1,:,:]*255)
        cv2.imwrite("label2"+ args.image.split("/")[-1], prediction[0,2,:,:]*255)
        cv2.imwrite("label3"+ args.image.split("/")[-1], prediction[0,3,:,:]*255)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test the U-Net model')

    parser.add_argument('--image',
                           type=str,
                           required=True,
                           help='the path to image')
    
    parser.add_argument('--model_path',
                           type=str,
                           required=True,
                           help='the path to model')

    args = parser.parse_args()
    test(args)