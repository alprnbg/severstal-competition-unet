# pylint: disable=W,C,R

#TODO Make the loss function weighted for the imbalanced class or balance the data.
# logsoftmax + NLLLoss
# [0000] -> class 0 
# [0001] -> class 1 
# [0010] -> class 2 
# [0100] -> class 3 
# [1000] -> class 4 
#TODO tensorboard


from utils import *
from dataset import *
from model import *

import argparse
import sys
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
#from torch_lr_finder import LRFinder
import numpy as np
import cv2
from matplotlib import pyplot as plt 


def train(args):

    if args.env == "kaggle":
        paths = [args.source, "/kaggle/input/severstal-steel-defect-detection/train_images/"]
    elif args.env == "colab":
        paths = [args.source, "/content/steel-data/train_images/"]
    else:
        raise Exception("Wrong environment.")

    dataset = SteelData(paths=paths, transform=True)
    
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

    net = UNet(3, args.n_classes).to(device).float()

    if args.model_path:
        net.load_state_dict(torch.load(args.model_path))

    optimizer = torch.optim.Adam(net.parameters())
    #optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    #optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-2)
    
    #criterion = torch.nn.BCELoss()    
    criterion = DiceLoss()    

    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0002, max_lr=args.lr, cycle_momentum=False)

    # Creating data indices for training and validation splits
    random_seed = 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if args.shuffle :
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    val_amount = args.val_size

    train_indices, val_indices = indices[val_amount:], indices[:val_amount]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=args.batch, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=args.batch, sampler=valid_sampler)
    loaders = {"train": train_loader, "val": val_loader}

    #lr_finder = LRFinder(net, optimizer, criterion, device=device)
    #lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
    #print(lr_finder.history)
    #sys.exit(0)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Saving step: {}
        Device: {}
    '''.format(args.epoch, args.batch, args.lr, len(train_indices),
               len(val_indices), args.save_step,str(device)))

    for epoch in range(1, args.epoch+1):
        
        print("Epoch {} started...".format(epoch))

        
        #for phase in ["train", "val"]:
        for phase in ["train"]:
            
            running_loss = 0.0
            
            if phase == "train":
                net.train()
            else:
                print("Validation in process...")
                net.eval()
            
            for i, batch in tqdm(enumerate(loaders[phase])):
                x, y_true = batch
                x, y_true = x.to(device).float(), y_true.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=="train"):
                    y_pred = net(x)                    
                    loss = criterion(y_pred, y_true)
                    running_loss += loss.item()

                    if phase == "train":
                        loss.backward()
                        optimizer.step()                        

            if phase == "train":
                train_loss = running_loss/(args.batch*len(train_loader))
                print("Epoch {} -->  Train Loss: {}".format(epoch, train_loss))
            else:
                val_loss = running_loss/(args.batch*len(val_loader))
                print("Epoch {} -->  Validation Loss: {}".format(epoch, val_loss))

        if (epoch) % args.save_step == 0:
            torch.save(net.state_dict(), "Checkpoints/" + 'CP{}.pth'.format(epoch))
            print('Checkpoint {} saved !'.format(epoch))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of a U-Net model')

    parser.add_argument('--source',
                           type=str,
                           required=True,
                           help='the path to train.csv file')

    parser.add_argument('--n_classes',
                           type=int,
                           default=4,
                           help='number of classes in dataset')

    parser.add_argument('--epoch',
                           type=int,
                           default=1,
                           help='the number of epochs')

    parser.add_argument('--batch',
                           type=int,
                           default=4,
                           help='the size of batches')
    
    parser.add_argument('--val_size',
                           type=int,
                           required=True,
                           help='the size of validation set')

    parser.add_argument('--lr',
                           type=float,
                           default=0,
                           help='the value of learning rate')
    
    parser.add_argument('--shuffle',
                           type=bool,
                           default=True,
                           help='shuffle the dataset')
    
    parser.add_argument('--env',
                           type=str,
                           required=True,
                           help='kaggle or colab')

    parser.add_argument('--save_step',
                           type=int,
                           default=1,
                           help='frequency of saving the model')
    
    parser.add_argument('--model_path',
                           type=str,
                           default=None,
                           help='path of the model for transfer learning')

    args = parser.parse_args()
    train(args)
