# pylint: disable=W,C,R

from torch.utils.data import Dataset
from utils import *
import cv2
import pandas as pd

class SteelData(Dataset):
    def __init__(self, paths, transform):
        self.transform_check = transform
        self.train_df = pd.read_csv(paths[0])
        self.img_paths = paths[1]


    def __len__(self):
        return int(len(self.train_df)/4)


    def __getitem__(self, idx):   
        im_name = self.train_df.iloc[idx*4, 0].split('_')[0]
        path = self.img_paths + im_name
        
        img = cv2.imread(path)
        mask = self.get_mask(idx*4)*255
        
        if self.transform_check:
            return transform(img, mask)
        else:
            return img.transpose(2,0,1), mask.transpose(2,0,1)
        

    def get_mask(self, start_idx):
        img_names = [str(i).split("_")[0] for i in self.train_df.iloc[start_idx:start_idx+4, 0].values]
        
        if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):
            raise ValueError

        labels = self.train_df.iloc[start_idx:start_idx+4, 1]
        mask = np.zeros((256, 1600, 4), dtype=np.uint8)

        for idx, label in enumerate(labels.values):
            if label is not np.nan:
                mask_label = np.zeros(1600*256, dtype=np.uint8)
                label = label.split(" ")
                positions = map(int, label[0::2])
                length = map(int, label[1::2])
                for pos, le in zip(positions, length):
                    mask_label[pos-1:pos+le-1] = 1
                mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')
        return mask