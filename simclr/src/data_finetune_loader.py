import numpy as np
import cv2
from PIL import Image
import re
import platform
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

train_transform =[
    ### IMPLEMENTATION 2-1 ###
    ### 1. Random resized crop w/ final size of (32, 32)
    ### 2. Random horizontal flip w/ p=0.5
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=0.5)
]

test_transform =[
    transforms.Resize(224)
]

list_transform =[
    transforms.Resize(224)
]


class OrientationDataset(Dataset):
    def __init__(self, project_dir, train_flag=1, noise_flag=1):
        self.project_dir = project_dir
        self.train_flag = train_flag
        self.noise_flag = noise_flag
        self.toTensor = transforms.ToTensor()

        train_path = self.project_dir + '/data_finetune/index/train.txt' 
        test_path = self.project_dir + '/data_finetune/index/test.txt'
        list_path = self.project_dir + '/data_finetune/index/file_names.txt'
        self.data_path = self.project_dir+'/data_finetune/blend_hairs_imgs/'
        # generate dataset
        if self.train_flag == 1:
            self.train_index = []
            self.train_index_path = train_path
            with open(self.train_index_path, 'r') as f:
                lines = f.readlines()
                for x in lines:
                    self.train_index.append(x.strip().split(' '))
        if self.train_flag == 0:
            self.test_index = []
            self.test_index_path = test_path
            with open(self.test_index_path, 'r') as f:
                lines = f.readlines()
                for x in lines:
                    self.test_index.append(x.strip().split(' '))
        if self.train_flag == -1:
            self.list_index = []
            self.list_index_path = list_path
            with open(self.list_index_path, 'r') as f:
                lines = f.readlines()
                for x in lines:
                    self.list_index.append(x.strip().split(' '))

    def __getitem__(self, index):
        if self.train_flag == 1:
            current_index = self.train_index[index][0]
        elif self.train_flag == 0:
            current_index = self.test_index[index][0]
        else:
            current_index = self.list_index[index][0]

        tmp = str(current_index).split('_')
        # print(tmp)
        if tmp[1] == 'v0.png':
            next_index = tmp[0] + '_' + 'v1.png'
        elif tmp[1] == 'v2.png':
            next_index = tmp[0] + '_' + 'v3.png'
        elif tmp[1] == 'v4.png':
            next_index = tmp[0] + '_' + 'v5.png'
        elif tmp[1] == 'v6.png':
            next_index = tmp[0] + '_' + 'v7.png'
        else:
            next_index = tmp[0] + '_' + 'v9.png'
        img1 = cv2.imread(self.data_path+current_index)
        img1 = Image.fromarray(img1)
        if self.train_flag != -1:
            img2 = cv2.imread(self.data_path+next_index)
            img2 = Image.fromarray(img2)

        if self.train_flag == 1:
            for t in train_transform :
                img1 = t(img1)
                img2 = t(img2)
        
        elif self.train_flag == 0 :
            for t in test_transform :
                img1 = t(img1)
                img2 = t(img2)

        else :
            for t in list_transform :
                img1 = t(img1)

        img1 = self.toTensor(img1)
        
        if self.train_flag != -1:
            img2 = self.toTensor(img2)
            return img1, img2
        return img1

    def __len__(self):
        if self.train_flag == 1:
            return len(self.train_index)
        elif self.train_flag == 0:
            return len(self.test_index)    
        else:
            return len(self.list_index)    
if __name__ == "__main__":
    BATCH_SIZE = 10
    EPOCH = 100
    train_data = OrientationDataset(project_dir=".",train_flag=-1)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
    for i in range(EPOCH):
        for j, data in enumerate(train_loader, 0):
            # img1, img2 = data
            img1 = data
            print(img1.shape)
            quit()