import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

import torchvision
import numpy as np

import os

import time

import torchvision.transforms as transforms

from simclr.src.data_loader import OrientationDataset


class NTXentLoss(torch.nn.Module):
    
    def __init__(self, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.cuda()

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(2 * self.batch_size).cuda().long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


def train(model, loader, batch_size=256, weight_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
  
    n_epoch = 200
    temperature = 0.07
    loss_fn = NTXentLoss(batch_size=batch_size, temperature=temperature, use_cosine_similarity=True)

    resume_epoch = 0
    if weight_path != None:
        model.load_state_dict(torch.load(weight_path))
        resume_epoch = int(weight_path.split("epoch_")[1].split(".pt")[0])
        print("resume training from epoch {}".format(resume_epoch))

    train_start = time.time()
    for epoch in range(resume_epoch+1, n_epoch + 1) :
        model.train()
        train_loss = 0

        epoch_start = time.time()
        for idx, data in enumerate(loader) :
            optimizer.zero_grad()
            zi, zj = [_data.cuda() for _data in data]
            feat_i = model(zi)
            feat_j = model(zj)
            loss = loss_fn(feat_i, feat_j)

            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start
        train_loss /= (idx + 1)
        print("Epoch\t", epoch, 
                "\tLoss\t", train_loss, 
                "\tTime\t", epoch_time,
                )
        path = "simclr/ckpt/epoch_{}.pt".format(epoch) 
        torch.save(model.state_dict(), path)
    
    elapsed_train_time = time.time() - train_start
    print('Finished training. Train time was:', elapsed_train_time)

if __name__ == "__main__":
    test_data = OrientationDataset(project_dir="./simclr",train_flag=0)
    BATCH_SIZE =  20
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)

    model = torchvision.models.resnet50()
    GPU_NUM = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM
    model.cuda()

    train(model, train_loader, batch_size = BATCH_SIZE, weight_path = "./simclr/ckpt/epoch_150.pt")


