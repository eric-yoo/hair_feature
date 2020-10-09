import torch
import torch.nn.functional as f
import torchvision.datasets as datasets
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import time

import torchvision.transforms as transforms
from simclr.src.data_loader import OrientationDataset



"""
    < Inference 실제 작동 방법 >
    1. 343 개 헤어 모델 데이터 (343, 3, 128, 128)
    2. up-sample (343, 3, 224, 224) & encoding (343, feature_dim)

    3. 테스트 데이터 (3, 128, 128)
    4. up-sample (3, 224, 224) & encoding (feature_dim)

    5. Cosine similarity (343, 1)
    6. argmax
    
    7. check accuracy

    < Test code 작동 방법>

    1. batch size n
    2. 타겟 데이터 로드 & up-sample (n, 3, 224, 224)  & encoding (n, feature_dim)
    3. 테스트 데이터 로드 (타겟의 각각 페어로 구성) & upsample (n, 3, 224, 224)  & encoding (n, feature_dim)
    4. 2번과 3번 cosine simililarity (n , n)
    5. 대각선에 있는 애들이 argmax 돼야 정답.
"""

def test(model, loader, BATCH_SIZE=20) :

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad() :
        for idx, data in enumerate(loader) :
            if idx % 100 == 0: print(idx) # 14060 = 20*703
    
            zi, zj = [_data.cuda() for _data in data] # (n, 3, 224, 224)
            n = zi.size(0)

            feat_i = model(zi) # (n, 1000)
            feat_j = model(zj) # (n, 1000)

            feat_i = f.normalize(feat_i, dim=1)
            feat_j = f.normalize(feat_j, dim=1)
            
            score = torch.matmul(feat_i, feat_j.T) # (n, n)
            predicted = torch.argmax(score, dim=1) # (n, 1)
            answer = torch.arange(n).cuda()

            total += n
            correct += (predicted == answer).sum().item()

            print("Accuracy : ", correct / total)
    
    print("Total Accuracy : ", correct / total)


"""
    inference function not tested
    model : resnet model
    target feature : "Normalized" Tensor (343, 1000) (미리 계산해놔야 함 !)
    data_path : inference data (image path) (3, 128, 128)

    1. Load image to tensor and upsample (3, 224, 224)
    2. featuree (1000)
    3. argmax 
"""

test_transform =[
    transforms.Resize(224),
    transforms.ToTensor()
]

def inference(model, target, data_path) :

    
    img = Image.open(data_path)
    for t in test_transform :
        img = t(img)

    model.eval()
    with torch.no_grad() :
        
        feat = model(img.cuda()) # (1000)
        feat = f.normalize(feat, dim=1) # dim = 0 ?
 
        score = torch.matmul(feat, target.T) # (1024)
        predicted = torch.argmax(score, dim=1).item
        print("Predicted ", predicted, ".data")

"""
    Not tested !
    
    target_data : (343, 3, 224, 224) Tensor

    output : (343, 1000) Normalized Tensor
""" 
def make_target(model, target_data) :

    model.eval()
    with torch.no_grad() :        
        feat = model(target_data.cuda()) # (343, 1000)
        feat = f.normalize(feat, dim=1) # (343, 1000)

    reutnr target_data

if __name__ == "__main__":
    test_data = OrientationDataset(project_dir="./simclr",train_flag=0)
    BATCH_SIZE =  20
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

    model = torchvision.models.resnet50()
    GPU_NUM = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NUM
    model.cuda()

    ckpt = "epoch_150.pt"
    state_dict = torch.load(ckpt)
    model.load_state_dict(state_dict)

    test(model, test_loader, 20)
