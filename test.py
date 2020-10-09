



"""
    < Inference 실제 작동 방법 >
    1. 343 개 헤어 모델 데이터 (343, 3, 128, 128)
    2. up-sample (343, 3, 224, 224) & unsquueze (343, 3 x 224 x 224) & encoding (343, feature_dim)

    3. 테스트 데이터 (3, 128, 128)
    4. up-sample (3, 224, 224) & unsqueeze (3 x 224 x 224) & encoding (feature_dim)

    5. Cosine similarity (343, 1)
    6. argmax
    
    7. check accuracy

    < Test code 작동 방법>

    1. batch size n
    2. 타겟 데이터 로드 & up-sample (n, 3, 224, 224) & unsqueeze & encoding (n, feature_dim)
    3. 테스트 데이터 로드 (타겟의 각각 페어로 구성) & upsample (n, 3, 224, 224) & unsqueeze & encoding (n, feature_dim)
    4. 2번과 3번 cosine simililarity (n , n)
    5. 대각선에 있는 애들이 argmax 돼야 정답.
"""

def test(model, loader, ckpt) :
    state_dict = torch.load(ckpt)
    model.load_state_dict(state_dict)

    for idx, data in enumerate(loader) :
        if idx % 100 == 0: print(idx) # 14060 = 20*703
        optimizer.zero_grad()
        zi, zj = [_data.cuda() for _data in data]
        feat_i = model(zi)
        feat_j = model(zj)
        loss = loss_fn(feat_i, feat_j)

        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()    