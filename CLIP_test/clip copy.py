import os
import cv2
import gc
import time
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

dataset = "8k"

if dataset == "8k":
    df = pd.read_csv("E:\AIA_Team_Project\content\captions.txt")
    # print(df)
    df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
    # print(df['id'])
    df.to_csv("E:\AIA_Team_Project\content\captions.csv", index=False)
    df = pd.read_csv("E:\AIA_Team_Project\content\captions.csv")
    image_path = "E:\AIA_Team_Project\content\Images"
    captions_path = "E:\AIA_Team_Project\content/"
elif dataset == "30k":
    df = pd.read_csv("/content/flickr30k_images/results.csv", delimiter="|")
    df.columns = ['image', 'caption_number', 'caption']
    df['caption'] = df['caption'].str.lstrip()
    df['caption_number'] = df['caption_number'].str.lstrip()
    df.loc[19999, 'caption_number'] = "4"
    df.loc[19999, 'caption'] = "A dog runs across the grass ."
    ids = [id_ for id_ in range(len(df) // 5) for _ in range(5)]
    df['id'] = ids
    df.to_csv("captions.csv", index=False)
    image_path = "/content/flickr30k_images/flickr30k_images"
    captions_path = "/content"

# print(df.head())

class CFG: # 사용될 구성에대한 정의
    debug = False
    image_path = image_path
    captions_path = captions_path
    batch_size = 32
    num_workers = 0
    head_lr = 1e-3
    image_encoder_lr = 1e-4 # 0.0001
    text_encoder_lr = 1e-5 # 0.00001
    weight_decay = 1e-3 # 0.001
    patience = 1
    factor = 0.8
    epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased" # uncased: 대소문자 구별x, MLM
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # 이미지 인코더와 텍스트 인코더
    trainable = True # 이미지 인코더와 텍스트 인코더
    temperature = 1.0 

    # image size
    size = 224

    # 이미지 및 텍스트 인코더에 모두 사용
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

'''
Image, Text 데이터 전처리
'''
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        
        Image 와 해당하는 Captions를 pair로 매칭
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length # 200
        )
        self.transforms = transforms # get_transforms (현재 mode = train이므로 train 만 탐)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        # print("self.image_filenames[idx]:: ",self.image_filenames[idx])
        # print("self.image_filenames[idx]:: ",self.image_filenames[idx])
        # image_filenames[idx]::  3541962817_78bcd3835b.jpg
        # print("item :: ", item)
        '''
        item ::  {
        'input_ids': tensor([ 101, 1996, 3586, 1998, 7945, 5376, 1037, 8638, 1012,  102,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0]), 
        'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
        '''
        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        # print("CLIPDataset item['image'] : ", item['image'][0])
        '''
        CLIPDataset item['image'] :  tensor([[-1.9809, -1.9124, -2.1179,  ..., -2.1179, -2.1179, -2.0837],
        [-2.0665, -2.0494, -2.1179,  ..., -2.1179, -2.1179, -2.1179],
        [-2.1179, -2.1179, -2.1179,  ..., -2.1179, -2.1179, -2.1179],
        ...,
        [-0.9192, -0.8335, -0.7137,  ...,  0.8276,  0.9132,  0.8618],
        [-1.4158, -1.3815, -1.5014,  ...,  0.9817,  1.0502,  1.0502],
        [-1.5699, -1.6042, -1.5699,  ...,  1.0331,  0.8104,  0.8789]])
        '''
        item['caption'] = self.captions[idx]
        # print("CLIPDataset item['caption'] : ", item['caption'][0])
        '''
        CLIPDataset item['caption'] :  A    
        '''

        return item


    def __len__(self):
        return len(self.captions)

def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose( # transform에 여러단계가 있는경우, Compose로 여러단계를 하나로 묵을수있음 transforms에 속한 함수들을 묶어서 한번에 처리
            [   # size : image size = 224 
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [   # size : image size = 224 
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
        
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    Resnet model 사용
    """

    def __init__(            # resnet50,            True,                     True
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__() # timm을 사용하여 모델 정의 
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
        
class TextEncoder(nn.Module):  # "distilbert-base-uncased",          True,                     True
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

# 이미지임베딩과 텍스트임베딩 차원 맞춰주기?
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim, # Image : 2048, Text : 768
        projection_dim=CFG.projection_dim, # 256 <- 출력 vector의 크기
        dropout=CFG.dropout # 0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature, # 1.0
        image_embedding=CFG.image_embedding, # 2048
        text_embedding=CFG.text_embedding, # 768
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        # print("CLIPModel forward image features :: ", image_features)
        '''
        CLIPModel forward image features ::  tensor([[0.0000, 0.0000, 0.0645,  ..., 0.0377, 0.2321, 0.0000],
        '''
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        print("CLIPModel forward text features :: ", text_features)
        '''
        CLIPModel forward text features ::  tensor([[ 0.1253, -0.2729, -0.2552,  ..., -0.3578,  0.7134,  0.0008],
        '''
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        # print("CLIPModel forward image embed :: ", image_embeddings)
        '''
        CLIPModel forward image embed ::  tensor([[ 0.8637, -2.4903, -2.9246,  ..., -1.2986,  0.5674, -3.0836],
        '''
        # print("CLIPModel forward text embed :: ", text_embeddings)
        '''
        CLIPModel forward text embed ::  tensor([[ 0.5126, -0.6403, -0.5461,  ...,  0.2908, -0.9984, -0.7614],
        '''
        # Calculating the Loss
        # @ 연산자 :: 간단하게 행렬곱을 표기
        logits = (text_embeddings @ image_embeddings.T) / self.temperature  # 템퍼쳐 값이 커질 수록 logits 값이 줄어듦. 기준이 빡세짐. 얘는 로그소프트맥스 취하고
        # print("CLIPModel logits :: ", logits)
        '''
        CLIPModel logits ::  tensor([[127.3474, 115.2535, 126.5811,  ..., 128.5265, 127.2788, 129.6808],
        '''
        images_similarity = image_embeddings @ image_embeddings.T           # 어텐션 에너지값은 그냥 소프트맥스 취해서 로스를 구함
        texts_similarity = text_embeddings @ text_embeddings.T
        # print("CLIPModel forward images similarity :: ", images_similarity)
        '''
        CLIPModel forward images similarity ::  tensor([[245.9526, 238.4569, 236.0058,  ..., 235.7797, 237.0153, 238.0736],
        '''
        # print("CLIPModel forward texts similarity :: ", texts_similarity)
        '''
        CLIPModel forward texts similarity ::  tensor([[244.5822, 239.2083, 239.5731,  ..., 234.4465, 239.8616, 237.8008],
        '''
        # 텍스트 피쳐와 이미지 피쳐를 행렬곱해서 트포처럼 어텐션 에너지값을 구함
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)   # targets.shape = (32,32)   preds.shape = (32,32)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
# A simple Example

# batch_size = 4
# dim = 256
# embeddings = torch.randn(batch_size, dim)
# out = embeddings @ embeddings.T
# print(F.softmax(out, dim=-1))

def make_train_valid_dfs(): # train valid data 정의(split)
    dataframe = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    # print("make train valid dfs max_id :: ", max_id) 
    '''
    make train valid dfs max_id ::  8090 # +1 뺀거. 1은 패딩토큰인것? 
    make train valid dfs max_id ::  8091 
    '''
    image_ids = np.arange(0, max_id)
    # print("make train valid dfs image_ids :: ", image_ids)
    '''
    make train valid dfs image_ids ::  [   0    1    2 ... 8088 8089 8090]
    '''
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    # print("make train valid dfs valid_ids :: ", valid_ids)
    '''
    make train valid dfs valid_ids ::  [4194 4166 1928 ... 4506 4450 2199]
    '''
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    # print("make train valid dfs train_ids :: ", train_ids)
    '''
    make train valid dfs train_ids ::  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18,... 2135, 2136, 2137, 2139, 2140, 2141, 2143, 2144,...]
    '''
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    # print('dataF:', train_dataframe.head())
    '''
    dataF:      image                               caption                          id
    0  1000268201_693b08cb0e.jpg  A child in a pink dress is climbing up a set o...   0
    1  1000268201_693b08cb0e.jpg              A girl going into a wooden building .   0
    2  1000268201_693b08cb0e.jpg   A little girl climbing into a wooden playhouse .   0
    3  1000268201_693b08cb0e.jpg  A little girl climbing the stairs to her playh...   0
    4  1000268201_693b08cb0e.jpg  A little girl in a pink dress going into a woo...   0
    '''
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe

# 

def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object: # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        # print("train_epoch :: ",batch)
        '''
        {'input_ids': tensor([[ 101, 1037, 2158,  ...,    0,    0,    0],
        [ 101, 1037, 2829,  ...,    0,    0,    0],
        [ 101, 1037, 2829,  ...,    0,    0,    0],
        ...,
        [ 101, 2048, 3057,  ...,    0,    0,    0],
        [ 101, 2093, 3455,  ...,    0,    0,    0],
        [ 101, 2195, 2111,  ...,    0,    0,    0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]), 'image': tensor([[[[-1.7069, -1.7412, -1.9467,  ...,  0.3823,  0.5364,  0.6906],
          [-1.6042, -1.7240, -1.8953,  ...,  0.2282,  0.5193,  0.6049],
          [-1.1589, -1.6898, -1.9124,  ...,  0.2796,  0.3309,  0.3994],
          ...,
          [-1.6727, -1.6555, -1.6384,  ..., -0.5082, -1.0562, -1.1760],
          [-1.6384, -1.6727, -1.6898,  ..., -1.4672, -1.2959, -1.2445],
          [-1.7240, -1.7583, -1.7412,  ..., -1.5014, -1.4843, -1.3987]],
        '''
        loss = model(batch)
        optimizer.zero_grad() # 미분값 초기화
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        # print(count)
        loss_meter.update(loss.item(), count)
        # print(loss_meter.update(loss.item(), count))


        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer)) 
        # tqdm + set_postfix logging 하는것까지 구현
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
        # set_postfix 로깅되지 않는 부분에 대해서 추가적으로 중간중간 결과를 볼수있음

    return loss_meter


def main():
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer) # "distilbert-base-uncased"
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    # print("main train loader :: ", train_loader)
    # print("main valid loader :: ", valid_loader)


    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr}, # 1e-4
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr}, # 1e-5 
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]               # 1e-3                     1e-3
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )                                       # 1                 # 0.8
    '''
    lr_scheduler : 학습률 스케줄러 -> 학습이 진행되면서 학습률을 그 상황에 맞게 가변적으로 적당하게
    변경될 수 있다면 더 낮은 손실값을 얻을 수 있음. 이를 위해 학습률을 스케줄이 필요함.
    ReduceLROnPlateau : 원하는 epoch 마다, 이전 학습률 대비 변경폭에 따라 학습률을 감소시켜주는 방식
    weight_decay : loss에 어떤 제약조건을 적용해 오버피팅을 최소화하는방법
    L1과 L2 이 있음. 특정값을 손실함수에 더해주어 가중치를 감소시킴, 
    ReduceLROnPlateau :: keras callback method 
    
    '''

    step = "epoch"
    start_time = time.time()
    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}") #  5
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)
    end_time = time.time() - start_time
    return end_time
        
end_time = main()
print('took', round(end_time), 'sec.')
print(f'epochs: {CFG.epochs}    batch size: {CFG.batch_size}') # 32

def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)

_, valid_df = make_train_valid_dfs()
model, image_embeddings = get_image_embeddings(valid_df, "best.pt")

def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5) # argmax 인데 제일 큰거부터 (n=)9*5개 인덱스 반환함
    print('indices', indices)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{CFG.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    
    plt.show()


find_matches(model, 
             image_embeddings,
            #  query="cats on the grass",
             query="two dogs are playing in the snow",
            #  query="people running on the grass",
            #  query="a girl jumping from swing",
             image_filenames=valid_df['image'].values,     
             n=9)

# took 2246 sec.
# epochs: 5    batch size: 32

# 구조를 쉽게 요약하면 어텐션 기법을 사용하여 이미지 피처와 텍스트 피처의 유사도를 계속 구하는 방식으로 훈련하고
# 예측의 경우 텍스트 피처를 넣으면 이미지 피처를 클래시파이어 클래스로 두고
# 그 중에서 topk - 5 의 방식으로 5장을 해당 텍스트에 관련된 이미지라고 예측