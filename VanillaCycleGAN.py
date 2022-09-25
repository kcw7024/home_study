
# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import os
from tqdm.auto import tqdm
from glob import glob
import cv2
import numpy as np
import pandas as pd
import PIL 
import urllib
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from random import uniform
from imgaug import augmenters as iaa

# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

import torch.utils.data as td
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
import time

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
device

# from google.colab import drive
# drive.mount._DEBUG = True
# drive.mount('/content/drive', force_remount=True)

# os.chdir('./drive/MyDrive/기계학습특강/')

# Commented out IPython magic to ensure Python compatibility.
# %%time
# !unzip "./summer2winter_yosemite.zip" -d "/content/drive/MyDrive/기계학습특강/summer2winter_yosemite/"

"""# **데이터 전처리**"""

import glob
train_X_path=sorted(glob.glob("./summer2winter_yosemite/summer2winter_yosemite/trainA/*",recursive=True))
train_Y_path=sorted(glob.glob("./summer2winter_yosemite/summer2winter_yosemite/trainB/*",recursive=True))
test_X_path=sorted(glob.glob("./summer2winter_yosemite/summer2winter_yosemite/testA/*",recursive=True))
test_Y_path=sorted(glob.glob("./summer2winter_yosemite/summer2winter_yosemite/testB/*",recursive=True))

len(train_X_path),len(train_Y_path)

len(test_X_path),len(test_Y_path)

train_X_files = np.array(train_X_path)
train_Y_files = np.array(train_Y_path)

test_X_files = np.array(test_X_path)
test_Y_files = np.array(test_Y_path)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x_dir, y_dir,transform=None):
        super().__init__()
        self.transform = transform
        self.x_img = x_dir
        self.y_img = y_dir   

    def __len__(self):
        return len(self.x_img)

    def __getitem__(self, idx):
        x_img = self.x_img[idx]
        y_img = self.y_img[idx%len(self.y_img)]

        x_img = cv2.imread(x_img)
        y_img = cv2.imread(y_img)

        x_img= cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        y_img= cv2.cvtColor(y_img, cv2.COLOR_BGR2RGB)

        if self.transform!=None:
            augmented = self.transform(image=x_img,image2=y_img)
            x_img = augmented['image']
            y_img = augmented['image2'] 

            y_img = np.transpose(y_img,(2,0,1))
            x_img = np.transpose(x_img,(2,0,1))
            # y_img = y_img.astype(np.float32)/255
        
        return x_img,y_img

import albumentations
import albumentations.pytorch
aug = albumentations.Compose([
                              albumentations.Resize(280, 280), 
                              albumentations.RandomCrop(256, 256),
                              albumentations.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                              ],additional_targets={'image2':'image'})

train_dataset = MyDataset(train_X_files,train_Y_files,transform=aug)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=3,shuffle=True)

##input과 label이 잘 나오나 확인

X,Y = next(iter(train_loader))
X=0.5*(X+1)
Y=0.5*(Y+1)
print(X.shape)
print(Y.shape)
plt.figure(figsize=(16,18))
plt.subplot(1,2,1)
plt.imshow(X[0].T)
plt.subplot(1,2,2)
plt.imshow(Y[0].T)
plt.show()

"""# **모델 생성**"""

def conv_block(in_dim,out_dim,act_fn):
    model=nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn
    )
    return model

def conv_trans_block(in_dim,out_dim,act_fn):
    model=nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim,kernel_size=3,stride=2,padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn
    )
    return model    

def maxpool():
    pool=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    return pool

def conv_block_2(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

class UnetGenerator(nn.Module):
    def __init__(self,in_dim,out_dim,num_filter):
        super(UnetGenerator,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        self.down_1 = conv_block_2(self.in_dim,self.num_filter,act_fn)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(self.num_filter*1,self.num_filter*2,act_fn)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2(self.num_filter*2,self.num_filter*4,act_fn)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2(self.num_filter*4,self.num_filter*8,act_fn)
        self.pool_4 = maxpool()
        self.down_5 = conv_block_2(self.num_filter*8,self.num_filter*16,act_fn)
        self.pool_5 = maxpool()

        self.bridge = conv_block_2(self.num_filter*16,self.num_filter*32,act_fn)

        self.trans_1 = conv_trans_block(self.num_filter*32,self.num_filter*16,act_fn)
        self.up_1 = conv_block_2(self.num_filter*32,self.num_filter*16,act_fn)
        self.trans_2 = conv_trans_block(self.num_filter*16,self.num_filter*8,act_fn)
        self.up_2 = conv_block_2(self.num_filter*16,self.num_filter*8,act_fn)
        self.trans_3 = conv_trans_block(self.num_filter*8,self.num_filter*4,act_fn)
        self.up_3 = conv_block_2(self.num_filter*8,self.num_filter*4,act_fn)
        self.trans_4 = conv_trans_block(self.num_filter*4,self.num_filter*2,act_fn)
        self.up_4 = conv_block_2(self.num_filter*4,self.num_filter*2,act_fn)
        self.trans_5 = conv_trans_block(self.num_filter*2,self.num_filter*1,act_fn)
        self.up_5 = conv_block_2(self.num_filter*2,self.num_filter*1,act_fn)

        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter,self.out_dim,3,1,1),
            nn.Tanh(),  #필수는 아님
        )

    def forward(self,input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)
        down_5 = self.down_5(pool_4)
        pool_5 = self.pool_5(down_5)

        bridge = self.bridge(pool_5) #torch.Size([2, 1024, 8, 8])

        trans_1 = self.trans_1(bridge)#torch.Size([2, 512, 16, 16])
        concat_1 = torch.cat([trans_1,down_5],dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2,down_4],dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3,down_3],dim=1)
        up_3 = self.up_3(concat_3)
        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4,down_2],dim=1)
        up_4 = self.up_4(concat_4)
        trans_5 = self.trans_5(up_4)
        concat_5 = torch.cat([trans_5,down_1],dim=1)
        up_5 = self.up_5(concat_5)

        out = self.out(up_5)
        return out

def make_disc_block(input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
    if not final_layer:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride)
        )
def fullyconnected(in_channel,out_channel):
    fc = nn.Linear(in_channel,out_channel)   
    return fc

class Discriminator(nn.Module):

    def __init__(self, im_chan=3, hidden_dim=8):
        super(Discriminator, self).__init__()
        
        # Discriminator모델 구성하기
        self.disc = nn.Sequential(
            make_disc_block(im_chan, hidden_dim, kernel_size=4),
            make_disc_block(hidden_dim, hidden_dim * 2),
            make_disc_block(hidden_dim*2, hidden_dim * 4),
            make_disc_block(hidden_dim*4, hidden_dim * 4),
            make_disc_block(hidden_dim * 4, hidden_dim*2),
            make_disc_block(hidden_dim * 2, 1,final_layer=True)
        )
        
        self.fc = fullyconnected(3*3, 1)
        self.act = nn.Sigmoid()

    def forward(self, image):
        disc_pred = self.disc(image)
        disc_pred = self.fc(disc_pred.view(len(disc_pred), -1))  # discriminator의 판별 결과 (0:fake, 1:real)
        disc_pred = self.act(disc_pred)
        return disc_pred.view(len(disc_pred), -1)

img_size = 256
in_dim = 3
out_dim = 3
num_filters = 32

generatorG = UnetGenerator(in_dim=in_dim,out_dim=out_dim,num_filter=num_filters).to(device)
generatorF = UnetGenerator(in_dim=in_dim,out_dim=out_dim,num_filter=num_filters).to(device)
discriminatorDy = Discriminator().to(device)
discriminatorDx = Discriminator().to(device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# 위에서 정의한 weights_init 함수를 통해 가중치 초기화하기
generatorG = generatorG.apply(weights_init)
generatorF = generatorF.apply(weights_init)
discriminatorDy = discriminatorDy.apply(weights_init)
discriminatorDx = discriminatorDx.apply(weights_init)

adversarial_loss=nn.BCELoss()
cycleConsistent_loss=nn.L1Loss()
identity_loss=nn.L1Loss()

lr=0.0002

optimizer_GG = torch.optim.Adam(generatorG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_GF = torch.optim.Adam(generatorF.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_Dy = torch.optim.Adam(discriminatorDy.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_Dx = torch.optim.Adam(discriminatorDx.parameters(), lr=lr, betas=(0.5, 0.999))

sum([param.nelement() for param in generatorG.parameters()])

"""# **모델 학습**"""

import time
from tqdm.auto import tqdm

lambdaA=10
n_epochs=100
start_time = time.time()
for epoch in tqdm(range(n_epochs)):
    for X,Y in tqdm((train_loader)):

        X, Y = X.float().to(device), Y.float().to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_Dy.zero_grad()
        optimizer_Dx.zero_grad()

        dis_real_X=discriminatorDx(X)
        dis_real_Y=discriminatorDy(Y)

        Y_hat_forward = generatorG(X) #여름사진을 겨울사진으로
        X_hat_forward = generatorF(Y) #겨울사진을 여름사진으로

        dis_fake_Y=discriminatorDy(Y_hat_forward)
        dis_fake_X=discriminatorDx(X_hat_forward)

        #discriminatorDy에 대한 adversarial loss
        real_loss_Dy = adversarial_loss(dis_real_Y, torch.ones_like(dis_real_Y))
        fake_loss_Dy = adversarial_loss(dis_fake_Y, torch.zeros_like(dis_fake_Y))
        d_loss_Dy = ((real_loss_Dy + fake_loss_Dy)/2) 

        #discriminatorDx에 대한 adversarial loss
        real_loss_Dx = adversarial_loss(dis_real_X, torch.ones_like(dis_real_X))
        fake_loss_Dx = adversarial_loss(dis_fake_X, torch.zeros_like(dis_fake_X))
        d_loss_Dx = ((real_loss_Dx + fake_loss_Dx)/2)

        #discriminator full loss
        Ld=d_loss_Dy+d_loss_Dx
        Ld.backward(retain_graph=True)

        optimizer_Dy.step()
        optimizer_Dx.step()


        # -----------------
        #  Train Generator
        # -----------------

        optimizer_GG.zero_grad()
        optimizer_GF.zero_grad()

        Y_hat_forward = generatorG(X) #여름사진을 겨울사진으로
        X_hat_forward = generatorF(Y) #겨울사진을 여름사진으로
        Y_hat_backward = generatorG(X_hat_forward) #여름사진을 겨울사진으로 바꾼 것을 다시 여름사진으로 
        X_hat_backward = generatorF(Y_hat_forward) #겨울사진을 여름사진으로 바꾼 것을 다시 겨울사진으로

        #양방향에 대한 cycle consistency loss
        cycle_forward=cycleConsistent_loss(X,X_hat_backward)
        cycle_backward=cycleConsistent_loss(Y,Y_hat_backward)
        Lcyc=lambdaA*(cycle_forward+cycle_backward)

        #generatorG에 대한 adversarial loss
        dis_fake_Y=discriminatorDy(Y_hat_forward)
        g_loss_Y = adversarial_loss(dis_fake_Y,torch.ones_like(dis_fake_Y))

        #generatorF에 대한 adversarial loss
        dis_fake_X=discriminatorDx(X_hat_forward)
        g_loss_X = adversarial_loss(dis_fake_X,torch.ones_like(dis_fake_X))
        
        #identity loss
        Lidentity=0.5*lambdaA*(identity_loss(generatorG(Y),Y)+identity_loss(generatorF(X),X))

        #generator에 대한 총 adversarial loss
        Lgan=g_loss_Y+g_loss_X

        #generator full loss
        Lg=Lcyc+Lgan+Lidentity
        Lg.backward(retain_graph=True)

        optimizer_GG.step()
        optimizer_GF.step()

    print('[epoch {}/{}] [D loss: {:.6f}] [G loss: {:.6f}] [Elapsed time: {:.2f}s]'.format(epoch,n_epochs,Ld,Lgan,time.time() - start_time))#에폭의 마지막 loss만 뽑아봄
    
    #output 결과 확인
    predict=Y_hat_forward[0].detach().cpu().numpy()
    predict=0.5*(predict+1)
    X=X[0].detach().cpu().numpy()
    X=0.5*(X+1)
    Y=Y[0].detach().cpu().numpy()
    Y=0.5*(Y+1)
    plt.figure(figsize=(16,18))
    plt.subplot(1,3,1)
    plt.imshow(np.transpose(X,(1,2,0)))
    plt.subplot(1,3,2)
    plt.imshow(np.transpose(Y,(1,2,0)))
    plt.subplot(1,3,3)
    plt.imshow(np.transpose(predict,(1,2,0)))
    plt.show()

    #모델 저장
    torch.save(generatorG.state_dict(), 'model_generatorG_s2w_.pt')
    torch.save(generatorF.state_dict(), 'model_generatorF_s2w_.pt')
    torch.save(discriminatorDy.state_dict(), 'model_discriminatorDy_s2w_.pt')
    torch.save(discriminatorDx.state_dict(), 'model_discriminatorDx_s2w_.pt')

"""# **학습된 모델 불러오기**"""

generatorG.load_state_dict(torch.load('model_generatorG_s2w.pt'))
generatorF.load_state_dict(torch.load('model_generatorF_s2w.pt'))
discriminatorDy.load_state_dict(torch.load('model_discriminatorDy_s2w.pt'))
discriminatorDx.load_state_dict(torch.load('model_discriminatorDx_s2w.pt'))

"""# **모델 테스트**"""

trans = albumentations.Compose([
                              albumentations.Resize(256, 256), 
                              albumentations.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                              ],additional_targets={'image2':'image'})

test_dataset = MyDataset(test_X_files,test_Y_files,transform=trans)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1,shuffle=False)

generatorG.eval()
generatorF.eval()
discriminatorDy.eval()
discriminatorDx.eval()
cur_step=0

with torch.no_grad():
    for X,Y in tqdm(test_loader):
      X,Y=X.float().to(device), Y.float().to(device)
      winter=generatorG(X).detach().cpu()
      summer=generatorF(Y).detach().cpu()
      if cur_step % 10 == 0 and cur_step > 0:
          X=X[0].detach().cpu().numpy()
          X=0.5*(X+1)
          Y=Y[0].detach().cpu().numpy()
          Y=0.5*(Y+1)
          winter=winter[0].detach().cpu().numpy()
          winter=0.5*(winter+1)
          summer=summer[0].detach().cpu().numpy()
          summer=0.5*(summer+1)
          print("summer->winter")
          plt.figure(figsize=(16,18))
          plt.subplot(1,2,1)
          plt.imshow(np.transpose(X,(1,2,0)))
          plt.subplot(1,2,2)
          plt.imshow(np.transpose(winter,(1,2,0)))
          plt.show()
          print("winter->summer")
          plt.figure(figsize=(16,18))
          plt.subplot(1,2,1)
          plt.imshow(np.transpose(Y,(1,2,0)))
          plt.subplot(1,2,2)
          plt.imshow(np.transpose(summer,(1,2,0)))
          plt.show()
      cur_step += 1