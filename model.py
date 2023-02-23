
from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time
import cv2


class Sequence_net(nn.Module):

    def __init__(self, use_cuda): # , snapshot=None
        super(Sequence_net, self).__init__()
        self.use_cuda = use_cuda
        self.feat_dim = 16
        # Initialize network trunks with Resnet pre-trained on ImageNet
        self.feat_trunk = torchvision.models.resnet18(pretrained=False)
        self.K = 20
        fc_in_features = self.feat_trunk.fc.in_features
        self.feat_trunk.fc = nn.Linear(fc_in_features, self.feat_dim)
        # Construct network branches for pushing and grasping
        self.fc_net = nn.Sequential(OrderedDict([
            ('sequ_relu0', nn.ReLU(inplace=True)),
            ('sequ_fc0', nn.Linear((self.feat_dim+1)*self.K*3, 4*self.K)),
            ('sequ_relu1', nn.ReLU(inplace=True)),
            ('sequ_fc1', nn.Linear(4*self.K, self.K))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'sequ_' in m[0] in m[0]:
                if isinstance(m[1], nn.Linear):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm1d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []


    def forward(self, input_Hts, items_Hts, templates, uncertainty):
        self.output_prob = []
        self.interm_feat = []
        n = len(templates)
        # Compute intermediate features
        if self.use_cuda:
            self.interm_feat = self.feat_trunk(input_Hts.cuda())
            reshape_feat = torch.cat((self.interm_feat, torch.reshape(torch.from_numpy(np.array(uncertainty)).cuda(),(3*n,1))),dim=1)
        else:
            self.interm_feat = self.feat_trunk(input_Hts)
            reshape_feat = torch.cat((self.interm_feat, torch.reshape(torch.from_numpy(np.array(uncertainty)),(3*n,1))),dim=1)
        # Set Mask
        # print(reshape_feat.shape)
        if n < self.K:
            if self.use_cuda:
                mask_feat = torch.cat((reshape_feat, torch.zeros((3*(self.K-n),self.feat_dim+1)).cuda()), dim = 0)
            else:
                mask_feat = torch.cat((reshape_feat, torch.zeros((3*(self.K-n),self.feat_dim+1))), dim = 0)
        else:
            mask_feat = reshape_feat
        mask_feat = torch.reshape(mask_feat, (1,-1))
        self.output_prob = self.fc_net(mask_feat)
        
        if n < self.K:
            self.output_prob[:,n:self.K] = -np.inf
        predictions = F.softmax(self.output_prob, dim = 1).cpu().data.numpy()
        predictions = [np.arange(n,0,-1)]
        pred = predictions[0][0:n]
        return pred
    
    
#For UNet
class Downsample(nn.Module):
    def __init__(self, channel):
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=2, padding=1,  bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self,x):
        return self.layer(x)


class Upsample(nn.Module):
    def __init__(self,channel):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel//2,kernel_size=(1,1),stride=1)

    def forward(self,x,featuremap):
        x = F.interpolate(x,size=featuremap.shape[2],mode='nearest')
        x = self.conv1(x)
        x = torch.cat((x,featuremap),dim=1)
        return x
    
class conv_block(nn.Module):
    def __init__(self,in_c,out_c):
        super(conv_block,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=1, padding=1,bias = False),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
class Placement_net(nn.Module):
    def __init__(self, use_cuda, transforms):
        super(Placement_net, self).__init__()
        self.use_cuda = use_cuda
        self.resolution = 200
        self.TopHeight = 0.3
        self.transforms = transforms
        self.channels = 8
        self.layer1 = conv_block(3,self.channels)
        self.layer2 = Downsample(self.channels)
        self.layer3 = conv_block(self.channels,self.channels*2)
        self.layer4 = Downsample(self.channels*2)
        self.layer5 = conv_block(self.channels*2,self.channels*4)
        self.layer6 = Downsample(self.channels*4)
        self.layer7 = conv_block(self.channels*4,self.channels*8)
        self.layer8 = Downsample(self.channels*8)
        self.layer9 = conv_block(self.channels*8,self.channels*16)
        self.layer10 = Upsample(self.channels*16)
        self.layer11 = conv_block(self.channels*16,self.channels*8)
        self.layer12 = Upsample(self.channels*8)
        self.layer13 = conv_block(self.channels*8,self.channels*4)
        self.layer14 = Upsample(self.channels*4)
        self.layer15 = conv_block(self.channels*4,self.channels*2)
        self.layer16 = Upsample(self.channels*2)
        self.layer17 = conv_block(self.channels*2,self.channels)
        self.layer18 = nn.Conv2d(self.channels,1,kernel_size=(1,1),stride=1)
        self.act = nn.Sigmoid()
        
    def Check_Available(self, template, item_Hts, item_Hbs, Hbox):
        Hc = cv2.resize(Hbox, (50, 50))
        output_mat = np.zeros((24,1,self.resolution,self.resolution))
        for i in range(len(item_Hts)):
            if template in ['bowl','mug','banana','bottle'] and i >= 4:
                continue
            w,h = round(item_Hts[i].shape[0]/4), round(item_Hts[i].shape[1]/4)
            Ht = cv2.resize(item_Hts[i], (h, w))
            Hb = cv2.resize(item_Hbs[i], (h, w))
            for X in range(0, 51-w):
                for Y in range(0, 51-h):
                    Z = np.max(Hc[X:X+w, Y:Y+h]-Hb)
                    Update = np.maximum((Ht>0)*(Ht+Z), Hc[X:X+w,Y:Y+h])
                    if np.max(Update) <= self.TopHeight:
                        score = 0.01*(X+Y)+np.sum(Update**1.5)-np.sum(Hc[X:X+w,Y:Y+h]**1.5)
                        top = np.max(Update)
                        score *= np.exp(top/2)
                        output_mat[i,0,4*X:4*X+4,4*Y:4*Y+4] = 1/score
        return output_mat
    
    def forward(self,x, template, item_Hts, item_Hbs, Hbox):
        if self.use_cuda:
            x = x.cuda()
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        x = self.layer5(x)
        f3 = x
        x = self.layer6(x)
        x = self.layer7(x)
        f4 = x
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x,f4)
        x = self.layer11(x)
        x = self.layer12(x,f3)
        x = self.layer13(x)
        x = self.layer14(x,f2)
        x = self.layer15(x)
        x = self.layer16(x,f1)
        x = self.layer17(x)
        x = self.layer18(x)
        self.output = self.act(x)
        mat = self.output.cpu().data.numpy()
        mat = self.Check_Available(template, item_Hts, item_Hbs, Hbox)
        return mat
    
