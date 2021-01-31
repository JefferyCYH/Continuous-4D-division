# -*- coding: utf-8 -*-
import os
from os.path import join
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from torch import nn
from torch.utils.data import DataLoader
import torch
import data_utils as data
import torch.utils.data as Datas
# from evaluation.Dice import Dice
import print_state_seg_1
from Network import DenseBiasNet
#from dataloader_2c import DatasetFromFolder3D_R
from loss import crossentry
from loss import dice_coef
import os
import numpy as np
import torch
import torch.utils.data.dataset as Dataset
from skimage.transform import resize
import SimpleITK as sitk
import data_utils_seg1 as data
import torch.utils.data as Datas
import Network as Network
import math
import torch.nn.functional as F
from torch.autograd import Variable
import metrics as criterion
# import print_state
import layer

#############################第一次预训练seg网络
criterion_L1 = torch.nn.L1Loss()
criterion_MSE = torch.nn.MSELoss()
criterion_BCE = torch.nn.BCEWithLogitsLoss()
criterion_CE = criterion.crossentry()
criterion_ncc = criterion.NCC().loss
criterion_grad = criterion.Grad('l2',2).loss
criterion_dice = criterion.DiceMeanLoss()


device = torch.device("cuda:0")

data = data.train_data
dataloder = Datas.DataLoader(dataset=data,batch_size=1,shuffle=True)
Segnet =Network.DenseBiasNet(n_channels=1, n_classes=4).to(device)
# Flownet = Network.VXm(2).to(device)

opt_seg = torch.optim.Adam(Segnet.parameters(),lr=0.0001)
##
# pretrained_dict = torch.load('./pkl/net_epoch_100-Flow-Network.pkl')
# model_dict = Flownet.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# Flownet.load_state_dict(model_dict)

pretrained_dict = torch.load('./pkl/net_epoch_99-Seg-Network.pkl')
model_dict = Segnet.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
Segnet.load_state_dict(model_dict)


for epoch in range(100):
    meansegdice= 0
    for step, (source,img_ed,img_es,labeled,labeles) in enumerate(dataloder):
        img_ed = img_ed.to(device).float()
        img_es = img_es.to(device).float()
        labeled = labeled.to(device).float()
        labeles = labeles.to(device).float()
        # mi3, mx3, mi4, mx4 = criterion.location(labeles, labeled, 5, 4)
        depth = img_ed.shape[2]
        ################################################################################
        # seg= Flownet(img_ed)

        # loss_seg_dice = 0.5*(criterion_dice(seg,es_source_2)+criterion_dice(seg,ed_source_2))
        for p in Segnet.parameters():  # reset requires_grad
            p.requires_grad = True
        opt_seg.zero_grad()
        ############
        seg_ed=Segnet(img_ed)
        seg_es=Segnet(img_es)
        loss_S = crossentry()
        loss_seg_dice = 0.5*(criterion_dice(seg_ed[:, :, 1:-1, :, :], labeled[:, :, 1:-1, :, :])+criterion_dice(seg_es[:, :, 1:-1, :, :], labeles[:, :, 1:-1, :, :]))
        loss_seg_ce=0.5*(loss_S(seg_ed, labeled)+loss_S(seg_es, labeles))
        errS = loss_seg_ce+loss_seg_dice
        errS.backward()
        opt_seg.step()

        if step % 10 == 0:
            torch.save(Segnet.state_dict(), './pkl/net_epoch_' + str(epoch) + '-Seg-Network.pkl')

        meansegdice += loss_seg_dice.data.cpu().numpy()

        if step % 1 == 0:
            print('EPOCH:', epoch, '|Step:',step,'|errS',errS,  '|loss_seg_dice:', loss_seg_dice.data.cpu().numpy(), '|loss_seg_ce:', loss_seg_ce.data.cpu().numpy())

            # with torch.no_grad():
            #     # flow_field_x1, ed_source, flow_field_x2, es_source = Flownet(img_es, img_ed, source)
            #     print_state_seg_1.printst(step, seg_ed)
    #
    print('epoch', epoch, '|seg_dice:',(meansegdice / 99))
