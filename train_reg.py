import os
import numpy as np
import torch
import torch.utils.data.dataset as Dataset
from skimage.transform import resize
import SimpleITK as sitk
import data_utils as data
import torch.utils.data as Datas
import Network as Network
import math
import torch.nn.functional as F
from torch.autograd import Variable
import metrics as criterion
import layer
import print_state

device = torch.device("cuda:0")
data = data.train_data
dataloder = Datas.DataLoader(dataset=data,batch_size=1,shuffle=True)

Flownet = Network.VXm(2).to(device)
### pretrained
# pretrained_dict = torch.load('./pkl/net_epoch_35-Flow-Network.pkl')
# model_dict = Flownet.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# Flownet.load_state_dict(model_dict)

###
criterion_L1 = torch.nn.L1Loss()
criterion_MSE = torch.nn.MSELoss()
criterion_BCE = torch.nn.BCEWithLogitsLoss()
criterion_CE = criterion.crossentry()
criterion_ncc = criterion.NCC().loss
criterion_grad = criterion.Grad('l2',2).loss
criterion_dice = criterion.DiceMeanLoss()

opt_flow = torch.optim.Adam(Flownet.parameters(),lr=0.0001)
for epoch in range(100):
    meanncc = 0
    meanregdice= 0
    meansegdice= 0
    for step, (source,img_ed,img_es,labeled,labeles) in enumerate(dataloder):

        img_ed = img_ed.to(device).float()
        source = source.to(device).float()##ed-0.25
        img_es = img_es.to(device).float()

        depth = img_ed.shape[2]

        opt_flow.zero_grad()
        flow_field_x1,ed_source,flow_field_x2,es_source= Flownet(img_es,img_ed,source)
        ############
        # loss_smooth = 0.5 * (criterion_grad(flow_field_x1[:, :, :, :,:]) +criterion_grad(flow_field_x2[:, :, :, :, :]))
        loss_smooth_x1 = criterion_grad(flow_field_x1[:, :, :, :, :])
        loss_smooth_x2 = criterion_grad(flow_field_x2[:, :, :, :, :])
        loss_smooth=0.5*(loss_smooth_x1+loss_smooth_x2)
        ##############

        # loss_reg_ncc = 0.5*(criterion_ncc(ed_source[:, :, 1:-1, :, :], source[:, :, 1:-1, :, :]) +criterion_ncc(es_source[:, :, 1:-1, :, :], source[:, :, 1:-1, :, :]))
        loss_reg_ncc_x1 = criterion_ncc(ed_source[:, :, 1:-1, :, :], source[:, :, 1:-1, :, :])
        loss_reg_ncc_x2 = criterion_ncc(es_source[:, :, 1:-1, :, :], source[:, :, 1:-1, :, :])
        loss_reg_ncc=0.5*(loss_reg_ncc_x1+loss_reg_ncc_x2)
        ###############
        loss_reg_x1=loss_reg_ncc_x1+loss_smooth_x1
        loss_reg_x2 = loss_reg_ncc_x2 + loss_smooth_x2
        loss_reg = 0.5*(loss_reg_x1  + loss_reg_x2)
        loss_reg.backward()
        opt_flow.step()
        #############
        if step % 10 == 0:
            torch.save(Flownet.state_dict(), '/media/cyh/pkl/net_epoch_' + str(epoch) + '-Flow-Network.pkl')
        meanncc += loss_reg_ncc.data.cpu().numpy()
        # meanregdice += loss_fse.data.cpu().numpy()
        if step % 1 == 0:
            print('EPOCH:', epoch, '|Step:', step, '|loss_reg',loss_reg,'|loss_reg_ncc:', loss_reg_ncc.data.cpu().numpy(), '|loss_smooth:', loss_smooth.data.cpu().numpy())

            # with torch.no_grad():
            #     print_state.printst(step,flow_field_x1, ed_source, flow_field_x2, es_source)

    print('epoch', epoch, '|ncc:', (meanncc / 894))