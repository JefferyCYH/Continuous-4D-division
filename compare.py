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
import print_state
import layer



device = torch.device("cuda:0")
print(torch.__version__)
data = data.train_data
dataloder = Datas.DataLoader(dataset=data,batch_size=1,shuffle=False)

Segnet =Network.Segmentation().to(device)
Flownet = Network.VXm(2).to(device)
# ##
# pretrained_dict = torch.load('./pkl/net_epoch_126-Flow-Network.pkl')
# model_dict = Flownet.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# Flownet.load_state_dict(model_dict)
#
# pretrained_dict = torch.load('./pkl/net_epoch_126-Seg-Network.pkl')
# model_dict = Segnet.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# Segnet.load_state_dict(model_dict)
##
criterion_L1 = torch.nn.L1Loss()
criterion_MSE = torch.nn.MSELoss()
criterion_BCE = torch.nn.BCEWithLogitsLoss()
criterion_CE = criterion.crossentry()
criterion_ncc = criterion.NCC().loss
criterion_grad = criterion.Grad('l2',2).loss
criterion_dice = criterion.DiceMeanLoss()

opt_flow = torch.optim.Adam(Flownet.parameters(),lr=0.0001)
opt_seg = torch.optim.Adam(Segnet.parameters(),lr=0.0001)


for epoch in range(200):
    loss_continous_motion=0
    meanncc = 0
    meanregdice= 0
    meansegdice= 0

    for step, (img_ed,img_pre,img_mid,img_aft,img_es,labeled,labeles) in enumerate(dataloder):
        img_ed = img_ed.to(device).float()
        img_pre = img_pre.to(device).float()  ##ed-0.25
        img_mid = img_mid.to(device).float()
        img_aft = img_aft.to(device).float()  ##ed-0.75
        img_es = img_es.to(device).float()
        labeled = labeled.to(device).float()
        labeles = labeles.to(device).float()
        mi3, mx3, mi4, mx4 = criterion.location(labeles, labeled, 5, 4)

        depth = img_ed.shape[2]
        # segment
        for p in Segnet.parameters():  # reset requires_grad
            p.requires_grad = True
        opt_seg.zero_grad()
        #output es_seg,ed_seg
        # fuse_es_seg, fuse_ed_seg, es_seg, ed_seg, es_seg_flow, ed_seg_flow = Segnet(img_es, img_ed,flow_filed)
        es_seg, ed_seg, es_seg_flow, ed_seg_flow = Segnet(img_es, img_ed)

        loss_seg_ce = (#criterion_CE(fuse_es_seg[:, :, 1:-1, :, :], labeles[:, :, 1:-1, :, :]) + \
                        #criterion_CE(fuse_ed_seg[:, :, 1:-1, :, :], labeled[:, :, 1:-1, :, :]) + \
                       criterion_CE(es_seg_flow[:, :, 1:-1, :, :], labeles[:, :, 1:-1, :, :]) + \
                       criterion_CE(ed_seg_flow[:, :, 1:-1, :, :], labeled[:, :, 1:-1, :, :])) / 4

        loss_seg_dice = (#criterion_dice(fuse_es_seg[:, :, 1:-1, :, :], labeles[:, :, 1:-1, :, :]) + \
                         #criterion_dice(fuse_ed_seg[:, :, 1:-1, :, :], labeled[:, :, 1:-1, :, :]) + \
                         criterion_dice(es_seg_flow[:, :, 1:-1, :, :], labeles[:, :, 1:-1, :, :]) + \
                         criterion_dice(ed_seg_flow[:, :, 1:-1, :, :], labeled[:, :, 1:-1, :, :])) / 4

        #desgin a Continuous motion loss loss_continous_motion


        loss_seg = loss_seg_ce + loss_seg_dice+loss_continous_motion

        img_seg_ed = ed_seg
        img_seg_es = es_seg
        img_seg_ed = img_seg_ed.to(device).float()
        img_seg_es = img_seg_es.to(device).float()

        for p in Flownet.parameters():  # reset requires_grad
            p.requires_grad = False
        opt_flow.zero_grad()




        warped_ed, flow_field, speed_field, step1_flow, step2_flow, step3_flow, step4_flow,endo = Flownet(img_seg_es,img_seg_ed)
############
        loss_smooth =  0.5* (criterion_grad(flow_field[:, :, :, mi3:mx3, mi4:mx4]) + \
                                   criterion_grad(flow_field[:, :, :, :, :]))
##############
        loss_reg_ncc = ((criterion_ncc(img_ed[:, :, 1:-1, mi3:mx3, mi4:mx4], warped_ed[:, :, 1:-1, mi3:mx3, mi4:mx4])+ \
                       criterion_ncc(img_es[:, :, 1:-1, mi3:mx3, mi4:mx4], endo[:, :, 1:-1, mi3:mx3, mi4:mx4]))/2 +\
                       (criterion_ncc(img_aft[:, :, 1:-1, mi3:mx3, mi4:mx4], step1_flow[:, :, 1:-1, mi3:mx3, mi4:mx4])+ \
                       criterion_ncc(img_mid[:, :, 1:-1, mi3:mx3, mi4:mx4], step2_flow[:, :, 1:-1, mi3:mx3, mi4:mx4])+ \
                       criterion_ncc(img_pre[:, :, 1:-1, mi3:mx3, mi4:mx4], step3_flow[:, :, 1:-1, mi3:mx3, mi4:mx4])+ \
                       criterion_ncc(img_ed[:, :, 1:-1, mi3:mx3, mi4:mx4], step4_flow[:, :, 1:-1, mi3:mx3, mi4:mx4]))/4)/2
##############
        speed_combin = 0.25 * (speed_field[:, 0:3, :, :, :] + speed_field[:, 3:6, :, :, :] + \
                               speed_field[:, 6:9, :, :, :] + speed_field[:, 9:12, :, :, :])
        fm = flow_field - flow_field.mean()
        f = fm/fm.std()
        sm = speed_combin - speed_combin.mean()
        s = sm / sm.std()
        loss_com = criterion_MSE(f, s)
#####################
        w0 = torch.unsqueeze(torch.unsqueeze(labeles[0, 0, :, :, :], 0), 0)
        w1 = torch.unsqueeze(torch.unsqueeze(labeles[0, 1, :, :, :], 0), 0)
        w2 = torch.unsqueeze(torch.unsqueeze(labeles[0, 2, :, :, :], 0), 0)
        w3 = torch.unsqueeze(torch.unsqueeze(labeles[0, 3, :, :, :], 0), 0)
        w0 = layer.SpatialTransformer((depth, 128, 128))(w0, flow_field)
        w1 = layer.SpatialTransformer((depth, 128, 128))(w1, flow_field)
        w2 = layer.SpatialTransformer((depth, 128, 128))(w2, flow_field)
        w3 = layer.SpatialTransformer((depth, 128, 128))(w3, flow_field)
        # print(flfinseg0.shape)
        ws = F.softmax(torch.cat([w0, w1, w2, w3], dim=1), dim=1)
        loss_fse = criterion_dice(ws[:, :, 1:-1, :, :], labeled[:, :, 1:-1, :, :])

###############
        loss_reg = loss_smooth+ loss_com + loss_reg_ncc + 0.01 *loss_fse
        loss_continous_motion=loss_reg_ncc
        flow_field_reg=flow_field
################################
        loss_seg.backward()
        loss_reg.backward(retain_graph=True)
        opt_seg.step()
        opt_flow.step()





        if step % 2 == 0:
            torch.save(Flownet.state_dict(), './pkl/net_epoch_' + str(epoch) + '-Flow-Network.pkl')
            torch.save(Segnet.state_dict(), './pkl/net_epoch_' + str(epoch) + '-Seg-Network.pkl')

        meanncc += loss_reg_ncc.data.cpu().numpy()
        meanregdice += loss_fse.data.cpu().numpy()
        meansegdice += loss_seg_dice.data.cpu().numpy()

        if step % 1 == 0:
            print('EPOCH:', epoch, '|Step:', step,'|loss_reg_ncc:', loss_reg_ncc.data.cpu().numpy(),\
                  '|loss_fse',(loss_fse).data.cpu().numpy(),'|loss_smooth:', loss_smooth.data.cpu().numpy(),\
                  '|loss_seg_dice:', loss_seg_dice.data.cpu().numpy(),'|loss_seg_ce:', loss_seg_ce.data.cpu().numpy())

            with torch.no_grad():
                print_state.printst(step, img_ed, img_es, img_pre, img_mid, img_aft, labeled, labeles, \
                                    warped_ed, step1_flow, step2_flow, step3_flow, step4_flow, \
                                    flow_field, speed_combin,\
                                    es_seg, ed_seg, es_seg_flow, ed_seg_flow)

    print('epoch',epoch,'|ncc:',(meanncc/159),'|reg_dice:',(meanregdice/159),'|seg_dice:',(meansegdice/159))
