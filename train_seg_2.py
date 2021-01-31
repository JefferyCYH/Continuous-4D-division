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
import print_state
import print_state_seg_2
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
import data_utils as data
import torch.utils.data as Datas
import Network as Network
import math
import torch.nn.functional as F
from torch.autograd import Variable
import metrics as criterion
# import print_state
import layer
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
Flownet = Network.VXm(2).to(device)

opt_seg = torch.optim.Adam(Segnet.parameters(),lr=0.00005)
##
pretrained_dict = torch.load('/media/cyh/pkl/net_epoch_source-Flow-Network.pkl')
model_dict = Flownet.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
Flownet.load_state_dict(model_dict)

pretrained_dict = torch.load('/media/cyh/pkl/net_epoch_Source-Seg-Network.pkl')
model_dict = Segnet.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
Segnet.load_state_dict(model_dict)


for epoch in range(50):
    meansegdice= 0
    for step, (source,img_ed,img_es,labeled,labeles) in enumerate(dataloder):
        img_ed = img_ed.to(device).float()
        img_es = img_es.to(device).float()
        labeled = labeled.to(device).float()
        labeles = labeles.to(device).float()
        source=source.to(device).float()
        # mi3, mx3, mi4, mx4 = criterion.location(labeles, labeled, 5, 4)
        depth = img_ed.shape[2]
        ################################################################################
        ed_seg= Segnet(img_ed)
        es_seg=Segnet(img_es)
        seg=Segnet(source)

        with torch.no_grad():
            flow_field_x1, ed_source, flow_field_x2, es_source = Flownet(img_es, img_ed, source)
        ed_source_1 = layer.SpatialTransformer((depth, 128, 128))(ed_seg[:, 0:1, :, :, :], flow_field_x1)
        ed_source_2 = layer.SpatialTransformer((depth, 128, 128))(ed_seg[:, 1:2, :, :, :], flow_field_x1)
        ed_source_3 = layer.SpatialTransformer((depth, 128, 128))(ed_seg[:, 2:3, :, :, :], flow_field_x1)
        ed_source_4 = layer.SpatialTransformer((depth, 128, 128))(ed_seg[:, 3:4, :, :, :], flow_field_x1)
        ed_source = torch.cat((ed_source_1, ed_source_2, ed_source_3, ed_source_4), 1)
        ####################################################################################################
        es_source_1 = layer.SpatialTransformer((depth, 128, 128))(es_seg[:, 0:1, :, :, :], flow_field_x2)
        es_source_2 = layer.SpatialTransformer((depth, 128, 128))(es_seg[:, 1:2, :, :, :], flow_field_x2)
        es_source_3 = layer.SpatialTransformer((depth, 128, 128))(es_seg[:, 2:3, :, :, :], flow_field_x2)
        es_source_4 = layer.SpatialTransformer((depth, 128, 128))(es_seg[:, 3:4, :, :, :], flow_field_x2)
        es_source = torch.cat((es_source_1, es_source_2, es_source_3, es_source_4), 1)

        loss_seg_motion = 0.5*(criterion_dice(seg,es_source)+criterion_dice(seg,ed_source))


        for p in Segnet.parameters():  # reset requires_grad
            p.requires_grad = True
        opt_seg.zero_grad()
        ############
        seg_ed = Segnet(img_ed)
        seg_es = Segnet(img_es)
        # seg_es=Segnet(img_es)
        loss_S = crossentry()
        # print(seg_ed.shape)
        # print(labeled.shape)
        loss_seg_dice = 0.5 * (criterion_dice(seg_ed[:, :, 1:-1, :, :], labeled[:, :, 1:-1, :, :]) + criterion_dice(seg_es[:, :, 1:-1, :, :], labeles[:, :, 1:-1, :, :]))
        loss_seg_ce = 0.5 * (loss_S(seg_ed, labeled) + loss_S(seg_es, labeles))
        #新增运动约束loss_seg_motion
        # labeles_trans = layer.SpatialTransformer((depth, 128, 128))(labeled, flow_field)
        # loss_motion=criterion_dice(labeles_trans[:, :, 1:-1, :, :],labeles[:, :, 1:-1, :, :])
        errS = loss_seg_ce+loss_seg_dice+0.5*loss_seg_motion
        errS.backward()
        opt_seg.step()

        if step % 2 == 0:
            torch.save(Segnet.state_dict(), '/media/cyh/pkl/net_epoch_' + str(epoch) + '-Seg2-Network.pkl')


        meansegdice += loss_seg_dice.data.cpu().numpy()

        if step % 1 == 0:
            print('EPOCH:', epoch, '|Step:', step, '|loss:',errS,'|loss_seg_dice:', loss_seg_dice.data.cpu().numpy(), '|loss_seg_ce:', loss_seg_ce.data.cpu().numpy(),'|loss_motion:',loss_seg_motion.data.cpu().numpy())

            # with torch.no_grad():
            #     # flow_field_x1, ed_source, flow_field_x2, es_source = Flownet(img_es, img_ed, source)
            #     print_state_seg_2.printst(step,ed_source,es_source, seg)

    # print('epoch', epoch, '|seg_dice:',(meansegdice / 894))








# class AverageMeter(object):
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count




# def train_epoch(net_S, opt_S, loss_S, dataloder, epoch, n_epochs, Iters):
#     loss_S_log = AverageMeter()
#     net_S.train()
#     for i in range(Iters):
#         net_S.zero_grad()
#         opt_S.zero_grad()
#         # 真实图像训练
#         input, target = next(dataloader_R.__iter__())
#         if torch.cuda.is_available():
#             input = input.cuda()
#             target = target.cuda()
#
#         seg = net_S(input)
#         errS = loss_S(seg, target)
#         errS.backward()
#         opt_S.step()
#         loss_S_log.update(errS.data, target.size(0))
#
#         res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs),
#                          'Iter: [%d/%d]' % (i + 1, Iters),
#                          'Loss_S %f' % (loss_S_log.avg)])
#         print(res)
#     return
#
#
# def train_net(n_epochs=2, batch_size=1, lr=1e-4, Iters=200, model_name="DenseBiasNet_aug"):
#     shape = (128, 128, 128)
#     train_image_dir = 'data/train/'
#     save_dir = 'results'
#     checkpoint_dir = 'weights'
#     test_image_dir = 'data/test/image/'
#
#     net_S = UNet(in_channel=1, out_channel=2)
#     # net_S.load_state_dict(torch.load('{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, model_name, "200")))
#
#     if torch.cuda.is_available():
#         net_S = net_S.cuda()
#
#     #train_dataset_R = DatasetFromFolder3D_R(train_image_dir, shape=shape, num_classes=2)
#     data = data.train_data
#     dataloder = Datas.DataLoader(dataset=data,batch_size=1,shuffle=True)
#
#     opt_S = torch.optim.Adam(net_S.parameters(), lr=lr)
#
#     loss_S = crossentry()
#
#     for epoch in range(n_epochs):
#         train_epoch(net_S, opt_S, loss_S, dataloder, epoch, n_epochs, Iters)
#         if epoch % 20 == 0:
#             torch.save(net_S.state_dict(), '{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, model_name, epoch))
#     # predict(net_S, save_dir, "test_once")
#     torch.save(net_S.state_dict(), '{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, model_name, n_epochs))
#     predict(net_S, save_dir, test_image_dir)


# def load_nii(path):
#     image = nib.load(path)
#     affine = image.affine
#     image = image.get_data()
#     return image, affine
#
#
# def predict(model, save_path, img_path):
#     print("Predict test data")
#     model.eval()
#     image_filenames = [x for x in os.listdir(img_path) if is_image3d_file(x)]
#     for imagename in image_filenames:
#         print(imagename)
#
#         image, affine = load_nii(img_path + imagename)
#         image = image.astype(np.float32)
#         image = image[np.newaxis, np.newaxis, :, :, :]
#         image = torch.from_numpy(image)
#         if torch.cuda.is_available():
#             image = image.cuda(0)
#         with torch.no_grad():
#             predict = model(image).data.cpu().numpy()
#
#         predict = np.argmax(predict[0], axis=0)
#         predict = predict.astype(np.uint16)
#         predict = nib.Nifti1Image(predict, np.eye(4))
#         nib.save(predict, imagename)
#
#
# def is_image3d_file(filename):
#     return any(filename.endswith(extension) for extension in [".nii"])
#
#
# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     train_net()
