import torch
import numpy as np
import torch.nn as nn
from torch import nn, cat, add
import torch.nn.functional as F
from torch.distributions.normal import Normal
import layer

device = torch.device("cuda:0")

def crop_and_concat( upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)

class Unet(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(Unet, self).__init__()

        self.inc = nn.Sequential(
            nn.Conv3d(inchannel,32,kernel_size=3,padding=1),
            nn.GroupNorm(4,32),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4,64),
            nn.LeakyReLU(inplace=False)
        )
        self.down1 = nn.Sequential(
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4,64),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4,128),
            nn.LeakyReLU(inplace=False)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4,128),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(4,256),
            nn.LeakyReLU(inplace=False)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(4,256),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.GroupNorm(4,512),
            nn.LeakyReLU(inplace=False)
        )
        self.uptconv1 = nn.ConvTranspose3d(512, 512, kernel_size=(1,2,2), stride=(1,2,2))
        self.up1 = nn.Sequential(
            nn.Conv3d(768, 256, kernel_size=3, padding=1),
            nn.GroupNorm(4,256),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(4,256),
            nn.LeakyReLU(inplace=False)
        )
        self.uptconv2 = nn.ConvTranspose3d(256, 256, kernel_size=(1,2,2), stride=(1,2,2))
        self.up2 = nn.Sequential(
            nn.Conv3d(384, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4,128),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(4,128),
            nn.LeakyReLU(inplace=False)
        )
        self.uptconv3 = nn.ConvTranspose3d(128, 128, kernel_size=(1,2,2), stride=(1,2,2))
        self.up3 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4,64),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4,64),
            nn.LeakyReLU(inplace=False)
        )
        self.outc = nn.Sequential(
            nn.Conv3d(64, outchannel, kernel_size=1)
        )

    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        uptconv1 = self.uptconv1(x4)
        x5 = self.up1(crop_and_concat(x3,uptconv1))
        uptconv2 = self.uptconv2(x5)
        x6 = self.up2(crop_and_concat(x2,uptconv2))
        uptconv3 = self.uptconv3(x6)
        x7 = self.up3(crop_and_concat(x1,uptconv3))
        x8 = self.outc(x7)

        return x8

class VXm(nn.Module):
    def __init__(self,inchannel):
        super(VXm, self).__init__()

        self.unet1 = Unet(inchannel,16)
        self.unet2=Unet(inchannel,16)

        # ####speed
        # self.speed = nn.Conv3d(16,3*4,kernel_size=3,padding=1)
        # self.speed.weight = nn.Parameter(Normal(0, 1e-5).sample(self.speed.weight.shape))
        # self.speed.bias = nn.Parameter(torch.zeros(self.speed.bias.shape))


        ####flow
        self.flow = nn.Conv3d(16,3,kernel_size=3,padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # self.transformer = layer.SpatialTransformer((10,192,192))

    def forward(self,ed,es,source):
        # print(ed.shape)
        # print(source.shape)
        x1 = torch.cat([ed, source], dim=1)
        x1 = self.unet1(x1)
        flow_field_x1 = self.flow(x1)
        depth = source.shape[2]
        # print(ed.shape)
        # print(flow_field_x1.shape)
        ed_source = layer.SpatialTransformer((depth, 128, 128))(ed, flow_field_x1)
        # ed_source_1 = layer.SpatialTransformer((depth, 128, 128))(ed, flow_field_x1[:,0:1,:,:,:])
        # ed_source_2 = layer.SpatialTransformer((depth, 128, 128))(ed, flow_field_x1[:, 1:2, :, :, :])
        # ed_source_3 = layer.SpatialTransformer((depth, 128, 128))(ed, flow_field_x1[:, 2:3, :, :, :])
        # ed_source = F.softmax(torch.cat((ed_source_1, ed_source_2, ed_source_3), 1), 1)
        # print(ed.shape)
        # print(flow_field_x1[:,0:1,:,:,:].shape)


        x2 = torch.cat([es, source], dim=1)
        x2 = self.unet2(x2)
        flow_field_x2 = self.flow(x2)
        es_source = layer.SpatialTransformer((depth, 128, 128))(es, flow_field_x2)
        # es_source_1 = layer.SpatialTransformer((depth, 128, 128))(es, flow_field_x2[:, 0:1, :, :, :])
        # es_source_2 = layer.SpatialTransformer((depth, 128, 128))(es, flow_field_x2[:, 1:2, :, :, :])
        # es_source_3 = layer.SpatialTransformer((depth, 128, 128))(es, flow_field_x2[:, 2:3, :, :, :])
        # es_source = F.softmax(torch.cat((es_source_1, es_source_2, es_source_3), 1), 1)


        return flow_field_x1,ed_source,flow_field_x2,es_source



        # depth = source.shape[2]
        # # y_source = layer.SpatialTransformer((depth, 192, 192))(source, source)
        #
        # x = torch.cat([source, target], dim=1)
        # x = self.unet(x)
        #
        # speed_field = self.speed(x)
        #
        # flow_field = self.flow(x)
        #
        # y_source = layer.SpatialTransformer((depth,128,128))(source, flow_field)
        #
        # step1_flow = layer.SpatialTransformer((depth,128,128))(source, 0.25*speed_field[:,0:3,:,:,:])
        # step2_flow = layer.SpatialTransformer((depth, 128, 128))(step1_flow, 0.25*speed_field[:,3:6,:,:,:])
        # step3_flow = layer.SpatialTransformer((depth, 128, 128))(step2_flow, 0.25*speed_field[:, 6:9, :, :, :])
        # step4_flow = layer.SpatialTransformer((depth, 128, 128))(step3_flow, 0.25*speed_field[:, 9:12, :, :, :])
        #
        # endo = layer.SpatialTransformer((depth,128,128))(target, -1*flow_field)
        #
        # return y_source,flow_field,speed_field,step1_flow,step2_flow,step3_flow,step4_flow,endo


# class Seg(nn.Module):
#     def __init__(self):
#         super(Seg, self).__init__()
#
#         model = Segcyc()
#
#     def forward(self,start,final,pre_img,mid_img,aft_img,warp):
#         image =
# #         flow = ttorch.stack((start,pre_img,mid_img,aft_img,final),dim=0)orch.stack((warp*-1,warp*0.25,warp*0.25,warp*0.25,warp*0.25),dim=0)




# class Segmentation(nn.Module):
#     def __init__(self):
#         super(Segmentation, self).__init__()
#
#         self.unet = Unet(1, 4)
#         self.vote1 = nn.Conv3d(4, 8, kernel_size=3,padding=1)
#         self.vote2 = nn.Conv3d(4, 8, kernel_size=3,padding=1)
#         self.fu = nn.Sequential(
#             nn.Conv3d(16, 8, kernel_size=3, padding=1),
#             nn.GroupNorm(4, 8),
#             nn.LeakyReLU(inplace=False),
#             nn.Conv3d(8, 4, kernel_size=1)
#         )
#
#     def forward(self, es, ed, speed):
#         depth = es.shape[2]
#
#         # flow = 0.25*(speed[:,0:3,:,:,:]+speed[:,3:6,:,:,:]+speed[:,6:9,:,:,:]+speed[:,9:12,:,:,:])
#
#         es_seg = F.softmax(self.unet(es), dim=1)
#         ed_seg = F.softmax(self.unet(ed), dim=1)
#
#         w0 = es_seg[:, 0:1, :, :, :]
#         w1 = es_seg[:, 1:2, :, :, :]
#         w2 = es_seg[:, 2:3, :, :, :]
#         w3 = es_seg[:, 3:4, :, :, :]
#         w0 = layer.SpatialTransformer((depth, 128, 128))(w0, speed)
#         w1 = layer.SpatialTransformer((depth, 128, 128))(w1, speed)
#         w2 = layer.SpatialTransformer((depth, 128, 128))(w2, speed)
#         w3 = layer.SpatialTransformer((depth, 128, 128))(w3, speed)
#         ed_seg_flow = torch.cat([w0, w1, w2, w3], dim=1)
#
#         w0 = ed_seg[:, 0:1, :, :, :]
#         w1 = ed_seg[:, 1:2, :, :, :]
#         w2 = ed_seg[:, 2:3, :, :, :]
#         w3 = ed_seg[:, 3:4, :, :, :]
#         w0 = layer.SpatialTransformer((depth, 128, 128))(w0, -1*speed)
#         w1 = layer.SpatialTransformer((depth, 128, 128))(w1, -1*speed)
#         w2 = layer.SpatialTransformer((depth, 128, 128))(w2, -1*speed)
#         w3 = layer.SpatialTransformer((depth, 128, 128))(w3, -1*speed)
#         es_seg_flow = torch.cat([w0, w1, w2, w3], dim=1)
#
#         fuse_es_seg = F.softmax(self.fu(torch.cat([self.vote2(es_seg_flow ),self.vote1(es_seg)],dim=1)), dim=1)
#         fuse_ed_seg = F.softmax(self.fu(torch.cat([self.vote2(ed_seg_flow),self.vote1(ed_seg)],dim=1)), dim=1)
#
#         return fuse_es_seg, fuse_ed_seg, es_seg, ed_seg, es_seg_flow, ed_seg_flow

##########################################################################################



class conv_bias(nn.Module):
    def __init__(self, in_ch, out_ch, bias_size=1):
        super(conv_bias, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.bias_size = bias_size

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        x_bias = x[:, 0:self.bias_size, :, :, :]
        return x_bias, x

class DenseBiasNet_base(nn.Module):
    def __init__(self, n_channels, depth=(16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16), bias=1):
        super(DenseBiasNet_base, self).__init__()
        self.depth = depth
        self.conv0 = conv_bias(n_channels, depth[0], bias_size=bias)

        self.conv1 = conv_bias(depth[0], depth[1], bias_size=bias)

        in_chan = bias
        self.conv2 = conv_bias(depth[1] + in_chan, depth[2], bias_size=bias)

        in_chan = in_chan + bias
        self.conv3 = conv_bias(depth[2] + in_chan, depth[3], bias_size=bias)

        in_chan = in_chan + bias
        self.conv4 = conv_bias(depth[3] + in_chan, depth[4], bias_size=bias)

        in_chan = in_chan + bias
        self.conv5 = conv_bias(depth[4] + in_chan, depth[5], bias_size=bias)

        in_chan = in_chan + bias
        self.conv6 = conv_bias(depth[5] + in_chan, depth[6], bias_size=bias)

        in_chan = in_chan + bias
        self.conv7 = conv_bias(depth[6] + in_chan, depth[7], bias_size=bias)

        in_chan = in_chan + bias
        self.conv8 = conv_bias(depth[7] + in_chan, depth[8], bias_size=bias)

        in_chan = in_chan + bias
        self.conv9 = conv_bias(depth[8] + in_chan, depth[9], bias_size=bias)

        in_chan = in_chan + bias
        self.conv10 = conv_bias(depth[9] + in_chan, depth[10], bias_size=bias)

        in_chan = in_chan + bias
        self.conv11 = conv_bias(depth[10] + in_chan, depth[11], bias_size=bias)

        in_chan = in_chan + bias
        self.conv12 = conv_bias(depth[11] + in_chan, depth[12], bias_size=bias)

        in_chan = in_chan + bias
        self.conv13 = conv_bias(depth[12] + in_chan, depth[13], bias_size=bias)

        in_chan = in_chan + bias
        self.conv14 = conv_bias(depth[13] + in_chan, depth[14], bias_size=bias)

        in_chan = in_chan + bias
        self.conv15 = conv_bias(depth[14] + in_chan, depth[15], bias_size=bias)

        in_chan = in_chan + bias
        self.conv16 = conv_bias(depth[15] + in_chan, depth[16], bias_size=bias)

        in_chan = in_chan + bias
        self.conv17 = conv_bias(depth[16] + in_chan, depth[17], bias_size=bias)

        self.up_1_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_1_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_2_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_2_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_2_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_2_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_3_0_2 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_3_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_3_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_3_1_2 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_3_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_3_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_4_0_3 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_4_0_2 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_4_0_1 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_4_0_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_4_1_3 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_4_1_2 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_4_1_1 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_4_1_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_3_2_2 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_3_2_1 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_3_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_3_3_2 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_3_3_1 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_3_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_2_2_1 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_2_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_2_3_1 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_2_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_1_2_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.up_1_3_0 = nn.ConvTranspose3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))

        self.down_0_0_1 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.down_0_0_2 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.down_0_0_3 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.down_0_0_4 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))

        self.down_0_1_1 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.down_0_1_2 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.down_0_1_3 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.down_0_1_4 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))

        self.down_1_0_2 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.down_1_0_3 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.down_1_0_4 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))

        self.down_1_1_2 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.down_1_1_3 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.down_1_1_4 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))

        self.down_2_0_3 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.down_2_0_4 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))

        self.down_2_1_3 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))
        self.down_2_1_4 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))

        self.down_3_0_4 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))

        self.down_3_1_4 = nn.Conv3d(bias, bias, kernel_size=(1,2,2), stride=(1,2,2))

        self.maxpooling = nn.MaxPool3d((1,2,2))
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)
    def forward(self, x):
        # block0
        x_bias_0_0_0, x = self.conv0(x)
        x_bias_0_1_0, x = self.conv1(x)

        x_bias_0_0_1 = self.down_0_0_1(x_bias_0_0_0)
        x_bias_0_0_2 = self.down_0_0_2(x_bias_0_0_1)
        x_bias_0_0_3 = self.down_0_0_3(x_bias_0_0_2)
        x_bias_0_0_4 = self.down_0_0_4(x_bias_0_0_3)

        x_bias_0_1_1 = self.down_0_1_1(x_bias_0_1_0)
        x_bias_0_1_2 = self.down_0_1_2(x_bias_0_1_1)
        x_bias_0_1_3 = self.down_0_1_3(x_bias_0_1_2)
        x_bias_0_1_4 = self.down_0_1_4(x_bias_0_1_3)

        # block1
        x = self.maxpooling(x)
        x_bias_1_0_1, x = self.conv2(cat([x, x_bias_0_0_1], dim=1))
        x_bias_1_1_1, x = self.conv3(cat([x, x_bias_0_0_1, x_bias_0_1_1], dim=1))

        x_bias_1_0_0 = self.up_1_0_0(x_bias_1_0_1)
        x_bias_1_0_2 = self.down_1_0_2(x_bias_1_0_1)
        x_bias_1_0_3 = self.down_1_0_3(x_bias_1_0_2)
        x_bias_1_0_4 = self.down_1_0_4(x_bias_1_0_3)

        x_bias_1_1_0 = self.up_1_1_0(x_bias_1_1_1)
        x_bias_1_1_2 = self.down_1_1_2(x_bias_1_1_1)
        x_bias_1_1_3 = self.down_1_1_3(x_bias_1_1_2)
        x_bias_1_1_4 = self.down_1_1_4(x_bias_1_1_3)

        # block2
        x = self.maxpooling(x)
        x_bias_2_0_2, x = self.conv4(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2], dim=1))
        x_bias_2_1_2, x = self.conv5(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2], dim=1))

        x_bias_2_0_1 = self.up_2_0_1(x_bias_2_0_2)
        x_bias_2_0_0 = self.up_2_0_0(x_bias_2_0_1)
        x_bias_2_0_3 = self.down_2_0_3(x_bias_2_0_2)
        x_bias_2_0_4 = self.down_2_0_4(x_bias_2_0_3)

        x_bias_2_1_1 = self.up_2_1_1(x_bias_2_1_2)
        x_bias_2_1_0 = self.up_2_1_0(x_bias_2_1_1)
        x_bias_2_1_3 = self.down_2_1_3(x_bias_2_1_2)
        x_bias_2_1_4 = self.down_2_1_4(x_bias_2_1_3)

        # block3
        x = self.maxpooling(x)
        x_bias_3_0_3, x = self.conv6(
            cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3], dim=1))
        x_bias_3_1_3, x = self.conv7(cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3,
                                          x_bias_2_1_3], dim=1))

        x_bias_3_0_2 = self.up_3_0_2(x_bias_3_0_3)
        x_bias_3_0_1 = self.up_3_0_1(x_bias_3_0_2)
        x_bias_3_0_0 = self.up_3_0_0(x_bias_3_0_1)
        x_bias_3_0_4 = self.down_3_0_4(x_bias_3_0_3)

        x_bias_3_1_2 = self.up_3_1_2(x_bias_3_1_3)
        x_bias_3_1_1 = self.up_3_1_1(x_bias_3_1_2)
        x_bias_3_1_0 = self.up_3_1_0(x_bias_3_1_1)
        x_bias_3_1_4 = self.down_3_1_4(x_bias_3_1_3)

        # block4
        x = self.maxpooling(x)
        x_bias_4_0_4, x = self.conv8(
            cat([x, x_bias_0_0_4, x_bias_0_1_4, x_bias_1_0_4, x_bias_1_1_4, x_bias_2_0_4, x_bias_2_1_4, x_bias_3_0_4],
                dim=1))
        x_bias_4_1_4, x = self.conv9(cat([x, x_bias_0_0_4, x_bias_0_1_4, x_bias_1_0_4, x_bias_1_1_4, x_bias_2_0_4,
                                          x_bias_2_1_4, x_bias_3_0_4, x_bias_3_1_4], dim=1))

        x_bias_4_0_3 = self.up_4_0_3(x_bias_4_0_4)
        x_bias_4_0_2 = self.up_4_0_2(x_bias_4_0_3)
        x_bias_4_0_1 = self.up_4_0_1(x_bias_4_0_2)
        x_bias_4_0_0 = self.up_4_0_0(x_bias_4_0_1)

        x_bias_4_1_3 = self.up_4_1_3(x_bias_4_1_4)
        x_bias_4_1_2 = self.up_4_1_2(x_bias_4_1_3)
        x_bias_4_1_1 = self.up_4_1_1(x_bias_4_1_2)
        x_bias_4_1_0 = self.up_4_1_0(x_bias_4_1_1)

        # block5
        x = self.up(x)
        x_bias_3_2_3, x = self.conv10(
            cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3, x_bias_2_1_3, x_bias_3_0_3,
                 x_bias_3_1_3, x_bias_4_0_3], dim=1))
        x_bias_3_3_3, x = self.conv11(
            cat([x, x_bias_0_0_3, x_bias_0_1_3, x_bias_1_0_3, x_bias_1_1_3, x_bias_2_0_3, x_bias_2_1_3, x_bias_3_0_3,
                 x_bias_3_1_3, x_bias_4_0_3, x_bias_4_1_3], dim=1))

        x_bias_3_2_2 = self.up_3_2_2(x_bias_3_2_3)
        x_bias_3_2_1 = self.up_3_2_1(x_bias_3_2_2)
        x_bias_3_2_0 = self.up_3_2_0(x_bias_3_2_1)

        x_bias_3_3_2 = self.up_3_3_2(x_bias_3_3_3)
        x_bias_3_3_1 = self.up_3_3_1(x_bias_3_3_2)
        x_bias_3_3_0 = self.up_3_3_0(x_bias_3_3_1)

        # block6
        x = self.up(x)
        x_bias_2_2_2, x = self.conv12(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2, x_bias_2_0_2,
                                          x_bias_2_1_2, x_bias_3_0_2, x_bias_3_1_2, x_bias_4_0_2, x_bias_4_1_2,
                                          x_bias_3_2_2], dim=1))
        x_bias_2_3_2, x = self.conv13(cat([x, x_bias_0_0_2, x_bias_0_1_2, x_bias_1_0_2, x_bias_1_1_2, x_bias_2_0_2,
                                          x_bias_2_1_2, x_bias_3_0_2, x_bias_3_1_2, x_bias_4_0_2, x_bias_4_1_2,
                                          x_bias_3_2_2, x_bias_3_3_2], dim=1))

        x_bias_2_2_1 = self.up_2_2_1(x_bias_2_2_2)
        x_bias_2_2_0 = self.up_2_2_0(x_bias_2_2_1)

        x_bias_2_3_1 = self.up_2_3_1(x_bias_2_3_2)
        x_bias_2_3_0 = self.up_2_3_0(x_bias_2_3_1)

        # block7
        x = self.up(x)
        x_bias_1_2_1, x = self.conv14(cat([x, x_bias_0_0_1, x_bias_0_1_1, x_bias_1_0_1, x_bias_1_1_1, x_bias_2_0_1,
                                           x_bias_2_1_1, x_bias_3_0_1, x_bias_3_1_1, x_bias_4_0_1, x_bias_4_1_1,
                                           x_bias_3_2_1, x_bias_3_3_1, x_bias_2_2_1], dim=1))
        x_bias_1_3_1, x = self.conv15(cat([x, x_bias_0_0_1, x_bias_0_1_1, x_bias_1_0_1, x_bias_1_1_1, x_bias_2_0_1,
                                           x_bias_2_1_1, x_bias_3_0_1, x_bias_3_1_1, x_bias_4_0_1, x_bias_4_1_1,
                                           x_bias_3_2_1, x_bias_3_3_1, x_bias_2_2_1, x_bias_2_3_1], dim=1))

        x_bias_1_2_0 = self.up_1_2_0(x_bias_1_2_1)
        x_bias_1_3_0 = self.up_1_3_0(x_bias_1_3_1)

        # block8
        x = self.up(x)
        x_bias_0_2_0, x = self.conv16(cat([x, x_bias_0_0_0, x_bias_0_1_0, x_bias_1_0_0, x_bias_1_1_0, x_bias_2_0_0,
                                           x_bias_2_1_0, x_bias_3_0_0, x_bias_3_1_0, x_bias_4_0_0, x_bias_4_1_0,
                                           x_bias_3_2_0, x_bias_3_3_0, x_bias_2_2_0, x_bias_2_3_0, x_bias_1_2_0],
                                          dim=1))

        x_bias_0_3_0, x = self.conv17(cat([x, x_bias_0_0_0, x_bias_0_1_0, x_bias_1_0_0, x_bias_1_1_0, x_bias_2_0_0,
                                           x_bias_2_1_0, x_bias_3_0_0, x_bias_3_1_0, x_bias_4_0_0, x_bias_4_1_0,
                                           x_bias_3_2_0, x_bias_3_3_0, x_bias_2_2_0, x_bias_2_3_0, x_bias_1_2_0,
                                           x_bias_1_3_0], dim=1))

        return x

class DenseBiasNet(nn.Module):
    def __init__(self, n_channels, n_classes, depth=(16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16), bias=4):
        super(DenseBiasNet, self).__init__()
        self.densebisanet = DenseBiasNet_base(n_channels, depth, bias)
        self.out_conv = nn.Conv3d(depth[-1], n_classes, 1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        Z = x.size()[2]
        print(Z)
        Y = x.size()[3]
        print(Y)
        X = x.size()[4]
        print(X)
        diffZ = (16 - x.size()[2] % 16) % 16
        diffY = (16 - x.size()[3] % 16) % 16
        diffX = (16 - x.size()[4] % 16) % 16

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        x = self.densebisanet(x)
        x = self.out_conv(x)
        x = self.softmax(x)
        return x[:, :, diffZ//2: Z+diffZ//2, diffY//2: Y+diffY//2, diffX // 2:X + diffX // 2]
