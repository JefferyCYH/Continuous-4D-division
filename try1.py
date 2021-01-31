import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import layer

device = torch.device("cuda:1")


def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)


class Unet(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Unet, self).__init__()

        self.inc = nn.Sequential(
            nn.Conv3d(inchannel, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=False)
        )
        self.down1 = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=False)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=False)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(inplace=False)
        )
        self.uptconv1 = nn.ConvTranspose3d(512, 512, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.up1 = nn.Sequential(
            nn.Conv3d(768, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=False)
        )
        self.uptconv2 = nn.ConvTranspose3d(256, 256, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.up2 = nn.Sequential(
            nn.Conv3d(384, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=False)
        )
        self.uptconv3 = nn.ConvTranspose3d(128, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.up3 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=False)
        )
        self.outc = nn.Sequential(
            nn.Conv3d(64, outchannel, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        uptconv1 = self.uptconv1(x4)
        x5 = self.up1(crop_and_concat(x3, uptconv1))
        uptconv2 = self.uptconv2(x5)
        x6 = self.up2(crop_and_concat(x2, uptconv2))
        uptconv3 = self.uptconv3(x6)
        x7 = self.up3(crop_and_concat(x1, uptconv3))
        x8 = self.outc(x7)

        return x8


class VXm(nn.Module):
    def __init__(self, inchannel):
        super(VXm, self).__init__()

        self.unet = Unet(inchannel, 16)
        self.flow = nn.Conv3d(16, 3, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # self.transformer = layer.SpatialTransformer((10,192,192))

    def forward(self, source, target):
        depth = source.shape[2]
        # y_source = layer.SpatialTransformer((depth, 192, 192))(source, source)

        x = torch.cat([source, target], dim=1)
        x = self.unet(x)
        flow_field = self.flow(x)
        y_source = layer.SpatialTransformer((depth, 192, 192))(source, flow_field)

        return y_source, flow_field


class Four(nn.Module):
    def __init__(self):
        super(Four, self).__init__()

        self.reg = VXm(2)
        self.unet = Unet(1, 4)
        # self.transformer = layer.SpatialTransformer((10, 192, 192))
        self.v1 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(inplace=False)
        )
        self.v2 = nn.Conv3d(16, 4, kernel_size=3, padding=1)

        # self.w1 = nn.Conv3d(4, 4, kernel_size=1)
        # self.w2 = nn.Conv3d(4, 4, kernel_size=1)
        #
        # self.vote1 = nn.Sequential(
        #     # nn.Conv3d(192, 64, kernel_size=3, padding=1),
        #     # nn.BatchNorm3d(64),
        #     # nn.LeakyReLU(inplace=False),
        #
        #     nn.Conv3d(4, 4, kernel_size=1)
        # )

    def forward(self, start, final, pre_img, mid_img, aft_img,labeled,labeles):
        depth = start.shape[2]

        fsim, fsflow = self.reg(start, final)
        preimflow = layer.SpatialTransformer((depth, 192, 192))(start, 0.25 * fsflow)
        midimflow = layer.SpatialTransformer((depth, 192, 192))(start, 0.5 * fsflow)
        afimflow = layer.SpatialTransformer((depth, 192, 192))(start, 0.75 * fsflow)
        fake_es = layer.SpatialTransformer((depth, 192, 192))(final, -1 * fsflow)

        startseg1 = self.unet(start)
        flfinseg0 = torch.unsqueeze(torch.unsqueeze(startseg1[0, 0, :, :, :], 0), 0)
        flfinseg1 = torch.unsqueeze(torch.unsqueeze(startseg1[0, 1, :, :, :], 0), 0)
        flfinseg2 = torch.unsqueeze(torch.unsqueeze(startseg1[0, 2, :, :, :], 0), 0)
        flfinseg3 = torch.unsqueeze(torch.unsqueeze(startseg1[0, 3, :, :, :], 0), 0)
        flfinseg0 = layer.SpatialTransformer((depth, 192, 192))(flfinseg0, fsflow)
        flfinseg1 = layer.SpatialTransformer((depth, 192, 192))(flfinseg1, fsflow)
        flfinseg2 = layer.SpatialTransformer((depth, 192, 192))(flfinseg2, fsflow)
        flfinseg3 = layer.SpatialTransformer((depth, 192, 192))(flfinseg3, fsflow)
        flfinseg = torch.cat([flfinseg0, flfinseg1, flfinseg2, flfinseg3], dim=1)

        finalseg1 = self.unet(final)
        flstaseg0 = torch.unsqueeze(torch.unsqueeze(finalseg1[0, 0, :, :, :], 0), 0)
        flstaseg1 = torch.unsqueeze(torch.unsqueeze(finalseg1[0, 1, :, :, :], 0), 0)
        flstaseg2 = torch.unsqueeze(torch.unsqueeze(finalseg1[0, 2, :, :, :], 0), 0)
        flstaseg3 = torch.unsqueeze(torch.unsqueeze(finalseg1[0, 3, :, :, :], 0), 0)
        flstaseg0 = layer.SpatialTransformer((depth, 192, 192))(flstaseg0, -1 * fsflow)
        flstaseg1 = layer.SpatialTransformer((depth, 192, 192))(flstaseg1, -1 * fsflow)
        flstaseg2 = layer.SpatialTransformer((depth, 192, 192))(flstaseg2, -1 * fsflow)
        flstaseg3 = layer.SpatialTransformer((depth, 192, 192))(flstaseg3, -1 * fsflow)
        flstaseg = torch.cat([flstaseg0, flstaseg1, flstaseg2, flstaseg3], dim=1)

        startseg = F.softmax(self.v2(self.v1(torch.cat([startseg1, flstaseg], dim=1))), dim=1)
        finalseg = F.softmax(self.v2(self.v1(torch.cat([finalseg1, flfinseg], dim=1))), dim=1)

        #####reg_label
        flowfn0 = torch.unsqueeze(torch.unsqueeze(labeles[0, 0, :, :, :], 0), 0)
        flowfn1 = torch.unsqueeze(torch.unsqueeze(labeles[0, 1, :, :, :], 0), 0)
        flowfn2 = torch.unsqueeze(torch.unsqueeze(labeles[0, 2, :, :, :], 0), 0)
        flowfn3 = torch.unsqueeze(torch.unsqueeze(labeles[0, 3, :, :, :], 0), 0)
        flowfn0 = layer.SpatialTransformer((depth, 192, 192))(flowfn0, fsflow)
        flowfn1 = layer.SpatialTransformer((depth, 192, 192))(flowfn1, fsflow)
        flowfn2 = layer.SpatialTransformer((depth, 192, 192))(flowfn2, fsflow)
        flowfn3 = layer.SpatialTransformer((depth, 192, 192))(flowfn3, fsflow)
        flowfn = torch.cat([flowfn0, flowfn1, flowfn2, flowfn3], dim=1)

        flowst0 = torch.unsqueeze(torch.unsqueeze(labeled[0, 0, :, :, :], 0), 0)
        flowst1 = torch.unsqueeze(torch.unsqueeze(labeled[0, 1, :, :, :], 0), 0)
        flowst2 = torch.unsqueeze(torch.unsqueeze(labeled[0, 2, :, :, :], 0), 0)
        flowst3 = torch.unsqueeze(torch.unsqueeze(labeled[0, 3, :, :, :], 0), 0)
        flowst0 = layer.SpatialTransformer((depth, 192, 192))(flowst0, -1 * fsflow)
        flowst1 = layer.SpatialTransformer((depth, 192, 192))(flowst1, -1 * fsflow)
        flowst2 = layer.SpatialTransformer((depth, 192, 192))(flowst2, -1 * fsflow)
        flowst3 = layer.SpatialTransformer((depth, 192, 192))(flowst3, -1 * fsflow)
        flowst = torch.cat([flowst0, flowst1, flowst2, flowst3], dim=1)


        return fsim, fake_es, preimflow, midimflow, afimflow, fsflow, startseg, finalseg, flowfn, flowst




