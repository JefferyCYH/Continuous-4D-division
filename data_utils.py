import os
import numpy as np
import torch
import torch.utils.data.dataset as Dataset
from skimage.transform import resize
import SimpleITK as sitk
import random
import scipy.io as sio
from scipy import ndimage,misc
# from batchgenerators.transforms import MirrorTransform, SpatialTransform
from aug_tool import Crop, MirrorTransform, SpatialTransform

def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0] / new_spacing[0] * float(image.shape[0]))),
                 int(np.round(old_spacing[1] / new_spacing[1] * float(image.shape[1]))),
                 int(np.round(old_spacing[2] / new_spacing[2] * float(image.shape[2]))))
                 # int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge')

# def read_image(image,spacing,spacing_target):
#     image = resize_image(image, spacing, spacing_target, order=1).astype(np.float32)
#     edsize = image.shape
#     image = resize(image, (edsize[0], 128, 128), order=1, mode='edge').astype(np.float32)
#     m = np.zeros((1, 128, 128))
#     image = np.append(image, m, axis=0)
#     image = np.append(m, image, axis=0)
#     return image
def read_image(image,spacing,spacing_target):
    image = resize_image(image, spacing, spacing_target, order=1).astype(np.float32)
    edsize = image.shape
    image = resize(image, (edsize[0], 128, 128), order=1, mode='edge').astype(np.float32)
    m = np.zeros((1, 128, 128))
    image = np.append(image, m, axis=0)
    image = np.append(m, image, axis=0)
    return image

def read_label(image,spacing,spacing_target):
    edsize = image.shape
    tem = convert_to_one_hot(image)
    vals = np.unique(image)
    result = []
    for i in range(len(tem)):
        result.append(resize_image(tem[i].astype(np.float32), spacing, spacing_target, order=1)[None])
    image = vals[np.vstack(result).argmax(0)]
    m = np.zeros((1, 128, 128))
    tem = convert_to_one_hot(image)
    vals = np.unique(image)
    result = []
    for i in range(len(tem)):
        result.append(resize(tem[i].astype(np.float32), (edsize[0],128,128), order=1, mode='edge')[None])
    image = vals[np.vstack(result).argmax(0)]
    image = np.append(image, m, axis=0)
    image = np.append(m, image, axis=0)

    return image



def normor(image):
    image -=image.mean()
    image /=image.std()
    return image

def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res


Path_mat='/media/cyh/mat/'
Path_4d = './data/4D/'
Path_lab = './data/label/'

Path_test_mat='./data/mat_test/'
# path_test = './data/test/4D/'
# path_test_label = './data/test/label/'

img_4D = []
img_4D_test = []

for root,dirs,files in os.walk(Path_mat):
    for file in files:
        mat_path = os.path.join(root, file)

        img_4D.append(mat_path)

for root,dirs,files in os.walk(Path_test_mat):
    for file in files:
        mat_test_path = os.path.join(root, file)

        img_4D_test.append(mat_test_path)



class Data(Dataset.Dataset):
    def __init__(self,im_4D):
        self.im_4D = im_4D

        self.mirror_transform = MirrorTransform()
        self.spatial_transform = SpatialTransform(do_elastic_deform=False,
                                              alpha=(0., 1000.),
                                              sigma=(10., 13.),
                                              do_rotation=True,
                                              angle_x=(0, 2 * np.pi),
                                              angle_y=(0, 0),
                                              angle_z=(0, 0),
                                              do_scale=True,
                                              scale=(0.75, 1.25),
                                              border_mode_data='constant',
                                              border_cval_data=0,
                                              order_data=1,
                                              random_crop=False)
    def __len__(self):
        return len(self.im_4D)

    def __getitem__(self, index):

        ###########读mat
        source=sio.loadmat(self.im_4D[index])['source']
        img_ed = sio.loadmat(self.im_4D[index])['moving_vol']
        # print(img_ed.shape)
        img_es = sio.loadmat(self.im_4D[index])['fixed_vol']
        spacing=(1.000,1.000,1.000)
        spacing=list(spacing)

        ############
        labeled=sio.loadmat(self.im_4D[index])['moving_gt']
        labeles = sio.loadmat(self.im_4D[index])['fixed_gt']

        spacing_target = (1.000,1.000,1.000)
        spacing_target = list(spacing_target)



        "数据增强"
        # img_ed = img_ed[np.newaxis, np.newaxis, :, :, :]
        # labeled = labeled[np.newaxis, np.newaxis, :, :, :]
        # img_es = img_es[np.newaxis, np.newaxis, :, :, :]
        # labeles = labeles[np.newaxis, np.newaxis, :, :, :]
        # source = source[np.newaxis, np.newaxis, :, :, :]
        "镜像"
        mirror_code = self.mirror_transform.rand_code()
        img_ed = self.mirror_transform.augment_mirroring(img_ed, mirror_code)
        labeled = self.mirror_transform.augment_mirroring(labeled, mirror_code)
        img_es = self.mirror_transform.augment_mirroring(img_es, mirror_code)
        labeles = self.mirror_transform.augment_mirroring(labeles, mirror_code)
        source = self.mirror_transform.augment_mirroring(source, mirror_code)
        "spatial transform"
        coords = self.spatial_transform.rand_coords(img_ed.shape)
        # coords = self.spatial_transform.rand_coords(img_ed.shape)
        img_ed = self.spatial_transform.augment_spatial(img_ed, coords, is_label=False)
        labeled = self.spatial_transform.augment_spatial(labeled, coords, is_label=True)
        img_es = self.spatial_transform.augment_spatial(img_es, coords, is_label=False)
        labeles = self.spatial_transform.augment_spatial(labeles, coords, is_label=True)
        source = self.spatial_transform.augment_spatial(source, coords, is_label=False)
        #

        #
        img_ed = np.transpose(img_ed, (2, 1, 0))  # xyz-zyx
        img_es = np.transpose(img_es, (2, 1, 0))  # xyz-zyx
        labeled = np.transpose(labeled, (2, 1, 0))  # xyz-zyx
        labeles = np.transpose(labeles, (2, 1, 0))  # xyz-zyx
        source=np.transpose(source,(2,1,0))

        # img_ed = image4[0,:,:,:]

        ####
        source=read_image(source,spacing,spacing_target)
        img_ed = read_image(img_ed,spacing,spacing_target)
        img_es = read_image(img_es,spacing,spacing_target)


        labeled = read_label(labeled, spacing, spacing_target)
        labeles = read_label(labeles, spacing, spacing_target)

        source=normor(source)
        img_ed = normor(img_ed)
        img_es = normor(img_es)
        # img_mid = normor(img_mid)
        # img_pre = normor(img_pre)
        # img_aft = normor(img_aft)
        labeled = convert_to_one_hot(labeled)
        labeles = convert_to_one_hot(labeles)
        # print(source.shape)
        # print(img_ed.shape)
        # print(img_es.shape)
        source=source[np.newaxis,:,:,:]
        img_ed = img_ed[np.newaxis, :, :, :]
        img_es = img_es[np.newaxis, :, :, :]
        # print(img_ed.shape)


        return source,img_ed,img_es,labeled,labeles

class Data_test(Dataset.Dataset):
    def __init__(self,img_4D_test):
        self.im_4D = img_4D_test

    def __len__(self):
        return len(self.im_4D)

    def __getitem__(self, index):

        #####4D
        source = sio.loadmat(self.im_4D[index])['source']
        source = np.transpose(source, (2, 1, 0))  # xyz-zxy
        img_ed = sio.loadmat(self.im_4D[index])['moving_vol']
        img_ed = np.transpose(img_ed, (2, 1, 0))  # xyz-zxy
        img_es = sio.loadmat(self.im_4D[index])['fixed_vol']
        img_es = np.transpose(img_es, (2, 1, 0))  # xyz-zxy

        # spacing = np.array(source.GetSpacing())[[2, 1, 0]]  ###[z,x,y]
        spacing = (1.000, 1.000, 1.000)
        spacing = list(spacing)

        ############
        labeled = sio.loadmat(self.im_4D[index])['moving_gt']
        labeled = np.transpose(labeled, (2, 1, 0))
        # labeled = sitk.GetArrayFromImage(labed).astype(float)
        labeles = sio.loadmat(self.im_4D[index])['fixed_gt']
        labeles = np.transpose(labeles, (2, 1, 0))
        # labeles = sitk.GetArrayFromImage(labes).astype(float)

        spacing_target = (1.000, 1.000, 1.000)
        spacing_target = list(spacing_target)
        # spacing_target[0] = spacing[0]

        # shapeim4 = image4.shape
        # loc_mid = int(shapeim4[0]*0.5)
        # loc_pre = int(shapeim4[0]*0.25)
        # loc_aft = int(shapeim4[0]*0.75)

        # img_ed = image4[0,:,:,:]

        ####
        source = read_image(source, spacing, spacing_target)
        img_ed = read_image(img_ed, spacing, spacing_target)
        img_es = read_image(img_es, spacing, spacing_target)

        labeled = read_label(labeled, spacing, spacing_target)
        labeles = read_label(labeles, spacing, spacing_target)

        source = normor(source)
        img_ed = normor(img_ed)
        img_es = normor(img_es)
        # img_mid = normor(img_mid)
        # img_pre = normor(img_pre)
        # img_aft = normor(img_aft)
        labeled = convert_to_one_hot(labeled)
        labeles = convert_to_one_hot(labeles)

        source = source[np.newaxis, :, :, :]
        img_ed = img_ed[np.newaxis, :, :, :]
        img_es = img_es[np.newaxis, :, :, :]
        print(source.shape)

        return source, img_ed, img_es, labeled, labeles


train_data = Data(img_4D)
test_data = Data_test(img_4D_test)

