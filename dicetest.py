#  imports
import numpy as np
import os
import glob
from scipy import misc
import scipy.io as sio
import nibabel as nib
from skimage.transform import resize
# 计算DICE系数！！！
# label用来计算DICE系数
def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0] / new_spacing[0] * float(image.shape[0]))),
                 int(np.round(old_spacing[1] / new_spacing[1] * float(image.shape[1]))),
                 int(np.round(old_spacing[2] / new_spacing[2] * float(image.shape[2]))))
                 # int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge')
def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res
def read_label(image,spacing=(1,1,1),spacing_target=(1,1,1)):
    edsize = image.shape
    tem = convert_to_one_hot(image)
    vals = np.unique(image)
    result = []
    for i in range(len(tem)):
        result.append(resize_image(tem[i].astype(np.int16), spacing, spacing_target, order=1)[None])
    image = vals[np.vstack(result).argmax(0)]
    m = np.zeros((1, 128, 128))
    tem = convert_to_one_hot(image)
    vals = np.unique(image)
    result = []
    for i in range(len(tem)):
        result.append(resize(tem[i].astype(np.int16), (edsize[0],128,128), order=1, mode='edge')[None])
    image = vals[np.vstack(result).argmax(0)]
    image = np.append(image, m, axis=0)
    image = np.append(m, image, axis=0)

    return image
def dice(img1, img2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((img1, img2))) # 输出一维数组
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        top = 2 * np.sum(np.logical_and(img1 == lab, img2 == lab))
        bottom = np.sum(img1 == lab) + np.sum(img2 == lab)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon. 机器最小的正数
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)


dir_img='D:/4dregression/voxel/state/dice_test_img/'
dir_label='D:/4dregression/voxel/state/dice_test_label/'
img1=dir_img+'seg_source0.nii'
img2=dir_label+'patient081_0.mat'
img1=nib.load(img1)
img1=img1.get_data()
img1 = np.transpose(img1, (2, 1, 0))  # xyz-zyx
print(img1.shape)
img2=sio.loadmat(img2)['moving_gt']
img2 = np.transpose(img2, (2, 1, 0))  # xyz-zyx
img2=read_label(img2)
print(img2.shape)
dices = dice(img1, img2, nargout=1)
print(dices)
# dice_result=[]
# with open(os.path.join(dir,'dice.txt'), 'a') as f:
#     for idx in range(0,len(img_paths)):
#         imgname=img_paths[idx]
#         img1 = sio.loadmat(imgname)['warp_gt']
#         img1 = img1[:, :, 1:11]
#         # img1 = threshold(img1)
#         img2 = sio.loadmat(imgname)['fixed_gt']
#         img2 = img2[:, :, 1:11]
#         dices = dice(img1, img2, nargout=1)
#         dice_result.append(dices)
#         f.write(str(dices))
#         f.write("\n")
#     dice_result = np.array(dice_result)
#     dice_sum = np.sum(dice_result, axis=0)
#     length = len(img_paths)
#     final = dice_sum/length
#     f.write("\n")
#     f.write("dice=")
#     f.write(str(final))
# print('length',length)
# print('dice',final)

