# coding: utf-8
import os
import numpy as np
import scipy.io as sio
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize
import math
from scipy.ndimage import rotate
# path = 'E:/Data/ACDC/training/patient001/patient001_4d.nii.gz'
# img_nii = nib.load(path)
# img = img_nii.get_data()
# header = img_nii.header
# size = img.shape
#
# for i in range(0,size[2]):
#
#     plt.figure()
#     plt.subplot(1,2,1)
#     plt.imshow(img[:,:,i,0], cmap='gray')
#     plt.subplot(1,2,2)
#     plt.imshow(img[:, :, i, 2], cmap='gray')
    # plt.subplot(4, 2, 3)
    # plt.imshow(img[:, :, i, 3], cmap='gray')
    # plt.subplot(4, 2, 4)
    # plt.imshow(img[:, :, i, 12], cmap='gray')
    # plt.subplot(4, 2, 5)
    # plt.imshow(img[:, :, i, 16], cmap='gray')
    # plt.subplot(4, 2, 6)
    # plt.imshow(img[:, :, i, 20], cmap='gray')
    # plt.subplot(4, 2, 7)
    # plt.imshow(img[:, :, i, 24], cmap='gray')
    # plt.subplot(4, 2, 8)
    # plt.imshow(img[:, :, i, 28], cmap='gray')

#dir = 'C:/Users/cai/Desktop/4DCT_data/ACDC/training/'
dir = 'D:/testing/testing'
out='D:/4dregression/preprocess/4D'
ed_out='D:/4dregression/preprocess/ed_out'
es_out='D:/4dregression/preprocess/es_out'
ed_gt_out='D:/4dregression/preprocess/ed_gt_out'
es_gt_out='D:/4dregression/preprocess/es_gt_out'

def resample(image, old_spacing, new_spacing, order=3):
    image=image.astype(np.uint16)
    labelsize=image.shape
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(float(image.shape[2])))

    return resize(image, new_shape, order=order, mode='edge')


def convert_to_one_hot(seg):
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res


def Normalization(data):
    data = np.asarray(data)
    min = np.min(data)
    max = np.max(data)
    data = (data - min) / (max - min)
    return data

def rotate_img(label_es,img):
    pi=3.1415926
    a= label_es[:,:,4]
    ymax=0
    xmax=0
    xmin=1000
    ymin=1000
    for i in range(0,a.shape[0]):
        for j in range(0,a.shape[1]):
            if(a[i][j]==1 and (a[i+1][j]==2 or a[i-1][j]==2)):
                if(j>ymax):
                    ymax=j
                    xmax=i
                if(j<ymin):
                    ymin=j
                    xmin=i
    ang = math.atan((xmax - xmin) / (ymax - ymin))
    label_rotate = rotate(label_es, angle=(180 * ang / pi), reshape=False, axes=(1, 0))
    img_rotate = rotate(img, angle=(180 * ang / pi), reshape=False, axes=(1, 0))
    return img_rotate

def crop_img(label_es, img, box_height=128, box_width=128):
    shape=label_es.shape()
    a = label_es.nonzero()
    a_x = a[0]
    a_x_middle = np.median(a[0])
    a_height = max((a_x)) - min((a_x)) + 1

    assert a_height < box_height, 'height小了'
    a_x_start = int(a_x_middle - box_height / 2)
    a_x_left=a_x_start;
    a_x_end = int(a_x_middle + box_height / 2)
    a_x_right=shape[0]-a_x_end

    a_y = a[1]
    a_y_middle = np.median(a_y)
    a_width = max(a_y) - min(a_y) + 1
    # print(a_width,a_height)
    assert a_width < box_width, 'width小了'
    a_y_start = int(a_y_middle - box_width / 2)
    a_y_up=a_y_start
    a_y_end = int(a_y_middle + box_width / 2)
    a_y_down=shape[1]-a_y_end

    img_1 = img[a_x_start:a_x_end, a_y_start:a_y_end, :]
    #plt.imshow(img_1[:,:,5], cmap='gray')
    return img_1,a_x_left,a_x_right,a_y_up,a_y_down

height = []
width = []
def box(img):
    a = img.nonzero()
    x = a[0]
    h = max(x) - min(x) + 1
    height.append(h)
    y = a[1]
    w = max(y) - min(y) + 1
    width.append(w)
    return height, width


def preprocess_image(image, is_seg=False, spacing=None, spacing_target=(1.25, 1.25, 16)):
    if not is_seg:

        image = resample(image, spacing, spacing_target, order=1).astype(np.float32)
        image=Normalization(image)

    else:

        tmp = convert_to_one_hot(image)
        vals = np.unique(image)
        results = []
        for i in range(len(tmp)):
            # results.append(tmp[i])
            results.append(resample(tmp[i].astype(float), spacing, spacing_target, 1)[None])
        image = vals[np.vstack(results).argmax(0)]
    return image


for id in os.listdir(dir):
    id_path = os.path.join(dir,id)
    files = os.listdir(id_path)
    data_path = os.path.join(id_path, files[1])
    data_nii = nib.load(data_path)

    data_4d = data_nii.get_data()

    header = data_nii.header

    size = data_4d.shape
    pixdim = header['pixdim'][1:4]

    imgs = {}

    # 开头和结束两帧,ED与ES帧
    path_img_1 = os.path.join(id_path, files[2])
    path_label_1 = os.path.join(id_path, files[3])
    path_img_2 = os.path.join(id_path, files[4])
    path_label_2 = os.path.join(id_path, files[5])

    img_1_nii = nib.load(path_img_1)
    label_1_nii = nib.load(path_label_1)
    img_2_nii = nib.load(path_img_2)
    label_2_nii = nib.load(path_label_2)



    imgs['moving_vol'] = img_1_nii.get_data()
    imgs['moving_gt'] = label_1_nii.get_data()
    imgs['fixed_vol'] = img_2_nii.get_data()
    imgs['fixed_gt'] = label_2_nii.get_data()

    #获取ES在哪一帧，比如在12帧
    time = int(files[4][-9:-7])
    # j=4
    # while (j<(time-1)):
    #     imgs['a'+str(j).zfill(2)]= data_4d[:, :, :, j]
    #     j += 4
    aa = data_4d[:, :, :, 0]
    assert aa.all() == imgs['moving_vol'].all(), "es图片不一致"
    assert data_4d[:, :, :, time - 1].all() == imgs['fixed_vol'].all(), "ed图片不一致"
    # process the es and ed
    for k in sorted(imgs):
        # print(k)(k == "moving_gt" or k == "fixed_gt")
        imgs[k] = preprocess_image(imgs[k], is_seg=(k == "moving_gt" or k == "fixed_gt"), spacing=pixdim, spacing_target=(1.25, 1.25, 16))
    label = imgs['moving_gt']
    for m in sorted(imgs):
        # imgs[m]= rotate_img(label,imgs[m])
        imgs[m],left,right,up,down= crop_img(label,imgs[m])

    # process 4d data

    #data = np.zeros((128,128,16,size[3]))
    data = np.zeros((128, 128, 16, time))
    for n in range(0,time):
        savepath = 'D:/4dregression/preprocess/mat_test/' + id +'_'+ str(n) +'.mat'
        temp = data_4d[:, :, :, n]
        temp = preprocess_image(temp, is_seg=False, spacing=pixdim, spacing_target=(1.25, 1.25, 16))
        print(temp.shape)
        # temp = rotate_img(label,temp)
        print(temp.shape)
        temp = crop_img(label, temp)
        #将当前temp和ed，es打包
        #存进来命名有moving_vol，moving_gt，fixed_vol，fixed_gt，source
        imgs['source']=temp
        imgs['left']=left
        imgs['right']=right
        imgs['up']=up
        imgs['down']=down
        sio.savemat(savepath,imgs)


        print(temp.shape)
        # data[:, :, :, n] = temp
        # nib.Nifti1Image(data, np.eye(4)).to_filename(
        #     out + '/' + id+ '_4d_ed_es.nii.gz')
        print(data.shape)

    # Td_ed=data[:, :, :, 0]
    # nib.Nifti1Image(Td_ed, np.eye(4)).to_filename(
    #     ed_out + '/' + id + '_ed.nii.gz')
    # Td_es= data[:, :, :, time - 1]
    # nib.Nifti1Image(Td_es, np.eye(4)).to_filename(
    #     es_out + '/' + id + '_es.nii.gz')


    # assert data[:, :, :, 0].all() == imgs['moving_vol'].all(), "处理后es图片不一致"
    # assert data[:, :, :, time - 1].all() == imgs['fixed_vol'].all(), "处理后ed图片不一致"
    # Td_ed_gt=imgs['moving_gt']
    # nib.Nifti1Image(Td_ed_gt, np.eye(4)).to_filename(
    #     ed_gt_out + '/' + id + '_ed_gt.nii.gz')
    # Td_es_gt=imgs['fixed_gt']
    # nib.Nifti1Image(Td_es_gt, np.eye(4)).to_filename(
    #     es_gt_out + '/' + id + '_es_gt.nii.gz')



    # imgs['data'] = data
    # imgs['time'] = time
    # savepath = 'C:/Users/cai/Desktop/4DCT_data/mat_4d_all/' + id + '.mat'
    # sio.savemat(savepath, imgs)



