from os.path import join
from os import listdir

from torch.utils import data
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from scipy import ndimage

from utils.batchgenerators.transforms import MirrorTransform, SpatialTransform

MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def norm_img(image): # 归一化像素值到（0，1）之间，且将溢出值取边界值
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def load_nii(path):
    image = nib.load(path)
    affine = image.affine
    image = image.get_data()
    return image, affine

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii"])

class DatasetFromFolder3D_R(data.Dataset):
    def __init__(self, file_dir, shape, num_classes):
        super(DatasetFromFolder3D_R, self).__init__()
        self.image_filenames = [x for x in listdir(file_dir + 'image/') if is_image_file(x)]
        self.file_dir = file_dir
        self.shape = shape
        self.num_classes = num_classes

    def __getitem__(self, index):
        # 读取image和label
        image, affine = load_nii(self.file_dir + 'image/' + self.image_filenames[index])
        # image = norm_img(image)
        
        target, affine = load_nii(self.file_dir + 'label/' + self.image_filenames[index])
        target = np.where(target > 0, 1, 0)
        # target = norm_img(target)

        target = self.to_categorical(target, self.num_classes)
        target = target.astype(np.float32)
        image = image.astype(np.float32)
        image = image[np.newaxis, :, :, :]
        return image, target

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n))
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __len__(self):
        return len(self.image_filenames)

