from os.path import join
from os import listdir
from torch.utils import data
import numpy as np

from aug_tool import Crop, MirrorTransform, SpatialTransform
from batchgenerators.transforms import GammaTransform


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".raw"])

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, file_dir, shape=(128, 128, 128), is_aug=True):
        super(DatasetFromFolder3D, self).__init__()
        self.is_aug = is_aug
        self.image_filenames = [x for x in listdir(join(file_dir, "image")) if is_image_file(x)]
        self.file_dir = file_dir
        self.shape = shape
        if is_aug:
            self.random_crop = Crop(shape)
            self.mirror_transform = MirrorTransform()
            self.spatial_transform = SpatialTransform(patch_center_dist_from_border=np.array(shape) // 2,
                                                 do_elastic_deform=False,
                                                 alpha=(0., 1000.),
                                                 sigma=(10., 13.),
                                                 do_rotation=True,
                                                 angle_x=(-np.pi / 9, np.pi / 9),
                                                 angle_y=(-np.pi / 9, np.pi / 9),
                                                 angle_z=(-np.pi / 9, np.pi / 9),
                                                 do_scale=True,
                                                 scale=(0.75, 1.25),
                                                 border_mode_data='constant',
                                                 border_cval_data=0,
                                                 order_data=1,
                                                 random_crop=True)
            self.gamma_transform = GammaTransform(gamma_range=(0.75, 1.25))
        else:
            self.random_crop = Crop(shape)

    def _load_image_label(self, index):
        shape = (200, 150, 150)

        _img = np.fromfile(join(self.file_dir, 'image', self.image_filenames[index]), np.int16)
        _img = _img.reshape(shape)

        target_tumor_kidney = np.fromfile(
            join(self.file_dir, "label_tumor_kidney", self.image_filenames[index][:-7] + "Label_200.raw"),
            dtype=np.uint16)
        target_tumor_kidney = target_tumor_kidney.reshape(shape)
        target_tumor_kidney = np.where(target_tumor_kidney == 100, 2, target_tumor_kidney)
        target_tumor_kidney = np.where(target_tumor_kidney == 300, 3, target_tumor_kidney)

        target_artery = np.fromfile(join(self.file_dir, "label_artery", self.image_filenames[index]), dtype=np.uint16)
        target_artery = target_artery.reshape(shape)
        target_artery = np.where(target_artery > 0, 4, 0)

        target_vein = np.fromfile(join(self.file_dir, "label_vein", self.image_filenames[index]), dtype=np.uint16)
        target_vein = target_vein.reshape(shape)
        target_vein = np.where(target_vein > 0, 1, 0)

        target = np.concatenate([target_tumor_kidney[np.newaxis, :, :, :],
                                 target_vein[np.newaxis, :, :, :],
                                 target_artery[np.newaxis, :, :, :]], axis=0)
        _lab = np.max(target, axis=0)

        return _img, _lab

    def __getitem__(self, index):
        image, target = self._load_image_label(index)
        image = image[np.newaxis, np.newaxis, :, :, :]
        target = target[np.newaxis, np.newaxis, :, :, :]
        if self.is_aug:
            # mirror
            mirror_code = self.mirror_transform.rand_code()
            image = self.mirror_transform.augment_mirroring(image, mirror_code)
            target = self.mirror_transform.augment_mirroring(target, mirror_code)
            # spatial
            coords = self.spatial_transform.rand_coords(image.shape)
            image = self.spatial_transform.augment_spatial(image, coords, is_label=False)
            target = self.spatial_transform.augment_spatial(target, coords, is_label=True)

        return image, target, self.image_filenames[index]

    def norm_image(self, image, target, shape):
        shape_image = image.shape
        if shape_image[2] < shape[2]:
            image = np.pad(image, ((0, 0), (0, 0), (0, shape[2] - shape_image[2])), mode='constant')
            target = np.pad(target, ((0, 0), (0, 0), (0, shape[2] - shape_image[2])), mode='constant')
        return image, target

    def __len__(self):
        return len(self.image_filenames)



