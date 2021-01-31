import nibabel as nib
import scipy.io as sio
import numpy as np
matsource='D:/4dregression/preprocess/mat/patient001_4.mat'
out='D:/4dregression/preprocess/test'
source=sio.loadmat(matsource)['source']
img_ed = sio.loadmat(matsource)['moving_vol']
print(img_ed.shape)
img_es = sio.loadmat(matsource)['fixed_vol']
nib.Nifti1Image(source, np.eye(4)).to_filename(
        out + '/' + '7_source.nii.gz')
nib.Nifti1Image(img_ed, np.eye(4)).to_filename(
        out + '/' + '_ed.nii.gz')
nib.Nifti1Image(img_es, np.eye(4)).to_filename(
        out + '/' + '_es.nii.gz')
img_ed_label=sio.loadmat(matsource)['moving_gt']
nib.Nifti1Image(img_ed_label, np.eye(4)).to_filename(
        out + '/' + '_ed_label.nii.gz')
img_es_label=sio.loadmat(matsource)['fixed_gt']
nib.Nifti1Image(img_es_label, np.eye(4)).to_filename(
        out + '/' + '_es_label.nii.gz')
