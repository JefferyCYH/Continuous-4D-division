import SimpleITK as sitk
from pandas import np
import nibabel as nib

def printst(step, ed_source,es_source,source):
    pt = ed_source[0, 0, :, :, :].data.cpu().numpy()
    # pt = np.transpose(pt, (2, 1, 0))
    out = sitk.GetImageFromArray(pt)
    out.SetSpacing((1.00, 1.00, 1.00))
    sitk.WriteImage(out, './state/ed-source' + str(step) + '.nii')

    pt = es_source[0, 0, :, :, :].data.cpu().numpy()
    # pt = np.transpose(pt, (2, 1, 0))
    out = sitk.GetImageFromArray(pt)
    out.SetSpacing((1.00, 1.00, 1.00))
    sitk.WriteImage(out, './state/es-source' + str(step) + '.nii')

    pt = source[0, 0, :, :, :].data.cpu().numpy()
    # pt = np.transpose(pt, (2, 1, 0))
    out = sitk.GetImageFromArray(pt)
    out.SetSpacing((1.00, 1.00, 1.00))
    sitk.WriteImage(out, './state/source' + str(step) + '.nii')



# mm = seg_ed[0, 1, :, :, :] * 1 + seg_ed[0, 2, :, :, :] * 2 + seg_ed[0, 3, :, :, :] * 3
    # pt = mm.data.cpu().numpy()
    # # pt = final_seg[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.000, 1.000, 1.000))
    # sitk.WriteImage(out, './state/seg-ed' + str(step) + '.nii')
    #
    # mm = es_seg_flow[0, 1, :, :, :] * 1 + es_seg_flow[0, 2, :, :, :] * 2 + es_seg_flow[0, 3, :, :, :] * 3
    # pt = mm.data.cpu().numpy()
    # # pt = flfinseg[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/reg-seg-es' + str(step) + '.nii')
    #
    # mm = ed_seg_flow[0, 1, :, :, :] * 1 + ed_seg_flow[0, 2, :, :, :] * 2 + ed_seg_flow[0, 3, :, :, :] * 3
    # pt = mm.data.cpu().numpy()
    # # pt = flstaseg[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/reg-seg-ed' + str(step) + '.nii')
    #
    # mm = seg_es[0, 1, :, :, :] * 1 + seg_es[0, 2, :, :, :] * 2 + seg_es[0, 3, :, :, :] * 3
    # pt = mm.data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # # pt = start_seg[0, 0, :, :, :].data.cpu().numpy()
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/seg-es' + str(step) + '.nii')
    #
    # mm = seg_ed[0, 1, :, :, :] * 1 + seg_ed[0, 2, :, :, :] * 2 + seg_ed[0, 3, :, :, :] * 3
    # pt = mm.data.cpu().numpy()
    # # pt = final_seg[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/seg-ed' + str(step) + '.nii')
