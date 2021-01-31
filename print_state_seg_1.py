import SimpleITK as sitk
from pandas import np
import nibabel as nib


def threshold(data):
 data[np.where((0.5 > data))] = 0
 data[np.where((0.5 <= data) & (data < 1.5))] = 1
 data[np.where((1.5 <= data) & (data < 2.5))] = 2
 data[np.where((2.5 <= data) & (data < 3.5))] = 3
 return data

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
  labels = np.unique(np.concatenate((img1, img2)))  # 输出一维数组
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


def printst(step, seg_source,seg_ed,seg_es,labeled,labeles):
 mm = seg_source[0, 1, :, :, :] * 1 + seg_source[0, 2, :, :, :] * 2 + seg_source[0, 3, :, :, :] * 3
 pt = mm.data.cpu().numpy()
 # pt = np.transpose(pt, (2, 1, 0))
 # pt = start_seg[0, 0, :, :, :].data.cpu().numpy()
 out = sitk.GetImageFromArray(pt)
 out.SetSpacing((1, 1, 1))
 sitk.WriteImage(out, './state/seg_source' + str(step) + '.nii')
    # pt=np.argmax(pt[1],axis=1)
    # out = sitk.GetImageFromArray(seg_source)
    # out.SetSpacing((1.000, 1.000, 1.000))
    # sitk.WriteImage(out, './state/seg-source' + str(step) + '.nii')

 dice_result_ed = []
 mm1 = seg_ed[0, 1, :, :, :] * 1 + seg_ed[0, 2, :, :, :] * 2 + seg_ed[0, 3, :, :, :] * 3
 mm2 = labeled[0, 1, :, :, :] * 1 + labeled[0, 2, :, :, :] * 2 + labeled[0, 3, :, :, :] * 3
 pt1 = mm1.data.cpu().numpy()
 pt1 = threshold(pt1)
 pt2 = mm2.data.cpu().numpy()
 # print(pt1.shape)
 # print(pt2.shape)
 # labeled=labeled.data.cpu.numpy()
 diceresult_ed = dice(pt1, pt2, nargout=1)
 dice_result_ed.append(diceresult_ed)
 dice_result_ed = np.array(dice_result_ed)
 dice_sum = np.sum(dice_result_ed, axis=0)
 print('step' + str(step) + 'diceresult_ed:|' + str(dice_sum))

 dice_result_es = []
 mm1 = seg_es[0, 1, :, :, :] * 1 + seg_es[0, 2, :, :, :] * 2 + seg_es[0, 3, :, :, :] * 3
 mm2 = labeles[0, 1, :, :, :] * 1 + labeles[0, 2, :, :, :] * 2 + labeles[0, 3, :, :, :] * 3
 pt1 = mm1.data.cpu().numpy()
 pt1=threshold(pt1)
 pt2=  mm2.data.cpu().numpy()
 # print(pt1.shape)
 # print(pt2.shape)
 # labeled=labeled.data.cpu.numpy()
 diceresult_es=dice(pt1,pt2,nargout=1)
 dice_result_es.append(diceresult_es)
 dice_result_es = np.array(dice_result_es)
 dice_sum = np.sum(dice_result_es, axis=0)
 print('step'+str(step)+'diceresult_es:|'+str(dice_sum))

 # out = sitk.GetImageFromArray(pt)
 # out.SetSpacing((1, 1, 1))
 # sitk.WriteImage(out, './state/seg_source' + str(step) + '.nii')




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
