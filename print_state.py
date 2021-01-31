import SimpleITK as sitk


def printst(step,flow_field_x1, ed_source, flow_field_x2, es_source):
    # pt = source[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.00, 1.00, 1.00))
    # sitk.WriteImage(out, './state/source' + str(step) + '.nii')

    # pt = img_ed[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.00, 1.00, 1.00))
    # sitk.WriteImage(out, './state/ed' + str(step) + '.nii')
    #
    # pt = img_es[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.00, 1.00, 1.00))
    # sitk.WriteImage(out, './state/es' + str(step) + '.nii')

    # pt = img_pre[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/esaft' + str(step) + '.nii')
    #
    # pt = img_mid[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/esmid' + str(step) + '.nii')
    #
    # pt = img_aft[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/espre' + str(step) + '.nii')

    # mm = labeled[0, 1, :, :, :] * 1 + labeled[0, 2, :, :, :] * 2 + labeled[0, 3, :, :, :] * 3
    # pt = mm.data.cpu().numpy()
    # # pt = labeled[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.00, 1.00, 1.00))
    # sitk.WriteImage(out, './state/labeled' + str(step) + '.nii')
    #
    # mm = labeles[0, 1, :, :, :] * 1 + labeles[0, 2, :, :, :] * 2 + labeles[0, 3, :, :, :] * 3
    # pt = mm.data.cpu().numpy()
    # # pt = labeles[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.00, 1.00, 1.00))
    # sitk.WriteImage(out, './state/labeles' + str(step) + '.nii')

    # pt = warped_ed[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/es-ed' + str(step) + '.nii')

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

    # pt = step1_flow[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/speed-es-pre' + str(step) + '.nii')
    #
    # pt = step2_flow[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/speed-es-mid' + str(step) + '.nii')
    #
    # pt = step3_flow[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/speed-es-aft' + str(step) + '.nii')
    #
    # pt = step4_flow[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/speed-es-ed' + str(step) + '.nii')

    pt = flow_field_x1[0, :, :, :, :].data.cpu().numpy()
    # pt = np.transpose(pt, (2, 1, 0))
    out = sitk.GetImageFromArray(pt)
    out.SetSpacing((1.00, 1.00, 1.00))
    sitk.WriteImage(out, './state/ed_source_flow' + str(step) + '.nii')

    pt = flow_field_x2[0, :, :, :, :].data.cpu().numpy()
    # pt = np.transpose(pt, (2, 1, 0))
    out = sitk.GetImageFromArray(pt)
    out.SetSpacing((1.00, 1.00, 1.00))
    sitk.WriteImage(out, './state/es_source_flow' + str(step) + '.nii')

    # pt = flow_field[0, :, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/wholeflow' + str(step) + '.nii')
    #
    # pt = speed_combin[0, :, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/speed-wholeflow' + str(step) + '.nii')
    #
    # mm = fuse_es_seg[0, 1, :, :, :] * 1 + fuse_es_seg[0, 2, :, :, :] * 2 + fuse_es_seg[0, 3, :, :, :] * 3
    # pt = mm.data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # # pt = start_seg[0, 0, :, :, :].data.cpu().numpy()
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/fuse-seg-es' + str(step) + '.nii')
    #
    # mm = fuse_ed_seg[0, 1, :, :, :] * 1 + fuse_ed_seg[0, 2, :, :, :] * 2 + fuse_ed_seg[0, 3, :, :, :] * 3
    # pt = mm.data.cpu().numpy()
    # # pt = final_seg[0, 0, :, :, :].data.cpu().numpy()
    #
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/fuse-seg-ed' + str(step) + '.nii')
    #
    # mm = es_seg[0, 1, :, :, :] * 1 + es_seg[0, 2, :, :, :] * 2 + es_seg[0, 3, :, :, :] * 3
    # pt = mm.data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # # pt = start_seg[0, 0, :, :, :].data.cpu().numpy()
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
    # sitk.WriteImage(out, './state/seg-es' + str(step) + '.nii')
    #
    # mm = ed_seg[0, 1, :, :, :] * 1 + ed_seg[0, 2, :, :, :] * 2 + ed_seg[0, 3, :, :, :] * 3
    # pt = mm.data.cpu().numpy()
    # # pt = final_seg[0, 0, :, :, :].data.cpu().numpy()
    # # pt = np.transpose(pt, (2, 1, 0))
    # out = sitk.GetImageFromArray(pt)
    # out.SetSpacing((1.25, 1.25, 16))
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
