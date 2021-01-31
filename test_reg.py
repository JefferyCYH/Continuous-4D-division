import os
import numpy as np
import torch
import torch.utils.data.dataset as Dataset
from skimage.transform import resize
import SimpleITK as sitk
import data_utils as data
import torch.utils.data as Datas
import Network as Network
import math
import torch.nn.functional as F
from torch.autograd import Variable
import metrics as criterion
import print_state
import layer
import print_state_reg
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda:0")
print(torch.__version__)
data = data.test_data
dataloder = Datas.DataLoader(dataset=data,batch_size=1,shuffle=False)

# Segnet =Network.DenseBiasNet(n_channels=1, n_classes=4).to(device)
Flownet = Network.VXm(2).to(device)
##
pretrained_dict = torch.load('./pkl/net_epoch_source-Flow-Network.pkl')
model_dict = Flownet.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
Flownet.load_state_dict(model_dict)

# pretrained_dict = torch.load('./pkl/net_epoch_99-Seg-Network.pkl',map_location='cpu')
# model_dict = Segnet.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# Segnet.load_state_dict(model_dict)

with torch.no_grad():
 for epoch in range(1):
  for step, (source,img_ed,img_es,labeled,labeles) in enumerate(dataloder):
        img_ed = img_ed.to(device).float()
        img_es = img_es.to(device).float()
        labeled = labeled.to(device).float()
        labeles = labeles.to(device).float()
        source=source.to(device).float()
      # mi3, mx3, mi4, mx4 = criterion.location(labeles, labeled, 5, 4)
        depth = source.shape[2]
        flow_field_x1, ed_source, flow_field_x2, es_source = Flownet(img_es, img_ed, source)



        print_state_reg.printst(step, ed_source,es_source,source)
        # mm = seg_source[0, 1, :, :, :] * 1 + seg_source[0, 2, :, :, :] * 2 + seg_source[0, 3, :, :, :] * 3
        # pt = mm.data.cpu().numpy()
        # out = sitk.GetImageFromArray(pt)
        # out.SetSpacing((1.000, 1.000, 1.000))
        # sitk.WriteImage(out, './state/seg-source' + str(step) + '.nii')










