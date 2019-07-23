# A script to check layer by layer activation differences between the Caffe2 and Pytorch models.

import numpy as np
import pickle
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='r50_nl', help='r50|r50_nl')
args = parser.parse_args()

#-----------------------------------------------------------------------------------------------#

# Generate a random input. Normalize for fun.
np.random.seed(123)
data = np.random.rand(4, 3, 32, 224, 224).astype(np.float32)*255
data = (data-114.75)/57.375

#-----------------------------------------------------------------------------------------------#

from caffe2.python import workspace
from models import model_builder_video, resnet_video_org

workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
workspace.ResetWorkspace()

c2_net = model_builder_video.ModelBuilder(
        name='test', train=False,
        use_cudnn=False, cudnn_exhaustive_search=False,
        split='val')

c2_net.net.Proto().type = 'dag'

workspace.CreateBlob('data')
workspace.CreateBlob('labels')

c2_net, out_blob = resnet_video_org.create_model(model=c2_net, data='data', labels='labels', split='val', use_nl=args.model=='r50_nl')

workspace.RunNetOnce(c2_net.param_init_net)
workspace.CreateNet(c2_net.net)

# load pretrained weights
if args.model=='r50':
    wt_file = 'pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl'
elif args.model=='r50_nl':
    wt_file = 'pretrained/i3d_nonlocal_32x2_IN_pretrain_400k.pkl'
wts = pickle.load(open(wt_file, 'rb'), encoding='latin')['blobs']

for key in wts:
    if type(wts[key]) == np.ndarray:
        workspace.FeedBlob(key, wts[key])

workspace.FeedBlob('data', data)
workspace.RunNet(c2_net.net.Proto().name)

c2_blobs = {key: workspace.FetchBlob(key) for key in workspace.Blobs()}

#-----------------------------------------------------------------------------------------------#
torch.backends.cudnn.enabled = False
from models import resnet

data = torch.from_numpy(data).cuda()

# load pretrained weights
if args.model=='r50':
    pth_net = resnet.i3_res50(num_classes=400)
    key_map = torch.load('pretrained/i3d_r50_kinetics.pth.keymap')
elif args.model=='r50_nl':
    pth_net = resnet.i3_res50_nl(num_classes=400)
    key_map = torch.load('pretrained/i3d_r50_nl_kinetics.pth.keymap')
key_map = {'.'.join(k.split('.')[:-1]): '_'.join(v.split('_')[:-1]) for k, v in key_map.items()}
pth_net.cuda().eval()
    
def hook(module, input, output):
    setattr(module, "_value_hook", output)

for name, module in pth_net.named_modules():
    module.register_forward_hook(hook)

pth_net({'frames':data})

pth_blobs = {}
for name, module in pth_net.named_modules():
    try:
        if len(name)>0:
            activation = module._value_hook.cpu().detach().numpy()
            pth_blobs[name] = activation
    except:
        pass

for key in sorted(key_map):

    pth_v = pth_blobs[key]
    c2_v = c2_blobs[key_map[key]]

    # For each activation value, print the max/min/mean abs difference
    # Most of these are <1e-6
    delta = np.abs(pth_v-c2_v)
    print (key, np.max(delta), np.min(delta), np.mean(delta))