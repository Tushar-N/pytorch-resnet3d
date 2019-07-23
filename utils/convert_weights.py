import pickle
import torch
import re
import sys

c2_weights = sys.argv[1] # 'pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl'
pth_weights_out = sys.argv[2] # 'pretrained/i3d_r50_kinetics.pth'

c2 = pickle.load(open(c2_weights, 'rb'), encoding='latin')['blobs']
c2 = {k:v for k,v in c2.items() if 'momentum' not in k}

downsample_pat = re.compile('res(.)_(.)_branch1_.*')
conv_pat = re.compile('res(.)_(.)_branch2(.)_.*')
nl_pat = re.compile('nonlocal_conv(.)_(.)_(.*)_.*')

m2num = dict(zip('abc',[1,2,3]))
suffix_dict = {'b':'bias', 'w':'weight', 's':'weight', 'rm':'running_mean', 'riv':'running_var'}

key_map = {}
key_map.update({'conv1.weight':'conv1_w',
			'bn1.weight':'res_conv1_bn_s',
			'bn1.bias':'res_conv1_bn_b',
			'bn1.running_mean':'res_conv1_bn_rm',
			'bn1.running_var':'res_conv1_bn_riv',
			'fc.weight':'pred_w',
			'fc.bias':'pred_b',
			})

for key in c2:

	conv_match = conv_pat.match(key)
	if conv_match:
		layer, block, module = conv_match.groups()
		layer, block, module = int(layer), int(block), m2num[module]
		name = 'bn' if 'bn_' in key else 'conv'
		suffix = suffix_dict[key.split('_')[-1]]
		new_key = 'layer%d.%d.%s%d.%s'%(layer-1, block, name, module, suffix)
		key_map[new_key] = key

	ds_match = downsample_pat.match(key)
	if ds_match:
		layer, block = ds_match.groups()
		layer, block = int(layer), int(block)
		module = 0 if key[-1]=='w' else 1
		name = 'downsample'
		suffix = suffix_dict[key.split('_')[-1]]
		new_key = 'layer%d.%d.%s.%d.%s'%(layer-1, block, name, module, suffix)
		key_map[new_key] = key

	nl_match = nl_pat.match(key)
	if nl_match:
		layer, block, module = nl_match.groups()
		layer, block = int(layer), int(block)
		name = 'nl.%s'%module
		suffix = suffix_dict[key.split('_')[-1]]
		new_key = 'layer%d.%d.%s.%s'%(layer-1, block, name, suffix)
		key_map[new_key] = key

from models import resnet
pth = resnet.I3Res50(num_classes=400, use_nl=True)
state_dict = pth.state_dict()

new_state_dict = {key: torch.from_numpy(c2[key_map[key]]) for key in state_dict if key in key_map}
torch.save(new_state_dict, pth_weights_out)
torch.save(key_map, pth_weights_out+'.keymap')

# check if weight dimensions match
for key in state_dict:

	if key not in key_map:
		continue

	c2_v, pth_v = c2[key_map[key]], state_dict[key]
	assert str(tuple(c2_v.shape))==str(tuple(pth_v.shape)), 'Size Mismatch'
	print ('{:23s} --> {:35s} | {:21s}'.format(key_map[key], key, str(tuple(c2_v.shape))))