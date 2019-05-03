import torch
import torch.nn as nn
import numpy as np
import argparse
import collections
import torchnet as tnt

from utils import util

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
parser.add_argument('--parallel', action ='store_true', default=False)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--mode', default='video', help='video|clip')
args = parser.parse_args()

def test():

    net.eval()

    correct, total = 0, 0
    loss_meters = collections.defaultdict(lambda: tnt.meter.AverageValueMeter())
    for idx, batch in enumerate(testloader):

        batch = util.batch_cuda(batch)
        pred, loss_dict = net(batch)

        loss_dict = {k:v.mean() for k,v in loss_dict.items() if v.numel()>0}
        loss = sum(loss_dict.values())

        for k, v in loss_dict.items():
            loss_meters[k].add(v.item())
        loss_meters['total_loss'].add(loss.item())

        _, pred_idx = pred.max(1)
        correct += (pred_idx==batch['label']).float().sum()
        total += pred.size(0)

        print ('%d/%d.. L: %.3f A: %.3f'%(idx, len(testloader), loss_meters['total_loss'].value()[0], correct/total))

    accuracy = 1.0*correct/total
    log_str = '(test) A: %.3f | '%(accuracy)
    log_str += ' | '.join(['%s: %.3f'%(k, v.value()[0]) for k,v in loss_meters.items()])
    print (log_str)


#----------------------------------------------------------------------------------------------------------------------------------------#
from data import kinetics
from models import resnet

if args.mode == 'video':
    testset = kinetics.KineticsMultiCrop(root='data/kinetics/', split='val', clip_len=32)
elif args.mode == 'clip':
    testset = kinetics.Kinetics(root='data/kinetics/', split='val', clip_len=32)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

net = resnet.i3_res50(num_classes=len(testset.labels))
net.cuda()

if args.parallel:
    net = nn.DataParallel(net)

with torch.no_grad():
    test()
