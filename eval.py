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
parser.add_argument('--model', default='r50_nl', help='r50|r50_nl')
args = parser.parse_args()

def test():

    net.eval()

    topk = [1, 5]
    loss_meters = collections.defaultdict(lambda: tnt.meter.AverageValueMeter())
    for idx, batch in enumerate(testloader):

        batch = util.batch_cuda(batch)
        pred, loss_dict = net(batch)

        loss_dict = {k:v.mean() for k,v in loss_dict.items() if v.numel()>0}
        loss = sum(loss_dict.values())

        for k, v in loss_dict.items():
            loss_meters[k].add(v.item())

        prec_scores = util.accuracy(pred, batch['label'], topk=topk)
        for k, prec in zip(topk, prec_scores):
            loss_meters['P%s'%k].add(prec.item(), pred.shape[0])

        stats = ' | '.join(['%s: %.3f'%(k, v.value()[0]) for k,v in loss_meters.items()])
        print ('%d/%d.. %s'%(idx, len(testloader), stats))

    print ('(test) %s'%stats)

#----------------------------------------------------------------------------------------------------------------------------------------#
from data import kinetics
from models import resnet

if args.mode == 'video':
    testset = kinetics.KineticsMultiCrop(root='data/kinetics/', split='val', clip_len=32)
elif args.mode == 'clip':
    testset = kinetics.Kinetics(root='data/kinetics/', split='val', clip_len=32)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

if args.model=='r50':
    net = resnet.i3_res50(num_classes=len(testset.labels))
elif args.model=='r50_nl':
    net = resnet.i3_res50_nl(num_classes=len(testset.labels))
net.cuda()

if args.parallel:
    net = nn.DataParallel(net)

with torch.no_grad():
    test()
