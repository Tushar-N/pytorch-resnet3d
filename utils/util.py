import torch
import torchvision.transforms as transforms
from PIL import Image

# NOTE: Single channel mean/stev (unlike pytorch Imagenet)
def kinetics_mean_std():
    mean = [114.75, 114.75, 114.75]
    std = [57.375, 57.375, 57.375]
    return mean, std

# Apply .cuda() to every element in the batch
def batch_cuda(batch):
    _batch = {}
    for k,v in batch.items():
        if type(v)==torch.Tensor:
            v = v.cuda()
        elif type(v)==list and type(v[0])==torch.Tensor:
            v = [v.cuda() for v in v]
        _batch.update({k:v})

    return _batch

import utils.gtransforms as gtransforms
def clip_transform(split, max_len):

    mean, std = kinetics_mean_std()
    if split=='train':
        transform = transforms.Compose([
                        gtransforms.GroupResize(256),
                        gtransforms.GroupRandomCrop(224),
                        gtransforms.GroupRandomHorizontalFlip(),
                        gtransforms.ToTensor(),
                        gtransforms.GroupNormalize(mean, std),
                        gtransforms.LoopPad(max_len),
                    ])

    elif split=='val':
        transform = transforms.Compose([
                        gtransforms.GroupResize(256),
                        gtransforms.GroupCenterCrop(256),
                        gtransforms.ToTensor(),
                        gtransforms.GroupNormalize(mean, std),
                        gtransforms.LoopPad(max_len),
            ])

    # Note: RandomCrop (instead of CenterCrop) because
    # We're doing 3 random crops per frame for validation
    elif split=='3crop':
        transform = transforms.Compose([
                gtransforms.GroupResize(256),
                gtransforms.GroupRandomCrop(256),
                gtransforms.ToTensor(),
                gtransforms.GroupNormalize(mean, std),
                gtransforms.LoopPad(max_len),
            ])

    return transform


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k)
    return res