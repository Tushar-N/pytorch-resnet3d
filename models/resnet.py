import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FrozenBN(nn.Module):
    def __init__(self, num_channels, momentum=0.1, eps=1e-5):
        super(FrozenBN, self).__init__()
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps
        self.params_set = False

    def set_params(self, scale, bias, running_mean, running_var):
        self.register_buffer('scale', scale)
        self.register_buffer('bias', bias)
        self.register_buffer('running_mean', running_mean)
        self.register_buffer('running_var', running_var)
        self.params_set = True

    def forward(self, x):
        assert self.params_set, 'model.set_params(...) must be called before the forward pass'
        return torch.batch_norm(x, self.scale, self.bias, self.running_mean, self.running_var, False, self.momentum, self.eps, torch.backends.cudnn.enabled)

    def __repr__(self):
        return 'FrozenBN(%d)'%self.num_channels

def freeze_bn(m, name):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.BatchNorm3d:
            frozen_bn = FrozenBN(target_attr.num_features, target_attr.momentum, target_attr.eps)
            frozen_bn.set_params(target_attr.weight.data, target_attr.bias.data, target_attr.running_mean, target_attr.running_var)
            setattr(m, attr_str, frozen_bn)
    for n, ch in m.named_children():
        freeze_bn(ch, n)
        
#-----------------------------------------------------------------------------------------------#

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride, use_nl=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1), stride=(temp_stride, 1, 1), padding=(temp_conv, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        outplanes = planes * 4
        self.nl = NonLocalBlock(outplanes, outplanes, outplanes//2) if use_nl else None


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.nl is not None:
            out = self.nl(out)

        return out

class NonLocalBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner):
        super(NonLocalBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = dim_inner  
        self.dim_out = dim_out

        self.theta = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.maxpool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))
        self.phi = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.g = nn.Conv3d(dim_in, dim_inner, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

        self.out = nn.Conv3d(dim_inner, dim_out, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.bn = nn.BatchNorm3d(dim_out)

    def forward(self, x):
        residual = x

        batch_size = x.shape[0]
        mp = self.maxpool(x)
        theta = self.theta(x)
        phi = self.phi(mp)
        g = self.g(mp)

        theta_shape_5d = theta.shape
        theta, phi, g = theta.view(batch_size, self.dim_inner, -1), phi.view(batch_size, self.dim_inner, -1), g.view(batch_size, self.dim_inner, -1)
      
        theta_phi = torch.bmm(theta.transpose(1, 2), phi) # (8, 1024, 784) * (8, 1024, 784) => (8, 784, 784)
        theta_phi_sc = theta_phi * (self.dim_inner**-.5)
        p = F.softmax(theta_phi_sc, dim=-1)

        t = torch.bmm(g, p.transpose(1, 2))
        t = t.view(theta_shape_5d)

        out = self.out(t)
        out = self.bn(out)

        out = out + residual
        return out


#-----------------------------------------------------------------------------------------------#

class I3Res50(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=400, use_nl=False):
        self.inplanes = 64
        super(I3Res50, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 0, 0))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        nonlocal_mod = 2 if use_nl else 1000
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, temp_conv=[1, 1, 1], temp_stride=[1, 1, 1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, temp_conv=[1, 0, 1, 0], temp_stride=[1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, temp_conv=[1, 0, 1, 0, 1, 0], temp_stride=[1, 1, 1, 1, 1, 1], nonlocal_mod=nonlocal_mod)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, temp_conv=[0, 1, 0], temp_stride=[1, 1, 1])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.drop = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, temp_conv, temp_stride, nonlocal_mod=1000):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or temp_stride[0]!=1:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=(temp_stride[0], stride, stride), padding=(0, 0, 0), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp_conv[0], temp_stride[0], False))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, temp_conv[i], temp_stride[i], i%nonlocal_mod==nonlocal_mod-1))

        return nn.Sequential(*layers)

    def forward_single(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.drop(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def forward_multi(self, x):
        clip_preds = []
        for clip_idx in range(x.shape[1]): # B, 10, 3, 3, 32, 224, 224
            spatial_crops = []
            for crop_idx in range(x.shape[2]): 
                clip = x[:, clip_idx, crop_idx]
                clip = self.forward_single(clip)
                spatial_crops.append(clip)
            spatial_crops = torch.stack(spatial_crops, 1).mean(1) # (B, 400)
            clip_preds.append(spatial_crops)
        clip_preds = torch.stack(clip_preds, 1).mean(1) # (B, 400)
        return clip_preds

    def forward(self, batch):

        # 5D tensor == single clip
        if batch['frames'].dim() == 5:
            pred = self.forward_single(batch['frames'])

        # 7D tensor == 3 crops/10 clips
        elif batch['frames'].dim() == 7:
            pred = self.forward_multi(batch['frames'])

        loss_dict = {}
        if 'label' in batch:
            loss = F.cross_entropy(pred, batch['label'], reduction='none')
            loss_dict = {'loss': loss}
            
        return pred, loss_dict

#-----------------------------------------------------------------------------------------------#

def i3_res50(num_classes):
    net = I3Res50(num_classes=num_classes, use_nl=False)
    state_dict = torch.load('pretrained/i3d_r50_kinetics.pth')
    net.load_state_dict(state_dict)
    # freeze_bn(net, "net") # Only needed for finetuning. For validation, .eval() works.
    return net

def i3_res50_nl(num_classes):
    net = I3Res50(num_classes=num_classes, use_nl=True)
    state_dict = torch.load('pretrained/i3d_r50_nl_kinetics.pth')
    net.load_state_dict(state_dict)
    # freeze_bn(net, "net") # Only needed for finetuning. For validation, .eval() works.
    return net

if __name__=='__main__':
    net = i3_res50_nl(400)
    inp = {'frames': torch.rand(4, 3, 32, 224, 224)}
    pred, losses = net(inp)
