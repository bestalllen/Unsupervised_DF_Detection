import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
import numpy as np
import os
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import init

class XceptionSimCLR(nn.Module):

    def __init__(self):
        super(XceptionSimCLR, self).__init__()
        self.backbone = Xception(num_classes=2048)
        dim_mlp = self.backbone.fc.out_features
        feat_dim = 128
        print("add projection head")
        # add mlp projection head
        self.backbone = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, feat_dim))
        # self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_in, feat_dim))

    # def get_xception(self, pretrained=False,num_classes=512,**kwargs):
    #     model = Xception(**kwargs)
    #     if pretrained:
    #         print("Load pretrained model")
    #         #model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    #         model.load_state_dict(torch.load("/home/xsc/experiment/DFGC/Det_model_training/xception-c0a72b38.pth"))
    #     print("更改输出feature维度")
    #     # model.fc = nn.Linear(2048, num_classes)
    #     return model

    def forward(self, x):
        return self.backbone(x)
       

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=2048):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        # self.fc = nn.Linear(2048, 128)

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        feature = x.view(x.size(0), -1)
        # x = self.fc(feature)

        return feature

def init_weights_(layer):
    if isinstance(layer, nn.Conv2d):
        n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
        layer.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)

class SupConXception(nn.Module):
    """backbone + projection head"""
    def __init__(self, head='mlp', feat_dim=128, encoder_pretrained=True):
        super(SupConXception, self).__init__()
        self.encoder = Xception()
        
        if encoder_pretrained:
            print("Load pretrained model")
            self.encoder.load_state_dict(torch.load("/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/xception-43020ad28.pth"), strict=False)
        
        dim_in = 2048
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        # self.dis_head = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        feat_contrast = F.normalize(self.head(feat), dim=1)
        
        # logits_pred = self.dis_head(feat)
        # logits_pred = torch.sigmoid(logits_pred)

        return feat_contrast

    def init_weights(self):
        self.encoder.apply(init_weights_)
        self.head.apply(init_weights_)
        return True

class SupCEXceptionNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='xception', num_classes=2, encoder_pretrained=False):
        super(SupCEXceptionNet, self).__init__()
        self.encoder = Xception()

        if encoder_pretrained:
            print("Load pretrained model")
            self.encoder.load_state_dict(torch.load("/home/xsc/experiment/Our_DeepfakeUCL/pesudo_label_experiment/xception-43020ad28.pth"), strict=False)
        dim_in = 2048
        # self.encoder.fc = nn.Linear(2048, num_classes)
        # self.dis_head = nn.Linear(dim_in, num_classes)
        self.dis_head = nn.Sequential(
                            nn.Linear(dim_in, 1024),
                            nn.ReLU(inplace=True),
                            nn.Linear(1024, 1024),
                            nn.ReLU(inplace=True),
                            nn.Linear(1024,num_classes))
        #self.top_layer = nn.Linear(1024,num_classes)

        # self.encoder.fc = nn.Linear(2048, num_classes)
        # dim_in = self.encoder.fc.out_features
        # self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        # return self.head(feat)
        # return self.fc(self.encoder(x))
        return feat, self.dis_head(feat)