import torch
import torch.nn as nn

class Unet_brain(nn.Module):
    def consecutive_conv(self, in_channels, out_channels, mid_channels = 0):
        if mid_channels == 0:
            mid_channels = out_channels
        return nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True))
    
    def consecutive_conv_f(self, in_channels, out_channels, mid_channels = 0):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.Sigmoid())

    def __init__(self, num_channels=64):
        super(Unet_brain, self).__init__()

        self.conv_initial = self.consecutive_conv(1, num_channels)

        self.conv_rest1 = self.consecutive_conv(num_channels, num_channels * 2)
        self.conv_rest2 = self.consecutive_conv(num_channels * 2, num_channels * 4)
        self.conv_rest3 = self.consecutive_conv(num_channels * 4, num_channels * 8)
        self.conv_bottom = self.consecutive_conv(num_channels * 8, num_channels * 8, num_channels * 16)
        
        self.conv_cls1 = self.consecutive_conv(num_channels * 8, num_channels * 4)
        self.conv_cls2 = self.consecutive_conv(num_channels * 4, num_channels * 2)
        self.dense = nn.Linear(num_channels*2, 1)
        #self.dense_ = nn.Linear(num_channels*2, 3)

        self.conv_up1 = self.consecutive_conv(num_channels * 16, num_channels * 8)
        self.conv_up2 = self.consecutive_conv(num_channels * 8, num_channels * 4)
        self.conv_up3 = self.consecutive_conv(num_channels * 4, num_channels * 2)
        self.conv_up4 = self.consecutive_conv(num_channels * 2, num_channels)

        self.conv_final = self.consecutive_conv_f(num_channels+1, 1)

        self.pooling = nn.MaxPool3d(2, stride=2)

        self.upsample1 = nn.ConvTranspose3d(num_channels*8, num_channels*8, 2, 2)
        self.upsample2 = nn.ConvTranspose3d(num_channels*8, num_channels*4, 2, 2)
        self.upsample3 = nn.ConvTranspose3d(num_channels*4, num_channels*2, 2, 2)
        self.upsample4 = nn.ConvTranspose3d(num_channels*2, num_channels*1, 2, 2)

    def forward(self, x):
        # 2*64*64*64 to 64*32*32*32
        x_64 = self.conv_initial(x)
        x_32 = self.pooling(x_64)

        # 64*32*32*32 to 128*16*16*16
        x_32 = self.conv_rest1(x_32)
        x_16 = self.pooling(x_32)

        # 128*16*16*16 to 256*8*8*8
        x_16 = self.conv_rest2(x_16)
        x_8 = self.pooling(x_16)

        # 256*8*8*8 to 512*4*4*4
        x_8 = self.conv_rest3(x_8)
        x_4 = self.pooling(x_8)

        # 64*16*16*16 to 64*8*8*8
        x_4 = self.conv_bottom(x_4)      

        #classification path
        c_4 = self.conv_cls1(x_4)   
        c_2 = self.pooling(c_4)

        c_2 = self.conv_cls2(c_2)    
        c_1 = self.pooling(c_2)
    
        c_1 = self.dense(torch.flatten(c_1, start_dim=1))
        
        cls = torch.sigmoid(c_1)

        # upsmapling path
        u_8 = self.upsample1(x_4)
        u_8 = self.conv_up1(torch.cat((x_8, u_8), 1))

        u_16 = self.upsample2(u_8)
        u_16 = self.conv_up2(torch.cat((x_16, u_16), 1))

        u_32 = self.upsample3(u_16)
        u_32 = self.conv_up3(torch.cat((x_32, u_32), 1))

        u_64 = self.upsample4(u_32)
        u_64 = self.conv_up4(torch.cat((x_64, u_64), 1))

        cls_to_dec = cls.view(cls.shape[0],1,1,1,cls.shape[1])
        cls_to_dec = cls_to_dec.repeat(1,1,u_64.shape[2],u_64.shape[3],u_64.shape[4])
        u_64 = torch.cat((u_64, cls_to_dec),1)

        seg = self.conv_final(u_64)   
  
        return seg, cls, x_4


class Unet_abdom(nn.Module):
    def consecutive_conv(self, in_channels, out_channels, mid_channels = 0):
        if mid_channels == 0:
            mid_channels = out_channels
        return nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True))

    def consecutive_conv_f(self, in_channels, out_channels, mid_channels = 0):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.Sigmoid())

    def __init__(self, num_channels=64):
        super(Unet_abdom, self).__init__()

        self.conv_initial = self.consecutive_conv(28, num_channels)

        self.conv_rest1 = self.consecutive_conv(num_channels, num_channels * 2)
        self.conv_rest2 = self.consecutive_conv(num_channels * 2, num_channels * 4)
        self.conv_rest3 = self.consecutive_conv(num_channels * 4, num_channels * 8)
        self.conv_bottom = self.consecutive_conv(num_channels * 8, num_channels * 8, num_channels * 16)
        
        self.conv_cls1 = self.consecutive_conv(num_channels * 8, num_channels * 4)
        self.conv_cls2 = self.consecutive_conv(num_channels * 4, num_channels * 2)
        self.dense = nn.Linear(num_channels*2, 1)

        self.conv_up1 = self.consecutive_conv(num_channels * 16, num_channels * 8)
        self.conv_up2 = self.consecutive_conv(num_channels * 8, num_channels * 4)
        self.conv_up3 = self.consecutive_conv(num_channels * 4, num_channels * 2)
        self.conv_up4 = self.consecutive_conv(num_channels * 2, num_channels)

        self.conv_final = self.consecutive_conv_f(num_channels, 1)

        self.pooling = nn.MaxPool3d(2, stride=2)

        self.upsample1 = nn.ConvTranspose3d(num_channels*8, num_channels*8, 2, 2)
        self.upsample2 = nn.ConvTranspose3d(num_channels*8, num_channels*4, 2, 2)
        self.upsample3 = nn.ConvTranspose3d(num_channels*4, num_channels*2, 2, 2)
        self.upsample4 = nn.ConvTranspose3d(num_channels*2, num_channels*1, 2, 2)

    def forward(self, x):
        # 2*64*64*64 to 64*32*32*32
        x_64 = self.conv_initial(x)
        x_32 = self.pooling(x_64)

        # 64*32*32*32 to 128*16*16*16
        x_32 = self.conv_rest1(x_32)
        x_16 = self.pooling(x_32)

        # 128*16*16*16 to 256*8*8*8
        x_16 = self.conv_rest2(x_16)
        x_8 = self.pooling(x_16)

        # 256*8*8*8 to 512*4*4*4
        x_8 = self.conv_rest3(x_8)
        x_4 = self.pooling(x_8)

        # 64*16*16*16 to 64*8*8*8
        x_4 = self.conv_bottom(x_4)      

        # upsmapling path
        u_8 = self.upsample1(x_4)
        u_8 = self.conv_up1(torch.cat((x_8, u_8), 1))

        u_16 = self.upsample2(u_8)
        u_16 = self.conv_up2(torch.cat((x_16, u_16), 1))

        u_32 = self.upsample3(u_16)
        u_32 = self.conv_up3(torch.cat((x_32, u_32), 1))

        u_64 = self.upsample4(u_32)
        u_64 = self.conv_up4(torch.cat((x_64, u_64), 1))

        seg = self.conv_final(u_64)   
        
        #classification path
        c_4 = self.conv_cls1(x_4)   
        c_2 = self.pooling(c_4)

        c_2 = self.conv_cls2(c_2)    
        c_1 = self.pooling(c_2)

        c_1 = self.dense(torch.flatten(c_1, start_dim=1))
        
        cls = torch.sigmoid(c_1)

        return seg, cls, x_4
    
class Unet_abdom_256(nn.Module):
    def consecutive_conv(self, in_channels, out_channels, mid_channels = 0):
        if mid_channels == 0:
            mid_channels = out_channels
        return nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True))
    
    def consecutive_conv_f(self, in_channels, out_channels, mid_channels = 0):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.Sigmoid())

    def __init__(self, num_channels=64):
        super(Unet_abdom_256, self).__init__()

        self.conv_initial = self.consecutive_conv(28, num_channels)

        self.conv_rest1 = self.consecutive_conv(num_channels, num_channels * 2)
        self.conv_rest2 = self.consecutive_conv(num_channels * 2, num_channels * 4)
        self.conv_rest3 = self.consecutive_conv(num_channels * 4, num_channels * 8)
        self.conv_bottom = self.consecutive_conv(num_channels * 8, num_channels * 8, num_channels * 16)
        
        self.conv_cls1 = self.consecutive_conv(num_channels * 8, num_channels * 4)
        self.conv_cls2 = self.consecutive_conv(num_channels * 4, num_channels * 2)
        self.conv_cls3 = self.consecutive_conv(num_channels * 2, num_channels * 1)
        self.dense = nn.Linear(num_channels, 1)

        self.conv_up1 = self.consecutive_conv(num_channels * 16, num_channels * 8)
        self.conv_up2 = self.consecutive_conv(num_channels * 8, num_channels * 4)
        self.conv_up3 = self.consecutive_conv(num_channels * 4, num_channels * 2)
        self.conv_up4 = self.consecutive_conv(num_channels * 2, num_channels)

        self.conv_final = self.consecutive_conv_f(num_channels, 1)

        self.pooling = nn.MaxPool3d(2, stride=2)

        self.upsample1 = nn.ConvTranspose3d(num_channels*8, num_channels*8, 2, 2)
        self.upsample2 = nn.ConvTranspose3d(num_channels*8, num_channels*4, 2, 2)
        self.upsample3 = nn.ConvTranspose3d(num_channels*4, num_channels*2, 2, 2)
        self.upsample4 = nn.ConvTranspose3d(num_channels*2, num_channels*1, 2, 2)

    def forward(self, x):
        # 2*64*64*64 to 64*32*32*32
        x_64 = self.conv_initial(x)
        x_32 = self.pooling(x_64)

        # 64*32*32*32 to 128*16*16*16
        x_32 = self.conv_rest1(x_32)
        x_16 = self.pooling(x_32)

        # 128*16*16*16 to 256*8*8*8
        x_16 = self.conv_rest2(x_16)
        x_8 = self.pooling(x_16)

        # 256*8*8*8 to 512*4*4*4
        x_8 = self.conv_rest3(x_8)
        x_4 = self.pooling(x_8)

        # 64*16*16*16 to 64*8*8*8
        x_4 = self.conv_bottom(x_4)      

        # upsmapling path
        u_8 = self.upsample1(x_4)
        u_8 = self.conv_up1(torch.cat((x_8, u_8), 1))

        u_16 = self.upsample2(u_8)
        u_16 = self.conv_up2(torch.cat((x_16, u_16), 1))

        u_32 = self.upsample3(u_16)
        u_32 = self.conv_up3(torch.cat((x_32, u_32), 1))

        u_64 = self.upsample4(u_32)
        u_64 = self.conv_up4(torch.cat((x_64, u_64), 1))

        seg = self.conv_final(u_64)   
        
        #classification path
        c_4 = self.conv_cls1(x_4)   
        c_2 = self.pooling(c_4)

        c_2 = self.conv_cls2(c_2)    
        c_1_ = self.pooling(c_2)
        
        c_1_ = self.conv_cls3(c_1_)  
        c_1 = self.pooling(c_1_)
    
        c_1 = self.dense(torch.flatten(c_1, start_dim=1))
        
        cls = torch.sigmoid(c_1)

        return seg, cls

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.fc_train1 = nn.Linear(512 * block.expansion, 512 * block.expansion)
        #self.fc_train2 = nn.Linear(512 * block.expansion, 512 * block.expansion)
        self.fc_train3 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_ = self.layer4(x)

        x = self.avgpool(x_)
        x = torch.flatten(x, 1)
        #x = self.fc_train1(x)
        #x = self.fc_train2(x)
        x = self.fc_train3(x)
        #x = self.fc(x)

        return x, x_

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)



def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)