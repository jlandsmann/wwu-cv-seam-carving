# Made with YouTube Tutorial https://www.youtube.com/watch?v=DkNIBBBvcPs
import torch
import torch.nn as nn


class Block(nn.Module):
    """
    For ResNet50 and above each residual block has 3 inner layers
    """
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        # value of x is stored in identity for the skip connection
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # apply the skip connection
        x += identity
        x = self.relu(x)
        return x


def get_name():
    return "Resnet"


class ResNet(nn.Module):
    """
    ResNet50 implementation
    """
    def __init__(self, block, layers, image_channels, num_classes):
        #layers is a list of how often we use the block in one layer (e.g. [3,4,6,3] for ResNet50)

        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # A ResNet has 4 layers with different amounts of residual blocks in it
        # the amount of blocks is stored in the array layers
        self.layer1 = self.make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self.make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self.make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self.make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
        self.name="Resnet50"

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(out_channels * 4))
            layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
            self.in_channels = out_channels * 4  # 64*4 = 256

            for i in range(num_residual_blocks - 1):
                # in_channels is 256 and through convolutions it is going to 64. But because out_channels is
                # multiplied by 4 out_channels is also 256 and therefore we dont need the identity_downsample stride
                # is default = 1
                layers.append(block(self.in_channels, out_channels))

            return nn.Sequential(*layers)


def res_net_50(img_channels=3, num_classes=1000):
    """
    get resnet50 model
    Args:
        img_channels (int): input channel first layer
        num_classes (int): number of classes for classification
    Returns:
        resnet50 model

    """
    model=ResNet(Block, [3, 4, 6, 3], img_channels, num_classes)
    model.name = 'ResNet50'
    return model


