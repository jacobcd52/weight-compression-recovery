"""ResNet-20 for CIFAR-10 (He et al. 2016, CIFAR variant).

3 stages x 3 BasicBlocks, channels {16, 32, 64}, option-A (parameter-free,
zero-padded) shortcuts. ~270k parameters. Structure follows the widely used
reference implementation (akamaster/pytorch_resnet_cifar10).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # Option A: identity shortcut with zero-padding for the extra channels
            # and stride-2 spatial subsampling. No extra parameters.
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2],
                                (0, 0, 0, 0, planes // 4, planes // 4),
                                "constant", 0))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10):
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = resnet20()
    n = count_params(m)
    print(f"ResNet-20 params: {n:,}")
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print("output shape:", tuple(y.shape))
    assert y.shape == (2, 10)
    # Sanity on the expected param count (~0.27M).
    assert 0.25e6 < n < 0.29e6, f"unexpected param count {n}"
    print("OK")
