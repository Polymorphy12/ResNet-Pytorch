from typing import List, Optional

import torch
from torch import nn



# Residual Block과 Shortcut(Skip) connection간의 차원이 맞지 않을때 사용하는 클래스.
# Shortcut Connection에 Linear Projection을 적용하여 Residual Block output과 차원을 맞춰준다.

# 개인적인 질문 : projection layer인데, 왜 Linear layer을 쓰지 않고 convolutional layer을 쓰지?
# 질문을 조금 바꿔서 : convolutional layer가 Linear projection을 수행할 수 있을까?
# 이에 대한 답변 : https://machinelearningmastery.com/introduction-to-1x1-convolutions-to-reduce-the-complexity-of-convolutional-neural-networks/
class ShortcutProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(ShortcutProjection, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(stride,stride))
        # 논문에서는 conv layer을 거친 후에 batch norm을 할 것을 권장한다.
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


'''
계획 :
먼저 residual block 그림을 보고 구현해본다.
그리고 labml 구현체와 비교하고 내가 어떤 부분을 놓쳤는지 확인한다.

1) shortcut connection을 구현하지 않았다.
2) conv2d를 써줄 때 padding을 감안하지 않았다.
-> 어떤 경우에 padding을 넣고 어떤 경우에 padding을 넣지 않는지 모호하게 느껴진다.
-> ResNet 아키텍쳐 사진을 보면 옆에 각 layer마다 내놓을 output size가 나와 있다.
   이것을 convolutional layer output 공식에 적용하면 어떤 패딩을 쓸지 알 수 있다.
'''
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(ResidualBlock, self).__init__()

        # stride가 2일 경우, output size가 반으로 줄어들게 된다.
        # 그리고 input과 output 간의 차원이 차이나게 된다.
        # 이는 나중에 Shortcut에다 linear projection을 적용할지 말지 결정하는데 영향을 준다.
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=(stride,stride), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        # conv1의 output size, out_channels와 같아야 하므로
        # conv1 때 stride가 2이더라도 conv2에서는 stride를 1로 설정해준다.
        # 또, in_channels와 out_channels는 conv1의 out_channels와 같게 설정해준다.
        self.conv2 = nn.Conv2d(in_channels= out_channels, out_channels=out_channels, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 조건문이 참일 경우, shortcut에 Linear Projection을 적용해준다.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels,out_channels,stride)
        else:
            self.shortcut = nn.Identity()

        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.relu2(x + shortcut)


'''
여기서 생기는 질문 :
1x1 convolution에 stride를 2 이상 적용시키면 손실되는 픽셀이 생기지 않나?
손실된다 함은, feature을 뽑아내는데 전혀 사용되지 않는 픽셀이 생긴다는 뜻이다.
그걸 감안하고 하는 것인가?

뇌피셜 :
maxpooling 처럼 손실되는 부분이 있어도 그걸 감안한다.

'''
class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):
        super(BottleneckResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=bottleneck_channels, kernel_size=(1,1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=bottleneck_channels, out_channels=bottleneck_channels, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=bottleneck_channels, out_channels=out_channels, kernel_size=(1,1))
        self.bn3 = nn.BatchNorm2d(in_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels,out_channels,stride)
        else:
            self.shortcut = nn.Identity()

        self.relu3 = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        return self.relu3(x + shortcut)


'''
config 형태로 여러 정보들을 넘겨 신경망을 구축한다.

n_blocks : 각 feature map size에 해당하는 block들의 수를 담는 리스트.
n_channels : 각 feature map size에 해당하는 block들의 채널 수를 담는 리스트.
bottlenecks : 각 bottleneck이 가지는 채널 수를 담는 리스트. 
              이게 None이면 일반 ResidualBlock을 사용하고, 아니면 BottleneckResidualBlock을 쓴다.
img_channels : input의 채널 수에 해당하는 변수
first_kernel_size : 첫 conv layer의 kernel size에 해당하는 변수
'''
class ResNetBase(nn.Module):
    def __init__(
            self,
            n_blocks: List[int],
            n_channels: List[int],
            bottlenecks: Optional[List[int]] = None,
            img_channels: int=3,
            first_kernel_size: int = 7
    ):
        super(ResNetBase, self).__init__()

        assert len(n_blocks) == len(n_channels)
        assert bottlenecks is None or len(bottlenecks) == len(n_channels)

        self.conv = nn.Conv2d(img_channels, n_channels[0], kernel_size=(first_kernel_size, first_kernel_size), stride = (2,2), padding= first_kernel_size // 2)
        self.bn = nn.BatchNorm2d(n_channels[0])

        blocks = []

        prev_channels = n_channels[0]

        for i, channels in enumerate(n_channels):
            stride = 2 if len(blocks)==0 else 1

            if bottlenecks is None:
                blocks.append(ResidualBlock(prev_channels, channels, stride=stride))
            else:
                blocks.append(BottleneckResidualBlock(prev_channels, bottlenecks[i],channels, stride=stride))

            prev_channels = channels

            for _ in range(n_blocks[i] - 1):
                if bottlenecks is None:
                    blocks.append(ResidualBlock(channels,channels, stride=1))
                else:
                    blocks.append(BottleneckResidualBlock(channels, bottlenecks[i], channels, stride=1))

        self.blocks = nn.Sequential(*blocks)

        self.linear = nn.Linear(n_channels[-1], 1000)

    def forward(self, x: torch.Tensor):

        x = self.bn(self.conv(x))
        x = self.blocks(x)

        x = x.view(x.shape[0], x.shape[1], -1)

        x = x.mean(dim=-1)

        return self.linear(x)


def test():
    n_blocks: List[int] = [3,4,6,3]
    n_channels: List[int] = [64, 128, 256, 512]

    bottlenecks: Optional[List[int]] = None

    first_kernel_size: int=7

    net = ResNetBase(n_blocks, n_channels, bottlenecks=bottlenecks, first_kernel_size=first_kernel_size)

    x = torch.randn(2, 3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)

test()