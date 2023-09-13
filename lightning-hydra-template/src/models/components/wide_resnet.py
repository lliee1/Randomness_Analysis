import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        # self.bn1 => layer1,2,3 동일
        self.bn1 = nn.BatchNorm2d(in_planes)

        # self.conv1
        # 1. self.layer1의 첫번째 제외, 두번째, 세번째, 네번째 block 모두 동일한 형태

        # 첫번째
        # => nn.Conv2d(16, 160, kernel_size=3, padding=1, bias=True, (stride=1))

        # 나머지
        # => nn.Conv2d(160, 160, kernel_size=3, padding=1, bias=True, (stride=1))
        # stride의 default는 1

        # 2. self.layer2의 첫번째 제외, 두번째, 세번째, 네번째 block 모두 동일한 형태
        # 첫번째
        # => nn.Conv2d(160, 320, kernel_size=3, padding=1, bias=True, (stride=1))

        # 나머지
        # => nn.Conv2d(320, 320, kernel_size=3, padding=1, bias=True, (stride=1))

        # 3. self.layer2의 첫번째 제외, 두번째, 세번째, 네번째 block 모두 동일한 형태
        # 첫번째
        # => nn.Conv2d(320, 640, kernel_size=3, padding=1, bias=True, (stride=1))

        # 나머지
        # => nn.Conv2d(640, 640, kernel_size=3, padding=1, bias=True, (stride=1)) 
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)


        # dropout
        # => layer1,2,3의 모든 block 동일
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # bn2
        # 1. self.layer1의 모든 block 동일
        # => nn.BatchNorm2d(160)

        # 2. self.layer2 "
        # => nn.BatchNorm2d(320)

        # 3. self.layer3 "
        # => nn.BatchNorm2d(640)
        self.bn2 = nn.BatchNorm2d(planes)


        # self.conv2
        # 1. self.layer1의 모든 block 동일
        # => nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1, bias=True)

        # 2. self.layer2의 첫번째 block만 다르고, 나머지 동일
        # 다른점 : stride
        # => nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1, bias=True)
        # => nn.Conv2d(320, 320, kernel_size=3, stride=1, padding=1, bias=True)

        # 3. self.layer3의 첫번째 block만 다르고, 나머지 동일
        # 다른점 : stride
        # => nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1, bias=True)
        # => nn.Conv2d(640, 640, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        # residual
        self.shortcut = nn.Sequential()

        # stride가 1이 아니면 input과 output의 size가 다를 것
        # 또한 in_planes이랑 planes가 다르면 input과 output의 channel 개수가 다를 것
        # 따라서 shortcut이라는 layer에서 조정해야한 뒤 output에 더해줄 필요가 있음

        # layer 1,2,3의 특정 block에서 in_planes와 planes가 다르기 때문에
        # shortcut을 다음과 같이 정의
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        # conv1을 통해서 각각 layer1,2,3의 첫번째 block에서
        # 16->160
        # 160->320
        # 320->640 으로 channel 증가

        # 나머지 block은 channel 유지 (160,320,640)
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))

        # 이후 batch norm, relu 지난뒤
        # 동일한 channel의 output 만들어내는 conv2 통과
        out = self.conv2(F.relu(self.bn2(out)))

        # output의 size에 맞춘 input을 output과 summation (residual)
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, dataset):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        # 
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear1 = nn.Linear(nStages[3], num_classes)
        self.linear2 = nn.Linear(nStages[3]*4, num_classes)
        self.dataset = Dataset
        
    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        # self.layer1 => [1,1,1,1]
        # self.layer2 => [2,1,1,1]
        # self.layer3 => [2,1,1,1]
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            # self.layer1 => 
            # [wide_basic(self.in_planes=16, nStages[1]=160, dropout_rate=0, stride=1),
            #  wide_basic(self.in_planes=160, nStages[1]=160, dropout_rate=0, stride=1), 
            #  wide_basic(self.in_planes=160, nStages[1]=160, dropout_rate=0, stride=1), 
            #  wide_basic(self.in_planes=160, nStages[1]=160, dropout_rate=0, stride=1) ]

            # self.layer2 =>
            # [wide_basic(self.in_planes=160, nStages[2]=320, dropout_rate=0, stride=2),
            #  wide_basic(self.in_planes=320, nStages[2]=320, dropout_rate=0, stride=1), 
            #  wide_basic(self.in_planes=320, nStages[2]=320, dropout_rate=0, stride=1), 
            #  wide_basic(self.in_planes=320, nStages[2]=320, dropout_rate=0, stride=1) ]

            # self.layer3 =>
            # [wide_basic(self.in_planes=320, nStages[3]=640, dropout_rate=0, stride=2),
            #  wide_basic(self.in_planes=640, nStages[3]=640, dropout_rate=0, stride=1),
            #  wide_basic(self.in_planes=640, nStages[3]=640, dropout_rate=0, stride=1), 
            #  wide_basic(self.in_planes=640, nStages[3]=640, dropout_rate=0, stride=1) ]

            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # 3-> 16으로 channel 증가
        out = self.conv1(x)

        # self.layer1이 가지는 4개의 block 통과
        # 앞에서 봤듯이 각각의 block은, bn1->relu->conv1->dropout->bn2->relu->conv2 구조로 되어있음
        out = self.layer1(out)
        
        # layer2,3은 위와 동일
        # 다만 layer1에서는 input size가 유지되는 반면
        # layer2,3은 첫번째 block에서 stride=2라서 size가 반절로 줄어듦
        out = self.layer2(out)
        out = self.layer3(out)
        # 이렇게 총 12개의 block을 지난 out은
        # bn1->relu를 통과
        # 이때 self.bn1은 위에서 확인할 수 있듯이, momentum 들어있음
        out = F.relu(self.bn1(out))

        # 마지막으로 average pooling layer를 추가
        # 8X8 pixel을 기준으로 average pooling
        # 즉, size가 /8 됨
        out = F.avg_pool2d(out, 8)
        
        # 여기까지 다시 짚어보면
        # 32X32 cifar 10,100 dataset은

        # layer2,3에서 각각 /2, /2 되어 8X8이 됨
        # average pooling 을 통해 /8 되어
        # 최종적으로 (batch_size, 640, 1, 1) 형태의 Tensor

        # linear에 통과하기 위해서 flatten
        # batch는 유지 해야하기 때문에
        # out.view(out.size(0)=batch_size, -1(batch를 제외한 나머지 차원을 알아서 계산))
        # tensor.view(dimension)은
        # tensor를 원하는 size로 reshape하기 위한 method
        out = out.view(out.size(0), -1)
    
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            out = self.linear1(out)
        else:
            out = self.linear2(out)

        return out

if __name__ == '__main__':
    net=Wide_ResNet(28, 10, 0.3, 10, dataset='cifar10')
    print(net)
    a = torch.rand([1,3, 32, 32])
    out = net(a)
