import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

from thop.profile import profile
from thop import clever_format
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        atten = torch.cat([avg_out, max_out], dim=1)
        atten = self.conv1(atten)  # 计算得到的注意力
        atten = self.sigmoid(atten)  # 将输入矩阵乘以对应的注意力
        return x * atten  # 将输入矩阵乘以对应的注意力

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class MultiRes1D(nn.Module):
    def __init__(self):
        super(MultiRes1D, self).__init__()
        self.Res1 = nn.Sequential(
                    nn.Conv1d(1, 128, 11, 1),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 11, 1),
                    nn.BatchNorm1d(128),
                    nn.MaxPool1d(150, 150),
                    )

        self.Res12 = nn.Sequential(
                    nn.Conv1d(1, 128, 51, 5),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 11, 1),
                    nn.BatchNorm1d(128),
                    nn.MaxPool1d(30, 30)
                    )


        self.Res13 = nn.Sequential(
                    nn.Conv1d(1, 128, 101, 10),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 11, 1, padding=3),
                    nn.BatchNorm1d(128),
                    nn.MaxPool1d(15, 15)
                    )



        self.Res2 = nn.Sequential(
                    nn.Conv1d(128, 128, 3, 1, 1),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(),
                    nn.Conv1d(128, 128, 3, 1, 1),
                    nn.BatchNorm1d(128)
                    )

        self.maxpool2 = nn.MaxPool1d(3, 3)

        self.Res3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.BatchNorm1d(256)
        )
        self.conv3 = nn.Conv1d(128, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool1d(3, 3)


        self.Res4 = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.BatchNorm1d(256)
        )

        self.maxpool4 = nn.MaxPool1d(3, 3)

        self.Res5 = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.BatchNorm1d(256)
        )

        self.maxpool5 = nn.MaxPool1d(3, 3)

        self.Res6 = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 1, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, 1, 1, 1),
            nn.BatchNorm1d(256)
        )

        self.maxpool6 = nn.MaxPool1d(3, 3)

        self.Res7 = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 1, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, 1, 1, 1),
            nn.BatchNorm1d(256)
        )

        self.maxpool7 = nn.MaxPool1d(3, 3)

        self.Res8 = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 1, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, 1, 1, 1),
            nn.BatchNorm1d(256)
        )
        self.maxpool8 = nn.MaxPool1d(3, 3)

        self.conv4 = nn.Sequential(nn.Conv1d(256, 256, 1, 1),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU())

        self.atten1 = ChannelAttention(3)
        self.atten2 = SpatialAttention(7)

        self.Linear = nn.Linear(256, 10)

        self.Leakrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten(1)
        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        print(x.shape)
        x1 = self.Res1(x)
        # print(x1.shape)
        x2 = self.Res12(x)
        # print(x2.shape)
        x3 = self.Res13(x)
        # print(x3.shape)
        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        x3 = torch.unsqueeze(x3, 1)
        x11 = torch.cat((x1, x2, x3), 1)
        # print(x.shape)
        x = self.atten1(x11) * x11
        x = self.atten2(x)
        # print(x.shape)
        x = torch.transpose(x, 1, 2)
        x = torch.flatten(x, 2)
        # print(x.shape)
        x = self.Res2(x) + x
        x = self.maxpool2(x)
        x = self.Res3(x) + self.conv3(x)
        x = self.maxpool3(x)
        x = self.Res4(x) + x
        x = self.maxpool4(x)
        x = self.Res5(x) + x
        x = self.maxpool5(x)
        x = self.Res6(x) + x
        x = self.maxpool6(x)
        x = self.Res7(x) + x
        x = self.maxpool7(x)
        x = self.Res8(x) + x
        x = self.maxpool8(x)
        x = self.drop1(x)
        # print("x = ",x)
        #5-5新增
        x = self.conv4(x)
        # x1 = self.drop(x1)
        # print("x1=",x1)


        x = self.flatten(x)
        x = self.Linear(x)

        return x

# net = MultiRes1D()
# # net = FusionNet(input_nc=6, output_nc=2)
# model_name = 'FusionNet'
# flops, params = get_model_complexity_info(net, (1, 176400), as_strings=True, print_per_layer_stat=True)
# print("%s |FLOPs: %s |params: %s" % (model_name, flops, params))
#
# input_size = (1, 1,  176400)
# inputs = torch.randn(input_size)
# macs, params = profile(net, (inputs,), verbose=False)  # ,verbose=False
# macs = clever_format([macs], "%.3f")
# print("MACs", macs)
