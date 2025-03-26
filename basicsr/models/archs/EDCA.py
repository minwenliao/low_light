# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # x和y是两个特征图，它们的形状都是(batch_size, channels, height, width)
# x = torch.randn(1, 256, 32, 32)
# y = torch.randn(1, 256, 32, 32)
#
# # 定义两个1x1卷积层
# conv1x1_x = nn.Conv2d(256, 256, 1)
# conv1x1_y = nn.Conv2d(256, 256, 1)
#
# # 通过卷积层得到新的特征图
# x_conv = conv1x1_x(x)
# y_conv = conv1x1_y(y)
#
# # 进行全局平均池化
# x_avgpool = F.adaptive_avg_pool2d(x_conv, (1, 1)).view(1, -1)
# y_avgpool = F.adaptive_avg_pool2d(y_conv, (1, 1)).view(1, -1)
#
# # 定义两个全连接层
# fc_x = nn.Linear(256, 256)
# fc_y = nn.Linear(256, 256)
#
# # 通过全连接层得到新的向量
# x_fc = fc_x(x_avgpool)
# y_fc = fc_y(y_avgpool)
#
# # 使用Sigmoid激活函数
# x_sigmoid = torch.sigmoid(x_fc).unsqueeze(-1).unsqueeze(-1)
# y_sigmoid = torch.sigmoid(y_fc).unsqueeze(-1).unsqueeze(-1)
#
# # 使用得到的向量和原始特征图进行注意力操作
# x_attention = x * y_sigmoid
# y_attention = y * x_sigmoid
#
# # 拼接两个特征图
# concatenated = torch.cat((x_attention, y_attention), dim=1)  # shape: (1, 512, 32, 32)
#
# # 使用一个1x1卷积来降维
# conv1x1_concat = nn.Conv2d(512, 256, 1)
# output = conv1x1_concat(concatenated)
#
# print(output.shape)
#
#











# 在这个类中，我们首先在构造函数__init__中定义了所有的需要的层。然后，在forward函数中，我们定义了如何通过这些层进行前向传播。在这个例子中，我们的模型接收两个输入x和y，并返回一个输出。

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossAttention, self).__init__()

        self.conv1x1_x = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1x1_y = nn.Conv2d(in_channels, out_channels, 1)

        self.fc_x = nn.Linear(out_channels, out_channels)
        self.fc_y = nn.Linear(out_channels, out_channels)

        self.conv1x1_concat = nn.Conv2d(2*out_channels, out_channels, 1)

    def forward(self, x, y):
        x_conv = self.conv1x1_x(x)
        y_conv = self.conv1x1_y(y)

        x_avgpool = F.adaptive_avg_pool2d(x_conv, (1, 1)).view(x.size(0), -1)
        y_avgpool = F.adaptive_avg_pool2d(y_conv, (1, 1)).view(y.size(0), -1)

        x_fc = self.fc_x(x_avgpool)
        y_fc = self.fc_y(y_avgpool)

        x_sigmoid = torch.sigmoid(x_fc).view(x.size(0), -1, 1, 1)
        y_sigmoid = torch.sigmoid(y_fc).view(y.size(0), -1, 1, 1)

        x_attention = x * y_sigmoid
        y_attention = y * x_sigmoid

        concatenated = torch.cat((x_attention, y_attention), dim=1)  
        output = self.conv1x1_concat(concatenated)

        return output

# Usage
model = CrossAttention(256, 256)
x = torch.randn(1, 256, 32, 32)
y = torch.randn(1, 256, 32, 32)
output = model(x, y)
print(output.shape)



import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()

        self.conv1x1_x = nn.Conv2d(0, 0, 1)  # 初始化卷积层，后续会更新权重
        self.conv1x1_y = nn.Conv2d(0, 0, 1)  # 初始化卷积层，后续会更新权重

        self.fc_x = nn.Linear(0, 0)  # 初始化全连接层，后续会更新权重
        self.fc_y = nn.Linear(0, 0)  # 初始化全连接层，后续会更新权重

        self.conv1x1_concat = nn.Conv2d(0, 0, 1)  # 初始化卷积层，后续会更新权重

    def forward(self, x, y):
        channels = x.size(1)  # 获取输入的通道数

        # 动态创建网络层，根据输入通道数更新权重
        if not isinstance(self.conv1x1_x, nn.Conv2d):
            self.conv1x1_x = nn.Conv2d(channels, channels, 1)
            self.conv1x1_y = nn.Conv2d(channels, channels, 1)
            self.fc_x = nn.Linear(channels, channels)
            self.fc_y = nn.Linear(channels, channels)
            self.conv1x1_concat = nn.Conv2d(2*channels, channels, 1)

        x_conv = self.conv1x1_x(x)
        y_conv = self.conv1x1_y(y)

        x_avgpool = F.adaptive_avg_pool2d(x_conv, (1, 1)).view(x.size(0), -1)
        y_avgpool = F.adaptive_avg_pool2d(y_conv, (1, 1)).view(y.size(0), -1)

        x_fc = self.fc_x(x_avgpool)
        y_fc = self.fc_y(y_avgpool)

        x_sigmoid = torch.sigmoid(x_fc).view(x.size(0), -1, 1, 1)
        y_sigmoid = torch.sigmoid(y_fc).view(y.size(0), -1, 1, 1)

        x_attention = x * y_sigmoid
        y_attention = y * x_sigmoid

        concatenated = torch.cat((x_attention, y_attention), dim=1)
        output = self.conv1x1_concat(concatenated)

        return output

# Usage
model = CrossAttention()
x = torch.randn(1, 256, 32, 32)
y = torch.randn(1, 256, 32, 32)
output = model(x, y)
print(output.shape)