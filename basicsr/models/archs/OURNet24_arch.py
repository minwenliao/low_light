import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.SwinT import SwinT
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def default_conv_stride2(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,stride=2,
        padding=(kernel_size//2), bias=bias)

class GateWeight(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  
        return x1 * x2


# Expert1---------专家1
class Expert1(nn.Module):
    def __init__(self, n_feats):
        super(Expert1, self).__init__()
        f = n_feats // 4 
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1) 
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0) 
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1) 
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1) 
        self.sigmoid = nn.Sigmoid()  
        self.relu = nn.ReLU(inplace=True) 

    def forward(self, x):  
        c1_ = self.conv1(x)  
        c1 = self.conv2(c1_)  
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)  
        v_range = self.relu(self.conv_max(v_max))  
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)  
  
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_) 
        c4 = self.conv4(c3 + cf) 
        m = self.sigmoid(c4)  

        return x * m  

############################################################################################
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y  # 返回输入x与通道注意力加权后的结果
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True) 
        max_out, _ = torch.max(x, dim=1, keepdim=True) 
        x = torch.cat([avg_out, max_out], dim=1) 
        x = self.conv1(x) 
        y = self.sigmoid(x)  
        return y * res 

class Expert2(nn.Module):
    def __init__(self, n_feats):
        super(Expert2, self).__init__()
        self.c1 = default_conv(n_feats, n_feats, 1)
        self.c2 = default_conv(n_feats, n_feats // 2, 3)
        self.c3 = default_conv(n_feats, n_feats // 2, 3)
        self.c4 = default_conv(n_feats*2, n_feats, 3)
        self.c5 = default_conv(n_feats // 2, n_feats // 2, 3)
        self.c6 = default_conv(n_feats*2, n_feats, 1)
        self.se = CALayer(channel=2*n_feats, reduction=16)
        self.sa = SpatialAttention()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        res = x  
        y1 = self.act(self.c1(x))
        y2 = self.act(self.c2(y1))
        y3 = self.act(self.c3(y1))
        cat1 = torch.cat([y1, y2, y3], 1)
        y4 = self.act(self.c4(cat1))
        y5 = self.c5(y3)
        cat2 = torch.cat([y2, y5, y4], 1)
        ca_out = self.se(cat2)
        sa_out = self.sa(cat2)
        y6 = ca_out + sa_out
        y7 = self.c6(y6)
        output = res + y7
        return output

# 通道注意力
class CA(nn.Module):
    def __init__(self, num_fea):
        super(CA, self).__init__()
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_fea, num_fea // 8, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_fea // 8, num_fea, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, fea):
        return self.conv_du(fea)

#空间注意力模块
class SA(nn.Module):
    def __init__(self, n_feats, conv):
        super(SA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)  #Conv2d(40, 12, kernel_size=(1, 1), stride=(1, 1))
        self.conv_f = conv(f, f, kernel_size=1) #Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1))
        self.conv_max = conv(f, f, kernel_size=3, padding=1) #Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0) #Conv2d(12, 12, kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = conv(f, f, kernel_size=3, padding=1) #Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_ = conv(f, f, kernel_size=3, padding=1) #Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = conv(f, n_feats, kernel_size=1) #Conv2d(12, 40, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid = nn.Sigmoid() #Sigmoid()
        self.relu = nn.ReLU(inplace=True) #ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x)) 
        c1 = self.conv2(c1_) 
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) #双线性插值
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf) #逐元素相加
        m = self.sigmoid(c4) 


# 中间的特征提取块------解码block部分
class ExpertBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        in_channels = c
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # GateWeight
        self.GateWeight = GateWeight()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.conv6 = nn.Conv2d(in_channels=c, out_channels=50, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv11 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv33 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 5x5 卷积，为了保持输出尺寸不变，padding 设置为 2
        self.conv55 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)

        self.Expert1 = Expert1(c)   
        self.Expert2 = Expert2(c)    

        self.t = 3 
        self.K = 3  

        # Gate Network(生成对应权重）
        self.GateNetwork = nn.Sequential(
            nn.Linear(c, c // 4, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(c // 4, self.K, bias=False), 
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 

    def forward(self, inp):
        x = inp
        a, b, c, d = inp.shape  # 输入的维度
        # 专家1和专家2得到的结果
        Expert1 = self.Expert1(x)
        Expert2 = self.Expert2(x)

        # print(Expert2.shape)

        # 专家3
        x = self.norm1(x)
        conv1_out = self.conv11(x)
        conv3_out = self.conv33(x)
        conv5_out = self.conv55(x)
        x = conv1_out + conv3_out + conv5_out
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.GateWeight(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.GateWeight(x)
        x = self.conv5(x)
        Expert3 = y + x * self.gamma

        # 动态调节机制--生成一组向量
        s = self.avg_pool(inp).view(a, b) 
        s = self.GateNetwork(s)  
        ax = F.softmax(s / self.t, dim=1)  # 归一化操作（s÷t）---得到两个权值 ax[ : ]
        # print(ax.shape) # ax[ax0  ax1   ax2]相加等于1

        return Expert1 * ax[:, 0].view(a, 1, 1, 1) + Expert2 * ax[:, 1].view(a, 1, 1, 1) + Expert3 * ax[:, 2].view(a, 1,
                                                                                                                   1, 1)
        # return y1 + y  # 加上原特征

######### Multi-scale feature enhancement##################
class MSFEblock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)

        self.conv = nn.Conv2d(dim // 2, dim, 1)
    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        # 减少attn1和attn2的通道维度
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V

    # class FIBlock(nn.Module):
#     def __init__(self):
#         super(FIBlock, self).__init__()
#
#         self.conv1x1_x = None
#         self.conv1x1_y = None
#         self.fc_x = None
#         self.fc_y = None
#         self.conv1x1_concat = None
#
#     def _initialize_layers(self, channels):
#         self.conv1x1_x = nn.Conv2d(channels, channels, 1)
#         self.conv1x1_y = nn.Conv2d(channels, channels, 1)
#         self.fc_x = nn.Linear(channels, channels)
#         self.fc_y = nn.Linear(channels, channels)
#         self.conv1x1_concat = nn.Conv2d(2 * channels, channels, 1)
#
#     def forward(self, x, y, channels):
#         if self.conv1x1_x is None or self.conv1x1_x.in_channels != channels:
#             self._initialize_layers(channels)
#
#         x_conv = self.conv1x1_x(x)
#         y_conv = self.conv1x1_y(y)
#
#         x_avgpool = F.adaptive_avg_pool2d(x_conv, (1, 1)).view(x.size(0), -1)
#         y_avgpool = F.adaptive_avg_pool2d(y_conv, (1, 1)).view(y.size(0), -1)
#
#         x_fc = self.fc_x(x_avgpool)
#         y_fc = self.fc_y(y_avgpool)
#
#         x_sigmoid = torch.sigmoid(x_fc).view(x.size(0), -1, 1, 1)
#         y_sigmoid = torch.sigmoid(y_fc).view(y.size(0), -1, 1, 1)
#
#         x_attention = x * y_sigmoid
#         y_attention = y * x_sigmoid
#
#         concatenated = torch.cat((x_attention, y_attention), dim=1)
#         output = self.conv1x1_concat(concatenated)
#
#         return output


class FIBlock(nn.Module):
    def __init__(self):
        super(FIBlock, self).__init__()
        self.conv1x1_x = None
        self.conv1x1_y = None
        self.fc_x = None
        self.fc_y = None
        self.conv1x1_concat = None
    def _initialize_layers(self, channels):
        self.conv1x1_x = nn.Conv2d(channels, channels, 1)
        self.conv1x1_y = nn.Conv2d(channels, channels, 1)
        self.fc_x = nn.Linear(channels, channels)
        self.fc_y = nn.Linear(channels, channels)
        self.conv1x1_concat = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x, y, channels):
        if self.conv1x1_x is None or self.conv1x1_x.in_channels != channels:
            self._initialize_layers(channels)

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



# 主干网络
class OURNet24(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        # 输入3通道   宽度16  中间块的数量1  编码器块  解码器块的数量
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True) 
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)  

        self.encoders = nn.ModuleList()  # 编码器部分
        self.decoders = nn.ModuleList()  # 解码器部分
        self.middle_blks = nn.ModuleList()  # 中间块
        self.ups = nn.ModuleList() 
        self.downs = nn.ModuleList()  

        chan = width  # chan=width=16
        #############编码器部分##################
        for num in enc_blk_nums:  # 循环编码器块的数量
            self.encoders.append(
                nn.Sequential(
                    *[ExpertBlock(chan) for _ in range(num)]  # 编码器部分-
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)  # 2X2的卷积核
            )  # 下采样操作
            chan = chan * 2  # 通道数在倍增
        #########中间块部分#######################
        self.middle_blks = \
            nn.Sequential(
                *[ExpertBlock(chan) for _ in range(middle_blk_num)] # 中间块-
            )
        #############解码器部分##################
        for num in dec_blk_nums:  # 解码器的数量
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),  # 1X1卷积
                    nn.PixelShuffle(2)  # 上采样操作
                )
            )
            chan = chan // 2  # 通道数减半
            self.decoders.append(
                nn.Sequential(
                    *[ExpertBlock(chan) for _ in range(num)]  # 解码器部分-----两层的特征增强块---CNN结构
                )
            )
        self.padder_size = 2 ** len(self.encoders)

        ######编解码器交叉注意力############
        # self.FIBlock  = FIBlock()

        ####### 多尺度增强  ####
        self.MSFEblock = MSFEblock(width)


    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)  # 检查输入图片的尺寸
        x = self.intro(inp)  
        x = self.MSFEblock(x)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x) # 输入特征x进入编码器
            encs.append(x)
            x = down(x) 
        x = self.middle_blks(x)  # 中间块

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x) # 特征图上采样
            # channels = x.size(1)  # 获取输入的通道数
            # print("channels", channels)
            # print("编码器特征", enc_skip.shape)
            # e_cda = self.FIBlock(x, enc_skip,channels)  # 输入两个特征--进行交叉注意力
            # print("e_cda", e_cda.shape)

            x = x + enc_skip # 编码器输出的特征和上采样之后的特征相加然后输入解码器
            # print("相加之后的特征", x.shape)
            x = decoder(x)  # 特征图经过解码器
            # x = decoder(e_cda) # 特征图经过解码器
            # print("经过解码之后的特征", x.shape)

        x = self.ending(x) # 最后一个卷积层
        x = x + inp # 输出的结果和原始图像相加

        return x[:, :, :H, :W]

    def check_image_size(self, x):  # 检查图片的尺寸
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# 模型参数计算
def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))


class WaterNetLocal(Local_Base, OURNet24):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        OURNet24.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


  # type: OURNet24
  # width: 24
  # enc_blk_nums: [2, 2, 4, 8]
  # middle_blk_num: 6
  # dec_blk_nums: [2, 2, 2, 2]


  # type: OURNet24
  # width: 48  # 改一下64
  # enc_blk_nums: [2, 2, 4, 8]
  # middle_blk_num: 12
  # dec_blk_nums: [2, 2, 2, 2]

if __name__ == '__main__':
    img_channel = 3
    width = 24

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 2, 4]
    middle_blk_num = 4
    dec_blks = [1, 1, 1, 1]

    net = OURNet24(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks,
                 dec_blk_nums=dec_blks)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info  # 缺少这个ptflops包的话去网站下载 然后pip安装

    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    MACs, params = get_model_complexity_info(net, inp_shape, as_strings=True, print_per_layer_stat=True)
    print('MACs: ', MACs, 'params: ', params)

    #  计算模型推理时间
    import time

    t0 = time.time()
    for i in range(2):
        out = OURNet24(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks,dec_blk_nums=dec_blks)
    t = time.time() - t0
    print('平均运行时间: ', t / 2)


    # a=torch.rand((1,3,128,128))#.to('cuda')
    # net = OURNet24(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks,dec_blk_nums=dec_blks)
    # count_parameters(net)
    # y=net
    # print(y)
    # from thop import profile
    # print('==> 方法1..')
    # input = torch.randn(1, 3, 256, 256)
    # flops, params = profile(net, (input,))
    # # 打印模型的计算量（flops和参数量）
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f K' % (flops / 1000000.0, params / 1000.0))


