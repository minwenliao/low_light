import torch
import torch.nn as nn

import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_

#定于2d卷积
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

#Swin Transformer Layer（STL）
class SwinT(nn.Module):
    def __init__(
            # self, conv, n_feats, kernel_size,
            # bias=True, bn=False, act=nn.ReLU(True)):
            self,  n_feats=50): #特征通道数50  中间层通道都是设置成50

        super(SwinT, self).__init__()
        m = [] #这里定义的数组和下面的m.append有关
        depth = 2 #Swin Transformer的个数为2
        num_heads = 25 #在不同层注意力头的个数
        window_size = 8 #窗口大小，默认为7 这里设置是8
        resolution = 64 #分辨率大小
        mlp_ratio = 2.0 #MLP隐藏层特征图通道与嵌入层特征图通道的比，默认为 4

        #参数传至单阶段的 SWin Transformer 基础层（运行BasicLayer）
        m.append(BasicLayer(dim=n_feats, #特征通道数50
                            depth=depth,
                            resolution=resolution,
                            num_heads=num_heads,
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True, qk_scale=None,
                            norm_layer=nn.LayerNorm))

        self.transformer_body = nn.Sequential(*m) #指针-m  list

    def forward(self, x):
        res = self.transformer_body(x)
        return res

# 单阶段的 SWin Transformer 基础层  BasicLayer包括两个部分（Swin Transformer Block 和 Patch Merging
class BasicLayer(nn.Module):
    """
    输入:
        dim (int): 输入特征的维度.
        input_resolution (tuple[int]): 输入分辨率.
        depth (int): SWin Transformer 块的个数.
        num_heads (int): 注意力头的个数.
        window_size (int): 本地(当前块中)窗口的大小.
        mlp_ratio (float): MLP隐藏层特征维度与嵌入层特征维度的比.
        qkv_bias (bool, optional): 给 query, key, value 添加一个可学习偏置，默认为 True.
        qk_scale (float | None, optional): 重写默认的缩放因子 scale.
        drop (float, optional): 随机丢弃神经元，丢弃率默认为 0.0.
        attn_drop (float, optional): 注意力图随机丢弃率，默认为 0.0.
        drop_path (float | tuple[float], optional): 深度随机丢弃率，默认为 0.0.
        norm_layer (nn.Module, optional): 归一化操作，默认为 nn.LayerNorm.
        downsample (nn.Module | None, optional): 结尾处的下采样层，默认没有.
        use_checkpoint (bool): 是否使用 checkpointing 来节省显存，默认为 False.
    """
    def __init__(self, dim, resolution, embed_dim=50, depth=2, num_heads=8, window_size=8,overlap_ratio=0.5,
                 mlp_ratio=1., qkv_bias=True, qk_scale=None, norm_layer=None):
        super().__init__()
        self.dim = dim # 输入特征的维度
        self.resolution = resolution # 输入分辨率
        self.depth = depth # SWin Transformer 块的个数 这里设置是2
        self.window_size = window_size #窗口尺寸是8
        self.overlap_ratio = overlap_ratio
        self.num_head = num_heads
        self.norm_layer = norm_layer
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale

        # 创建 Swin Transformer 网络 在这里改
        # nn.ModuleList迭代性，常用于大量重复网络构建，通过for循环实现超分构建
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, resolution=resolution,  #第①部分
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
        #     OCAB(dim=dim,
        #         input_resolution=resolution,
        #         window_size=window_size,
        #         overlap_ratio=overlap_ratio,
        #         num_heads=num_heads,
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         mlp_ratio=mlp_ratio,
        #         norm_layer=norm_layer)
            for i in range(depth)])  #深度为2 for循环

        # self.blocks =nn.ModuleList([
        #     OCAB(
        #                     dim=dim,
        #                     input_resolution=resolution,
        #                     window_size=window_size,
        #                     overlap_ratio=overlap_ratio,
        #                     num_heads=num_heads,
        #                     qkv_bias=qkv_bias,
        #                     qk_scale=qk_scale,
        #                     mlp_ratio=mlp_ratio,
        #                     norm_layer=norm_layer
        #                     )
        #     for i in range(depth)])  # 深度为2 for循环

        self.patch_embed = PatchEmbed(  #第②部分
            embed_dim=dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
    #检查图片的尺寸
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x, h, w

    #定义前向传播
    def forward(self, x):
        x, h, w = self.check_image_size(x) #检查图片的尺寸
        _, _, H, W = x.size()
        x_size = (H, W)
        x = self.patch_embed(x)  #将2维图像转变成1维patch embeddings
        for blk in self.blocks:
            x = blk(x, x_size)
        x = self.patch_unembed(x, x_size) #将1维 patch embeddings 转变为2维图像
        if h != H or w != W:
            x = x[:, :, 0:h, 0:w].contiguous()
        return x

# Swin Transformer 块
class SwinTransformerBlock(nn.Module):
    """
    输入:
        dim (int): 输入特征的维度.
        input_resolution (tuple[int]): 输入特征图的分辨率.
        num_heads (int): 注意力头的个数.
        window_size (int): 窗口的大小.
        shift_size (int): SW-MSA 的移位值.
        mlp_ratio (float): 多层感知机隐藏层的维度和嵌入层的比.
        qkv_bias (bool, optional): 给 query, key, value 添加一个可学习偏置，默认为 True.
        qk_scale (float | None, optional): 重写默认的缩放因子 scale.
        drop (float, optional): 随机神经元丢弃率，默认为 0.0.
        attn_drop (float, optional): 注意力图随机丢弃率，默认为 0.0.
        drop_path (float, optional): 深度随机丢弃率，默认为 0.0.
        act_layer (nn.Module, optional): 激活函数，默认为 nn.GELU.
        norm_layer (nn.Module, optional): 归一化操作，默认为 nn.LayerNorm.
    """
    def __init__(self, dim, resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim # 输入特征的维度
        self.resolution = to_2tuple(resolution) # 输入特征图的分辨率
        self.num_heads = num_heads # 注意力头的个数
        self.window_size = window_size # 窗口的大小
        self.shift_size = shift_size # SW-MSA 的移位大小
        self.mlp_ratio = mlp_ratio # 多层感知机隐藏层的维度和嵌入层的比
        # if min(self.input_resolution) <= self.window_size:  # 如果输入分辨率小于等于窗口大小
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0 # 移位大小为 0
        #     self.window_size = min(self.input_resolution) # 窗口大小等于输入分辨率大小
        # 断言移位值必须小于等于窗口的大小
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        # 层归一化
        self.norm1 = norm_layer(dim) #LayerNorm((50,), eps=1e-05, elementwise_affine=True) 层归一化1
        # 窗口注意力
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale)
        # 层归一化
        self.norm2 = norm_layer(dim) #LayerNorm((50,), eps=1e-05, elementwise_affine=True) 层归一化2
        mlp_hidden_dim = int(dim * mlp_ratio) # 多层感知机隐藏层维度
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        if self.shift_size > 0:   #如果移位值大于 0
            attn_mask = self.calculate_mask(self.resolution)
        else:
            attn_mask = None # 注意力 mask 赋空

        self.register_buffer("attn_mask", attn_mask) # 保存注意力 mask，不参与更新

    # 计算注意力 mask
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA 计算SW-MSA的注意掩码
        H, W = x_size  # 特征图的高宽
        img_mask = torch.zeros((1, H, W, 1)) # 新建张量，结构为 [1, H, W, 1]
        # 以下两 slices 中的数据是索引，具体缘由尚未搞懂
        h_slices = (slice(0, -self.window_size), # 索引 0 到索引倒数第 window_size
                    slice(-self.window_size, -self.shift_size), # 索引倒数第 window_size 到索引倒数第 shift_size
                    slice(-self.shift_size, None)) # 索引倒数第 shift_size 后所有索引
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt # 将 img_mask 中 h, w 对应索引范围的值置为 cnt
                cnt += 1  # 加1

        mask_windows = window_partition(img_mask, self.window_size)  # 窗口分割，返回值结构为 [nW, window_size, window_size, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # 重构结构为二维张量，列数为 [window_size*window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # 增加第 2 维度减去增加第 3 维度的注意力 mask
        # 用浮点数 -100. 填充注意力 mask 中值不为 0 的元素，再用浮点数 0. 填充注意力 mask 中值为 0 的元素
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
    #前向传播
    def forward(self, x, x_size):
        H, W = x_size # 输入特征图的分辨率
        B, L, C = x.shape # 输入特征的 batch 个数，长度和维度

        shortcut = x
        x = self.norm1(x) # 归一化
        x = x.view(B, H, W, C) # 重构 x 为结构 [B, H, W, C]

        # 循环移位
        if self.shift_size > 0:  # 如果移位值大于 0
            # 第 0 维度上移 shift_size 位，第 1 维度左移 shift_size 位
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x # 不移位

        # 对移位操作得到的特征图分割窗口, nW 是窗口的个数
        x_windows = window_partition(shifted_x, self.window_size) # 结构为 [nW*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # 结构为 [nW*B, window_size*window_size, C]

        #####################计算多头自注意力#########################
        # W-MSA/SW-MSA, 用在分辨率是窗口大小的整数倍的图像上进行测试  计算多头自注意力
        if self.resolution == x_size: # 输入分辨率与设定一致，不需要重新计算注意力 mask
            attn_windows = self.attn(x_windows, mask=self.attn_mask) # 注意力窗口，结构为 [nW*B, window_size*window_size, C]
        else: # 输入分辨率与设定不一致，需要重新计算注意力 mask
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        #合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # 结构为 [-1, window_size, window_size, C]
        shifted_x = window_reverse(attn_windows, self.window_size, H, W) # 结构为 [B, H', W', C]

        # 逆向循环移位
        if self.shift_size > 0:
            # 第 0 维度下移 shift_size 位，第 1 维度右移 shift_size 位
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x # 不逆向移位
        x = x.view(B, H * W, C) # 结构为 [B, H*W， C]

        # FFN
        x = shortcut + x #引入残差
        x = x + self.mlp(self.norm2(x)) # 归一化后通过 MLP
        # x = x + self.mlp(x)
        return x
# 窗口注意力
class WindowAttention(nn.Module):
    """ 基于有相对位置偏差的多头自注意力窗口，支持移位的(shifted)或者不移位的(non-shifted)窗口.
        输入:
        dim (int): 输入特征的维度.
        window_size (tuple[int]): 窗口的大小.
        num_heads (int): 注意力头的个数.
        qkv_bias (bool, optional): 给 query, key, value 添加可学习的偏置，默认为 True.
        qk_scale (float | None, optional): 重写默认的缩放因子 scale.
        attn_drop (float, optional): 注意力权重的丢弃率，默认为 0.0.
        proj_drop (float, optional): 输出的丢弃率，默认为 0.0.
    """
    def __init__(self, dim, window_size, num_heads,qkv_bias=True,qk_scale=None):
        super().__init__()
        self.dim = dim # 输入特征的维度
        self.window_size = window_size # 窗口的高 Wh,宽 Ww
        self.num_heads = num_heads # 注意力头的个数
        head_dim = dim // num_heads # 注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5 # 缩放因子 scale

        # define a parameter table of relative position bias
        # 定义相对位置偏移的参数表，结构为[2 * Wh - 1 * 2 * Ww - 1, num_heads]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        # 获取窗口内每个 token 的成对的相对位置索引
        coords_h = torch.arange(self.window_size[0]) # 高维度上的坐标 (0, 7)
        coords_w = torch.arange(self.window_size[1]) # 宽维度上的坐标 (0, 7)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 坐标，结构为 [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1) # 重构张量结构为 [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # 相对坐标，结构为 [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # 交换维度，结构为 [Wh*Ww, Wh*Ww, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 第0个维度移位
        relative_coords[:, :, 1] += self.window_size[1] - 1  # 第1个维度移位
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # 第1个维度的值乘以2倍的 Ww，再减 1
        relative_position_index = relative_coords.sum(-1) # 相对位置索引，结构为 [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index) # 保存数据，不再更新

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #Linear(in_features=50, out_features=150, bias=True) 线性层(特征维度变为原来的3倍)
        self.proj = nn.Linear(dim, dim) #Linear(in_features=50, out_features=50, bias=True) 线性层(特征维度不变)

        trunc_normal_(self.relative_position_bias_table, std=.02) # 截断正态分布，限制标准差为 0.02
        self.softmax = nn.Softmax(dim=-1) #激活函数Softmax(dim=-1)
    #前向传播
    def forward(self, x, mask=None):
        """
            输入x: 输入特征图，结构为 [num_windows*B, N, C]
            mask: (0/-inf) mask, 结构为 [num_windows, Wh*Ww, Wh*Ww] 或者没有 mask
        """
        B_, N, C = x.shape # 输入特征图的结构
        # 将特征图的通道维度按照注意力头的个数重新划分，并再做交换维度操作
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)方便后续写代码，重新赋值
        # q 乘以缩放因子
        q = q * self.scale
        # @ 代表常规意义上的矩阵相乘
        attn = (q @ k.transpose(-2, -1)) #q和k相乘后并交换最后两个维度

        # 相对位置偏移，结构为 [Wh*Ww, Wh*Ww, num_heads]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # 相对位置偏移交换维度，结构为 [num_heads, Wh*Ww, Wh*Ww]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # 带相对位置偏移的注意力图
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:  # 判断是否有mask
            nW = mask.shape[0] # mask的宽
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0) # 注意力图与mask相加
            attn = attn.view(-1, self.num_heads, N, N) # 恢复注意力图原来的结构
            attn = self.softmax(attn) # 激活注意力图 [0, 1] 之间
        else:
            attn = self.softmax(attn)
        # 注意力图 qk 与 v 相乘得到新的注意力图
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x) # 通过线性层
        return x

# 将多个不重叠窗口重新合并
def window_reverse(windows, window_size, H, W):
    """
    输入:
        windows: (num_windows*B, window_size, window_size, C)  # 分割得到的窗口(已处理)
        window_size (int): Window size  # 窗口大小
        H (int): Height of image  # 原分割窗口前特征图的高
        W (int): Width of image  # 原分割窗口前特征图的宽
    返回:
        x: (B, H, W, C)  # 返回与分割前特征图结构一样的结果
    """
    # 以下就是分割窗口的逆向操作
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

#多层感知机----MLP模块
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features # 输入特征的维度
        hidden_features = hidden_features or in_features # 隐藏特征维度
        self.fc1 = nn.Linear(in_features, hidden_features) # 线性层
        self.act = act_layer() # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features) # 线性层

    def forward(self, x):
        x = self.fc1(x) # Linear(in_features=50, out_features=100, bias=True) 线性层
        x = self.act(x) # GELU() 激活
        x = self.fc2(x) # Linear(in_features=100, out_features=50, bias=True) 线性层
        return x

#将输入分割为多个不重叠窗口
def window_partition(x, window_size): #window_size为窗口的大小
    B, H, W, C = x.shape #输入x: (B, H, W, C)
    # 将输入x重构为结构[batch 个数，高方向的窗口个数，窗口大小，宽方向的窗口个数，窗口大小，通道数]的张量
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 交换重构后x的第3和4维度，5和6维度，再次重构为结构[高和宽方向的窗口个数乘以 batch 个数，窗口大小，窗口大小，通道数] 的张量
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

#恢复之前分割的窗口
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# Patch 编码---将2维图像转变成1维patch embeddings
class PatchEmbed(nn.Module):  # 图像转成 Patch Embeddings
    def __init__(self, embed_dim=50, norm_layer=None):
        super().__init__()

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)  #线性projection输出的通道数，默认为50 （embed_dim=50）
        else:
            self.norm = None
    #前向传播
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C  结构为 [B, num_patches, C]
        if self.norm is not None:
            x = self.norm(x) #LayerNorm((50,), eps=1e-05, elementwise_affine=True)  归一化
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


#将1维 patch embeddings 转变为2维图像
class PatchUnEmbed(nn.Module):  #从Patch Embeddings组合图像
    def __init__(self, embed_dim=50):
        super().__init__()
        self.embed_dim = embed_dim #线性projection输出的通道数

    def forward(self, x, x_size):
        B, HW, C = x.shape #输入x的结构
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C 输出结构为[B, Ph*Pw, C]
        return x

    def flops(self):
        flops = 0
        return flops

#
#
# #OCAB模块----可以看看窗口注意力模块  有点类似
# class OCAB(nn.Module):
#     # overlapping cross-attention block
#     # 和EDSR中的 Swin Transformer 块有点类似
#     def __init__(self, dim,
#                 input_resolution,
#                 window_size,
#                 overlap_ratio,
#                 num_heads,
#                 qkv_bias=True,
#                 qk_scale=None,
#                 mlp_ratio=2,
#                 norm_layer=nn.LayerNorm
#                 ):
#
#         super().__init__()
#         self.dim = dim # 输入特征的维度
#         self.input_resolution = input_resolution # 输入特征图的分辨率
#         self.window_size = window_size # 窗口的大小
#         self.num_heads = num_heads # 注意力头的个数
#         head_dim = dim // num_heads # 注意力头的维度
#         self.scale = qk_scale or head_dim**-0.5 # 缩放因子 scale
#         self.overlap_win_size = int(window_size * overlap_ratio) + window_size
#
#         self.norm1 = norm_layer(dim) #层归一化
#         self.qkv = nn.Linear(dim, dim * 3,  bias=qkv_bias) #线性层(特征维度变为原来的3倍)
#         self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
#
#         # 定义相对位置偏移的参数表，结构为 [2*Wh-1 * 2*Ww-1, num_heads]
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
#
#         trunc_normal_(self.relative_position_bias_table, std=.02) # 截断正态分布，限制标准差为 0.02
#
#         self.softmax = nn.Softmax(dim=-1) # 激活函数 softmax
#
#         self.proj = nn.Linear(dim,dim) # 线性层，特征维度不变
#
#         self.norm2 = norm_layer(dim) #层归一化
#         mlp_hidden_dim = int(dim * mlp_ratio) # 多层感知机隐藏层维度
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU) #MLP-多层感知机
#
#     # 和 Swin Transformer 块的前向传播差不多
#     def forward(self, x, x_size, rpi):
#         h, w = x_size # 输入特征图的尺寸
#         b, _, c = x.shape # 输入特征图的结构
#
#         shortcut = x
#         ################层归一化######################
#         x = self.norm1(x)  #层归一化
#         x = x.view(b, h, w, c) # 重构 x 为结构 [b, h, w, c]
#         ################ OCA 重叠注意力窗口 ######################
#         # 将特征图的通道维度按照注意力头的个数重新划分，并再做交换维度操作
#         qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2) # 3, b, c, h, w
#         q = qkv[0].permute(0, 2, 3, 1) # b, h, w, c
#         kv = torch.cat((qkv[1], qkv[2]), dim=1) # b, 2*c, h, w
#
#         # 窗口划分------q
#         q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
#         q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c
#         # 窗口划分------k v
#         kv_windows = self.unfold(kv) # b, c*w*w, nw
#         kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous() # 2, nw*b, ow*ow, c
#         k_windows, v_windows = kv_windows[0], kv_windows[1] # nw*b, ow*ow, c
#
#         b_, nq, _ = q_windows.shape
#         _, n, _ = k_windows.shape
#         d = self.dim // self.num_heads # 注意力头的维度
#         q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, nq, d
#         k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d
#         v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d
#
#         q = q * self.scale # q 乘以缩放因子
#         # @ 代表常规意义上的矩阵相乘
#         attn = (q @ k.transpose(-2, -1)) # q 和 k 相乘后并交换最后两个维度
#
#         # 相对位置偏移，结构为 [Wh*Ww, Wh*Ww, num_heads]
#         relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
#             self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)  # ws*ws, wse*wse, nH
#         # 相对位置偏移交换维度，结构为 [num_heads, Wh*Ww, Wh*Ww]
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
#         attn = attn + relative_position_bias.unsqueeze(0)  # 带相对位置偏移的注意力图
#
#         attn = self.softmax(attn)  # 激活注意力图 [0, 1] 之间
#
#         # 注意力图与 v 相乘得到新的注意力图attn_windows（新的注意力图）
#         attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim) # 注意力图与 v 相乘得到新的注意力图
#
#         # 合并注意力窗口（新的注意力attn_windows）
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
#         x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c 将多个不重叠窗口重新合并
#         x = x.view(b, h * w, self.dim)
#         x = self.proj(x) + shortcut #通过线性层之后的结果加上x 引入残差
#
#         ##################   层归一化+MLP   #########################
#         x = x + self.mlp(self.norm2(x))  # 归一化后通过 MLP
#         return x


#patch合并
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x


if __name__ == '__main__':
    x = torch.randn((1,50,170,170))
    model = SwinT()
    out = model(x)
    print(out.shape)
