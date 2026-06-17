# # -*- coding: utf-8 -*-
# """
# The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
# Added residual connections for better gradient flow
# """
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ConvBlock(nn.Module):
    """two convolution layers with batch norm, leaky relu and residual connection"""

    def __init__(self, in_channels, out_channels, dropout_p, use_se=False):
        super(ConvBlock, self).__init__()
        self.use_se = use_se
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # 残差连接：如果输入输出通道数不同，使用1x1卷积调整
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.LeakyReLU()

        if self.use_se:
            self.se = SELayer(out_channels)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv_conv(x)
        x = x + residual  # 残差连接
        x = self.activation(x)
        if self.use_se:
            x = self.se(x)
        return x


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock with residual connection"""

    def __init__(self, in_channels, out_channels, dropout_p, use_se=False):
        super(DownBlock, self).__init__()
        # 下采样路径
        self.downsample = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p, use_se)
        # 残差连接：下采样+1x1卷积调整通道数
        self.shortcut = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.downsample(x)
        x = self.conv(x)
        x = x + residual  # 残差连接
        return x


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock with residual connection"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True, use_se=False):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        self.use_se = use_se

        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)

        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p, use_se)
        # 上采样残差连接
        self.shortcut = nn.Conv2d(in_channels2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        # x1: 来自上层的特征, x2: 来自编码器的跳跃连接特征
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)

        # 残差连接准备
        residual = self.shortcut(x2)

        # 拼接跳跃连接特征
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = x + residual  # 残差连接

        return x


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.use_se = self.params.get('use_se', False)  # Default to False if not specified
        self.dropout = self.params['dropout']
        assert len(self.ft_chns) == 5

        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0], self.use_se)
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1], self.use_se)
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2], self.use_se)
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3], self.use_se)
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4], self.use_se)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.use_se = self.params.get('use_se', False)
        assert len(self.ft_chns) == 5

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3],
            dropout_p=0.0, bilinear=self.bilinear, use_se=self.use_se
        )
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2],
            dropout_p=0.0, bilinear=self.bilinear, use_se=self.use_se
        )
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1],
            dropout_p=0.0, bilinear=self.bilinear, use_se=self.use_se
        )
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0],
            dropout_p=0.0, bilinear=self.bilinear, use_se=self.use_se
        )

        # 最终输出的残差连接
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.ft_chns[0], self.ft_chns[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.ft_chns[0]),
            nn.LeakyReLU(),
            nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=1)
        )

        self.decoder_features = []  # 存储解码器中间特征

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        self.decoder_features.clear()  # 清空旧特征

        x = self.up1(x4, x3)
        self.decoder_features.append(x)  # 第一层上采样结果
        x = self.up2(x, x2)
        self.decoder_features.append(x)  # 第二层上采样结果
        x = self.up3(x, x1)
        self.decoder_features.append(x)  # 第三层上采样结果
        x = self.up4(x, x0)
        self.decoder_features.append(x)  # 第四层上采样结果

        # 最终输出的残差连接
        residual = x
        output = self.out_conv(x)
        output = output + self.out_conv(residual)  # 最终输出的残差连接

        return output

class UNet_f(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_f, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=in_chns,
            classes=class_num,
            activation=None,
            # 关键修正：解码器块数量为5，因此提供5个通道值
            # 格式：(最深层通道, ..., 最浅层通道)，最后一个值对应up1的通道数
            decoder_channels=(256, 128, 64, 32, 16)  # 5个值，对应5个块
        )
        self.features = {}
        # 调整特征融合的输入通道（根据实际打印的up1~up4通道数）
        # self.feat_fusion = FeatureFusion(feat_channels=[32, 64, 128, 256])
        # self.cbam = CBAM(in_channels=64 + class_num)

        # 注册解码器特征钩子（适配5个块，取前3个用于up1~up3）
        self.decoder_block0_out = None  # blocks[0]（第1个块）
        self.decoder_block1_out = None  # blocks[1]（第2个块）
        self.decoder_block2_out = None  # blocks[2]（第3个块）
        self.decoder_block3_out = None  # blocks[3]（第4个块）  # 新增

        # 定义钩子函数（新增block3的钩子）
        def hook_block0(m, i, o):
            self.decoder_block0_out = o

        def hook_block1(m, i, o):
            self.decoder_block1_out = o

        def hook_block2(m, i, o):
            self.decoder_block2_out = o

        def hook_block3(m, i, o):
            self.decoder_block3_out = o  # 新增

        # 注册5个块中的前4个（根据需要调整索引）
        if len(self.model.decoder.blocks) >= 4:  # 检查是否有足够的块
            self.model.decoder.blocks[0].register_forward_hook(hook_block0)
            self.model.decoder.blocks[1].register_forward_hook(hook_block1)
            self.model.decoder.blocks[2].register_forward_hook(hook_block2)
            self.model.decoder.blocks[3].register_forward_hook(hook_block3)  # 新增
        else:
            raise ValueError(f"解码器特征块数量不足，实际为{len(self.model.decoder.blocks)}")

    def forward(self, x):
        encoder_features = self.model.encoder(x)

        # 检查 encoder 输出
        # print("Encoder features length:", len(encoder_features))
        # print("Encoder feature shapes:", [f.shape for f in encoder_features])
        # print("Decoder input length:", len(encoder_features))  # 应该是 5
        # print("Decoder input unpacked shapes:", [f.shape for f in encoder_features])

        decoder_output = self.model.decoder(*encoder_features)  # 解码器最终输出（up4）,加了个解包操作 *encoder_features

        # 打印各特征形状，确认通道数（关键！）
        # print("up4（decoder_output）形状:", decoder_output.shape)  # 应输出 (B, 32, 512, 512)
        # print("up3（block3输出）形状:", self.decoder_block3_out.shape)  # (B, 64, 256, 256)
        # print("up2（block2输出）形状:", self.decoder_block2_out.shape)  # (B, 128, 128, 128)
        # print("up1（block1输出）形状:", self.decoder_block1_out.shape)  # (B, 256, 64, 64)

        # 特征映射（根据打印结果调整up1~up4的对应关系）
        self.features['up4'] = decoder_output  # 32通道（最深层）
        self.features['up3'] = self.decoder_block3_out  # 64通道
        self.features['up2'] = self.decoder_block2_out  # 128通道
        self.features['up1'] = self.decoder_block1_out  # 256通道
        self.bottleneck = encoder_features[-1]

        # 后续特征融合和输出逻辑不变
        output = self.model.segmentation_head(decoder_output)
        # self.fr = self.feat_fusion([
        #     self.features['up4'],
        #     self.features['up3'],
        #     self.features['up2'],
        #     self.features['up1']
        # ])
        # self.fr = nn.functional.interpolate(
        #     self.fr, size=x.shape[2:], mode='bilinear', align_corners=True
        # )

        return output, self.features, self.bottleneck


class FeatureFusion(nn.Module):
    def __init__(self, feat_channels, out_channels=64):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in feat_channels
        ])
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, features):
        target_size = features[-1].shape[2:]
        fused = []
        for i, feat in enumerate(features):
            upsampled = nn.functional.interpolate(
                feat, size=target_size, mode='bilinear', align_corners=True
            )
            fused.append(self.convs[i](upsampled))
        fused = torch.cat(fused, dim=1)
        return self.fuse_conv(fused)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        ca = self.channel_attention(x)
        x_ca = x * ca
        max_pool = torch.max(x_ca, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        sa_input = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(sa_input)
        x_sa = x_ca * sa
        return self.final_conv(x_sa)



