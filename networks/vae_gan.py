import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np


class ScaleGradient(torch.autograd.Function):
    """
    自定义的梯度缩放函数，用于对抗训练中梯度反转和权重平衡
    """
    @staticmethod
    def forward(ctx, x, scale):
        """原样输出"""
        # 自动将Python数值类型转换为张量
        if isinstance(scale, (int, float)):
            scale = torch.tensor(scale, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """缩放梯度"""
        scale, = ctx.saved_tensors
        return grad_output * scale, None


class NLLNormal(nn.Module):
    def __init__(self, sigma=1.0):
        super(NLLNormal, self).__init__()
        self.sigma = sigma  # 正态分布的标准差
        # 直接计算常数，不需要存储为属性
        self.multiplier_val = 1.0 / (2.0 * sigma ** 2)
        self.c = -0.5 * np.log(2 * np.pi)

    def forward(self, pred, target):
        # 保护数值范围
        pred = torch.clamp(pred, min=-5.0, max=5.0)
        target = torch.clamp(target, min=-5.0, max=5.0)

        tmp = pred - target
        tmp = tmp ** 2
        # 限制极端值
        tmp = torch.clamp(tmp, max=10.0) + 1e-6  # 防止除零

        # 使用预先计算的常数
        tmp = -self.multiplier_val * tmp
        tmp += self.c

        # 返回均值而非总和
        return torch.mean(torch.sum(tmp, dim=1))

class KLDStandardNormal(nn.Module):
    def forward(self, mu, logvar):
        # 添加logvar数值稳定性保护
        logvar = torch.clamp(logvar, min=-5, max=5)  # 防止指数运算溢出
        mu = torch.clamp(mu, min=-5, max=5)
        # 数值稳定的KL散度计算
        kl = 0.5 * torch.sum(
            torch.exp(logvar) + mu.pow(2) - 1.0 - logvar,
            dim=1
        ).mean()
        kl = torch.clamp(kl, min=0, max=10.0)  # 限制KL损失最大为10，防止溢出

        return kl


class NormalEncoder(nn.Module):
    def __init__(self, n_in, n_out, fused_feat_dim, weight_init=None, bias_init=0.0):
        super(NormalEncoder, self).__init__()
        self.n_out = n_out
        self.fused_feat_dim = fused_feat_dim  # 融合特征的维度（来自分割网络的frs/frt）

        # 初始化权重
        if weight_init is None:
            weight_init = nn.init.xavier_normal_

        # 生成潜变量z的均值和方差
        self.z_mu = nn.Linear(n_in, n_out)
        self.z_logvar = nn.Linear(n_in, n_out)

        # 初始化参数
        weight_init(self.z_mu.weight)
        nn.init.constant_(self.z_mu.bias, bias_init)
        weight_init(self.z_logvar.weight)
        nn.init.constant_(self.z_logvar.bias, bias_init)

        # 新增：融合后潜向量的投影层（可选，用于降维或调整维度）
        self.fuse_proj = nn.Linear(n_out + fused_feat_dim, n_out)  # z维度 + 融合特征维度 → 新z维度

        self.z_samples = None
        self._batch_size = None

    def samples(self, batch_size, device):
        """生成标准正态分布的随机样本，以便重参数化技巧生成潜在变量z"""
        if self._batch_size != batch_size or self.z_samples is None:
            self.z_samples = torch.randn(batch_size, self.n_out, device=device)
            self._batch_size = batch_size
        return self.z_samples

    def forward(self, h_enc, fused_feat):
        batch_size = h_enc.size(0)
        device = h_enc.device

        # 生成原始潜变量z
        z_mu = self.z_mu(h_enc)  # (batch_size, n_out)
        z_logvar = self.z_logvar(h_enc)  # (batch_size, n_out)
        z_eps = self.samples(batch_size, device)  # 重参数化技巧生成的噪声样本
        z = z_mu + torch.exp(0.5 * z_logvar) * z_eps  # (batch_size, n_out)

        # 核心：将z与融合特征fused_feat拼接融合
        # 确保fused_feat的形状为(batch_size, fused_feat_dim)
        if fused_feat.dim() == 4:
            # 全局平均池化去掉H和W，得到 (B, C, 1, 1)
            fused_feat = F.adaptive_avg_pool2d(fused_feat, (1, 1))
        fused_feat = fused_feat.view(batch_size, -1)  # 展平融合特征（如需）
        z_fused = torch.cat([z, fused_feat], dim=1)  # (batch_size, n_out + fused_feat_dim)

        # 可选：通过投影层调整融合后的维度（保持与原始z维度一致，便于解码器兼容）
        z_fused = self.fuse_proj(z_fused)  # (batch_size, n_out)

        # 计算KL散度（仍基于原始z的分布）
        kld = KLDStandardNormal()(z_mu, z_logvar)

        return z_fused, kld  # 返回融合后的潜向量

class AEGAN(nn.Module):
    def __init__(self, encoder, latent_encoder, decoder, discriminator,
                 recon_depth=9, discriminate_sample_z=True,
                 discriminate_ae_recon=True, recon_vs_gan_weight=5e-5,
                 real_vs_gen_weight=0.5, eps=1e-6):
        super(AEGAN, self).__init__()
        self.encoder = encoder
        self.latent_encoder = latent_encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.recon_depth = recon_depth
        self.discriminate_sample_z = discriminate_sample_z
        self.discriminate_ae_recon = discriminate_ae_recon
        self.recon_vs_gan_weight = recon_vs_gan_weight
        self.real_vs_gen_weight = real_vs_gen_weight
        self.eps = eps
        self.recon_error = NLLNormal()  # 负对数似然损失，用于正态分布

        # 创建负梯度解码器
        self.decoder_neggrad = self._create_neggrad_decoder()

        # 处理重构深度
        self.discriminator_recon = None
        if recon_depth > 0:
            self.discriminator_recon = self._create_recon_discriminator()

    def _create_neggrad_decoder(self):
        decoder_neggrad = nn.Sequential(*[l for l in self.decoder.children()])
        for p, p_neg in zip(self.decoder.parameters(), decoder_neggrad.parameters()):
            p_neg.data = p.data.clone()
            p_neg.requires_grad = False  # 共享参数但不单独更新
        return decoder_neggrad

    def _create_recon_discriminator(self):
        # 创建用于重构损失的判别器部分
        recon_layers = list(self.discriminator.children())[:self.recon_depth]
        discriminator_recon = nn.Sequential(*recon_layers)
        for p, p_recon in zip(self.discriminator.parameters(), discriminator_recon.parameters()):
            p_recon.data = p.data.clone()
            p_recon.requires_grad = False  # 共享参数但不单独更新
        return discriminator_recon

    def encode(self, x, fused_feat):
        enc = self.encoder(x)
        z, encoder_loss = self.latent_encoder(enc, fused_feat)
        return z, encoder_loss

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, fused_feat):
        batch_size = x.size(0)
        device = x.device

        # 编码过程
        enc = self.encoder(x)  # x=frs/frt
        z, encoder_loss = self.latent_encoder(enc, fused_feat)  # fused_feat=fzs/fzt

        # 解码过程与重构损失
        x_tilde = self.decoder(z)
        if self.recon_depth > 0 and self.discriminator_recon is not None:
            x_combined = torch.cat([x_tilde, x], dim=1)
            d_generated = self.discriminator_recon(x_combined)

            x_real_combined = torch.cat([x, x], dim=1)
            d_real = self.discriminator_recon(x_real_combined)
            recon_loss = self.recon_error(d_generated, d_real)
        else:
            recon_loss = self.recon_error(x_tilde, x)

        total_loss = encoder_loss + recon_loss

        gen_size = 0
        z_gan = z

        # 准备生成样本
        if self.discriminate_ae_recon:
            gen_size += batch_size
            z_gan = ScaleGradient.apply(z_gan, torch.tensor(0.0, device=device))

        if self.discriminate_sample_z:
            gen_size += batch_size
            z_samples = self.latent_encoder.samples(batch_size, device)
            if self.discriminate_ae_recon:
                z_gan = torch.cat([z_gan, z_samples], dim=0)
            else:
                z_gan = z_samples

        if gen_size == 0:
            raise ValueError('GAN does not receive any generated samples.')

        x_gen = self.decoder_neggrad(z_gan)

        k = gen_size // batch_size
        x_repeated = torch.repeat_interleave(x, repeats=k, dim=0)
        x_gan_combined = torch.cat([x_repeated, x_gen], dim=1)

        dis_batch_size = batch_size
        real_weight = self.real_vs_gen_weight
        gen_weight = (1 - self.real_vs_gen_weight) * float(batch_size) / gen_size

        weights_real = torch.ones(batch_size, device=device) * real_weight
        weights_gen = torch.ones(gen_size - batch_size, device=device) * gen_weight
        weights = torch.cat([weights_real, weights_gen], dim=0)

        weights = weights.view(-1, *([1] * (x_gan_combined.dim() - 1)))

        x_gan_scaled = ScaleGradient.apply(x_gan_combined, 1.0 / weights.view(-1, *([1] * (x_gan_combined.dim() - 1))))
        d = self.discriminator(x_gan_scaled)
        d = d.view(-1)
        d_scaled = ScaleGradient.apply(d, weights)
        batch_size_d = d_scaled.shape[0]

        sign = torch.ones(batch_size_d, device=device)
        sign[batch_size:] = -1.0
        offset = torch.zeros_like(sign)
        offset[batch_size:] = 1.0

        gan_loss = -torch.log(d_scaled * sign + offset + self.eps).mean()

        total_loss += self.recon_vs_gan_weight * gan_loss

        if torch.isnan(z).any():
            print("NaN in latent z, skipping batch.")
            dummy = torch.tensor(float('nan'), device=x.device)
            return torch.zeros_like(x), dummy, dummy, dummy, dummy

        return x_tilde, total_loss, encoder_loss, recon_loss, gan_loss

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1)
        max_out = self.fc(max_out).view(b, c, 1, 1)
        return x * (avg_out + max_out)  # 保持通道数不变


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 保持空间维度
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)  # 2通道
        attn = self.sigmoid(self.conv(x_cat))  # 1通道注意力图
        return x * attn  # 保持输入通道数不变

# 解码器（保持与编码器对称）
class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=1, stride=2,
            padding=0, output_padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv(x)
        return self.relu(x + residual)


