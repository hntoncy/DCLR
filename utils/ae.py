import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NLLNormal(nn.Module):
    """负对数似然损失（基于正态分布）"""

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.multiplier = 1.0 / (2.0 * self.sigma ** 2)
        self.c = -0.5 * np.log(2 * np.pi)  # 常数项

    def forward(self, pred, target):
        """
        Args:
            pred: 预测值，形状为(batch_size, ...)
            target: 目标值，形状同pred
        Returns:
            每个样本的损失值，形状为(batch_size,)
        """
        pred = (pred - pred.mean()) / (pred.std() + 1e-6)
        target = (target - target.mean()) / (target.std() + 1e-6)

        tmp = pred - target
        tmp = tmp ** 2.0
        tmp = -self.multiplier * tmp
        tmp += self.c
        # 对特征维度求和，保留batch维度
        return torch.sum(tmp, dim=tuple(range(1, tmp.dim())))


class MaskedNLLNormal(NLLNormal):
    """带掩码的负对数似然损失（仅计算掩码区域）"""

    def __init__(self, sigma=1.0):
        super().__init__(sigma=sigma)

    def forward(self, pred, target, mask):
        """
        Args:
            pred: 预测值，形状为(batch_size, ...)
            target: 目标值，形状同pred
            mask: 掩码，形状同pred（1表示有效区域，0表示忽略区域）
        Returns:
            每个样本的损失值，形状为(batch_size,)
        """
        pred = (pred - pred.mean()) / (pred.std() + 1e-6)
        target = (target - target.mean()) / (target.std() + 1e-6)

        tmp = pred - target
        tmp = tmp ** 2.0
        tmp = -self.multiplier * tmp
        tmp += self.c
        tmp *= mask  # 应用掩码
        # 对特征维度求和，保留batch维度
        return torch.sum(tmp, dim=tuple(range(1, tmp.dim())))


class KLDStandardNormal(nn.Module):
    """与标准正态分布的KL散度损失"""

    def forward(self, mu, logvar):
        """
        Args:
            mu: 均值，形状为(batch_size, latent_dim)
            logvar: 对数方差，形状同mu
        Returns:
            每个样本的KL散度，形状为(batch_size,)
        """
        # KL散度公式：0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)
        return kl


class ScaleGradient(torch.autograd.Function):
    """自定义梯度缩放操作（用于对抗训练）"""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class NormalEncoder(nn.Module):
    """VAE的标准编码器（输出均值和方差）"""

    def __init__(self, n_in, n_out, weight_init=None, bias_init=0.0):
        super().__init__()
        self.n_out = n_out

        # 初始化权重函数
        if weight_init is None:
            weight_init = nn.init.xavier_normal_

        # 均值和对数方差的线性层
        self.fc_mu = nn.Linear(n_in, n_out)
        self.fc_logvar = nn.Linear(n_in, n_out)

        # 初始化权重
        weight_init(self.fc_mu.weight)
        weight_init(self.fc_logvar.weight)
        nn.init.constant_(self.fc_mu.bias, bias_init)
        nn.init.constant_(self.fc_logvar.bias, bias_init)

    def forward(self, h_enc):
        """
        Args:
            h_enc: 编码器的中间特征，形状为(batch_size, n_in)
        Returns:
            z: 采样的隐变量，形状为(batch_size, n_out)
            kld_loss: KL散度损失，形状为(batch_size,)
        """
        mu = self.fc_mu(h_enc)
        logvar = self.fc_logvar(h_enc)

        # 新增方差约束（防止logvar过大/过小）
        logvar = torch.clamp(logvar, min=-5, max=5)  # <--- 添加这行

        # 重参数化采样
        std = torch.exp(0.5 * logvar) + 1e-6  # 方差加上一个微小值以避免数值溢出
        eps = torch.randn_like(std)
        z = mu + eps * std

        # 计算KL散度
        kld_loss = KLDStandardNormal()(mu, logvar)
        return z, kld_loss


class AdversarialEncoder(nn.Module):
    """带对抗损失的编码器"""

    def __init__(self, n_in, n_out, discriminator, weight_init=None,
                 bias_init=0.0, recon_weight=0.01, eps=1e-4):
        super().__init__()
        self.n_out = n_out
        self.discriminator = discriminator
        self.recon_weight = recon_weight
        self.eps = eps

        if weight_init is None:
            weight_init = nn.init.xavier_normal_

        # 编码器输出层
        self.fc_z = nn.Linear(n_in, n_out)
        weight_init(self.fc_z.weight)
        nn.init.constant_(self.fc_z.bias, bias_init)

    def forward(self, h_enc):
        """
        Args:
            h_enc: 编码器的中间特征，形状为(batch_size, n_in)
        Returns:
            z: 编码的隐变量（带梯度缩放），形状为(batch_size, n_out)
            adv_loss: 对抗损失，标量
        """
        batch_size = h_enc.size(0)
        z = self.fc_z(h_enc)

        # 生成随机样本（标准正态分布）
        z_samples = torch.randn_like(z)

        # 对抗判别（对真实样本和生成样本进行判别）
        z_adv = torch.cat([z_samples, ScaleGradient.apply(z, -1.0)], dim=0)
        d_out = self.discriminator(z_adv)

        # 计算对抗损失
        labels = torch.cat([
            torch.ones(batch_size, 1, device=z.device),  # 真实样本标签
            -torch.ones(batch_size, 1, device=z.device)  # 生成样本标签
        ], dim=0)
        offset = torch.cat([
            torch.zeros(batch_size, 1, device=z.device),
            torch.ones(batch_size, 1, device=z.device)
        ], dim=0)

        adv_loss = -torch.sum(torch.log(d_out * labels + offset + self.eps))
        adv_loss = adv_loss / batch_size  # 归一化

        # 对用于重构的z进行梯度缩放
        z_recon = ScaleGradient.apply(z, self.recon_weight)
        return z_recon, adv_loss


class Autoencoder(nn.Module):
    """自编码器基础类"""

    def __init__(self, encoder, latent_encoder, decoder, recon_loss='bce'):
        super().__init__()
        self.encoder = encoder  # 特征编码器（如CNN）
        self.latent_encoder = latent_encoder  # 隐变量编码器（如NormalEncoder）
        self.decoder = decoder  # 解码器

        # 重构损失函数
        if recon_loss == 'bce':
            self.recon_loss = nn.BCEWithLogitsLoss(reduction='none')
        elif recon_loss == 'mse':
            self.recon_loss = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"不支持的重构损失: {recon_loss}")

    def forward(self, x):
        """
        Args:
            x: 输入数据，形状为(batch_size, ...)
        Returns:
            x_tilde: 重构数据，形状同x
            total_loss: 总损失（标量）
            loss_components: 损失组成部分（字典）
        """
        batch_size = x.size(0)

        # 编码过程
        h_enc = self.encoder(x)
        z, encoder_loss = self.latent_encoder(h_enc)

        # 解码过程
        x_tilde = self.decoder(z)

        # 计算重构损失
        recon_loss = torch.sum(self.recon_loss(x_tilde, x),
                               dim=tuple(range(1, x_tilde.dim())))

        # 总损失
        total_loss = torch.mean(encoder_loss + recon_loss)

        return {
            'x_tilde': x_tilde,
            'total_loss': total_loss,
            'loss_components': {
                'encoder_loss': torch.mean(encoder_loss),
                'recon_loss': torch.mean(recon_loss)
            }
        }

    def encode(self, x):
        """仅编码过程"""
        h_enc = self.encoder(x)
        z, _ = self.latent_encoder(h_enc)
        return z

    def decode(self, z):
        """仅解码过程"""
        return self.decoder(z)
