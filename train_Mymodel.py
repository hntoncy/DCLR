import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

from torch import nn
import argparse
import logging
import random
import shutil
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from dataloaders.my_dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler, WeakStrongAugment, OVRWeakStrongAugment
# Bra2021
# from dataloaders.my_brats import BaseDataSets, RandomGenerator, TwoStreamBatchSampler, WeakStrongAugment_3Dircadb
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from networks import g
from val_Mymodel import test_single_volume
from networks.vae_gan import AEGAN, NormalEncoder, ChannelAttention, SpatialAttention, ResidualUpBlock
from utils.losses import BoundaryAwareDiceLoss
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import spectral_norm
from torch.optim.lr_scheduler import CosineAnnealingLR

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='')
parser.add_argument('--exp', type=str, default='')
parser.add_argument('--model', type=str, default='unet_f')
parser.add_argument('--max_iterations', type=int, default=2400)
parser.add_argument('--batch_size', type=int, default=8)  #
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.0001)
parser.add_argument('--patch_size', type=list, default=[512, 512])
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--num_classes', type=int, default=2)

parser.add_argument('--labeled_bs', type=int, default=4)
parser.add_argument('--labeled_num', type=int, default=10)

parser.add_argument('--ema_decay', type=float, default=0.99)
parser.add_argument('--consistency', type=float, default=0.1)
parser.add_argument('--consistency_rampup', type=float, default=3000.0)
parser.add_argument('--contrast_weight', type=float, default=0.01)
parser.add_argument('--contrast_rampup', type=float, default=3000.0)

# VAE-GAN参数
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--vae_gan_weight', type=float, default=0.001)
parser.add_argument('--uncertainty_threshold', type=float, default=0.1)

# 动态阈值参数
parser.add_argument('--initial_uncertainty_threshold', type=float, default=0.05)
parser.add_argument('--target_uncertainty_threshold', type=float, default=0.5)
parser.add_argument('--threshold_rampup', type=float, default=3000)

# 训练迭代数
parser.add_argument('--stage1', type=int, default=0)  # 1K
parser.add_argument('--stage2', type=int, default=0)  # 2K

args = parser.parse_args()


# 工具函数
def patients_to_slices(dataset, pat_num):
    ref = {
        "ACDC": {"3": 68, "7": 136},
        "ISIC2017": {"10": 193, "20": 387, "30": 580},
        "KvasirSEG": {"10": 90, "20": 180, "30": 270},
        "3Dircadb": {"10": 51, "20": 102, "30": 153},
        "lits17": {"10": 647, "20": 1294, "30": 1941},
        "Bra2021": {"10": 112, "20": 225, "30": 337},
        "Prostate": {"2": 27}
    }
    for k, v in ref.items():
        if k in dataset:
            return v[str(pat_num)]
    raise ValueError("Unsupported dataset")


def get_current_weight(w, i, ramp):
    return w * ramps.sigmoid_rampup(i, ramp)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(alpha).add_(p.data, alpha=1 - alpha)


def create_model(ema=False):
    model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
    if ema:
        for p in model.parameters():
            p.detach_()
    return model


def create_vae_gan(latent_dim=args.latent_dim):
    # 编码器：严格保证通道流转 1→32→64→128
    vae_encoder = nn.Sequential(
        # 输入: [B, 1, 512, 512]
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),  # 输出: [B, 32, 256, 256]

        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),  # 输出: [B, 64, 128, 128]

        # 注意力模块：输入64通道，输出64通道
        ChannelAttention(64),
        SpatialAttention(),  # 保持64通道

        # 关键层：必须接收64通道输入
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),  # 输出: [B, 128, 64, 64]

        ChannelAttention(128),
        SpatialAttention(),  # 保持128通道

        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 64),
        nn.ReLU()
    )

    vae_decoder = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Unflatten(1, (128, 1, 1)),
        nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True),
        ResidualUpBlock(128, 64),
        ResidualUpBlock(64, 32),
        nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
        nn.Sigmoid()
    )

    # 潜变量编码器
    latent_encoder = NormalEncoder(
        n_in=64,
        n_out=latent_dim,
        fused_feat_dim=64  # fr 改
    )

    # 判别器
    discriminator = nn.Sequential(
        spectral_norm(nn.Conv2d(2, 32, 3, stride=2, padding=1)),
        nn.LeakyReLU(0.2),
        nn.LayerNorm([32, 256, 256]),
        spectral_norm(nn.Conv2d(32, 64, 3, stride=2, padding=1)),
        nn.LeakyReLU(0.2),
        nn.LayerNorm([64, 128, 128]),
        spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
        nn.LeakyReLU(0.2),
        nn.LayerNorm([128, 64, 64]),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        spectral_norm(nn.Linear(128, 1)),
        nn.Sigmoid()
    )

    # 组装AEGAN
    aegan = AEGAN(
        encoder=vae_encoder,
        latent_encoder=latent_encoder,
        decoder=vae_decoder,
        discriminator=discriminator,
        recon_depth=3,
        discriminate_sample_z=True,
        discriminate_ae_recon=True,
        recon_vs_gan_weight=1e-5
    )

    return aegan


# 训练步骤
def uncertainty_aware_replacement(pred_logits, pred_soft, generated_logits, args, iter_num, entropy=None):
    """
    动态调整uncertainty_threshold
    """
    # 计算预测熵（衡量不确定性）
    # if entropy is None:
    #     entropy = -torch.sum(pred_soft * torch.log(pred_soft + 1e-8), dim=1)  # 形状：(B, H, W)

    # 计算改进的预测熵（衡量不确定性，包含邻域方差惩罚）
    if entropy is None:
        entropy = improved_uncertainty_entropy(
            pred_soft,
            spatial_weight=0.1,  # 可调参数
            kernel_size=3
        )  # 形状：(B, H, W)

    # 基于迭代次数的基础阈值
    initial_threshold = args.initial_uncertainty_threshold  # 训练初期阈值
    target_threshold = args.target_uncertainty_threshold    # 训练后期目标阈值
    rampup_iter = args.threshold_rampup        # 阈值达到目标值的迭代次数

    rampup = ramps.sigmoid_rampup(iter_num, rampup_iter)
    base_threshold = initial_threshold + (target_threshold - initial_threshold) * rampup

    # 结合当前批次熵的分布微调
    entropy_flat = entropy.view(-1)
    entropy_valid = entropy_flat[entropy_flat.isfinite()]
    if len(entropy_valid) == 0:
        current_threshold = base_threshold
    else:
        entropy_mean = torch.mean(entropy_valid)
        entropy_std = torch.std(entropy_valid)
        fine_tune = torch.clamp(0.2 * entropy_std + 0.05 * entropy_mean, -0.1, 0.2)
        current_threshold = base_threshold + fine_tune.item()

    # 裁剪阈值范围
    current_threshold = max(0.03, min(current_threshold, 0.5))

    # 生成不确定性掩码
    uncertainty_mask = entropy > current_threshold

    # 替换logits
    replaced_logits = torch.where(
        uncertainty_mask.unsqueeze(1),  # 扩展维度匹配logits（B, C, H, W）
        generated_logits,
        pred_logits
    )

    return replaced_logits, uncertainty_mask, current_threshold

def training_step(model, ema_model, vae_gan, volume, label, optimizer_seg, optimizer_vae, iter_num, args):
    model.train()
    ema_model.eval()
    vae_gan.train()

    device = volume.device
    batch_size = volume.size(0)
    labeled_bs = args.labeled_bs

    # 分割网络前向传播
    pred_s, feat_s = model(volume)
    pred_s_soft = torch.clamp(F.softmax(pred_s, dim=1), min=1e-7, max=1 - 1e-7)
    target_s_onehot = F.one_hot(label[:labeled_bs].long(), num_classes=args.num_classes).permute(0, 3, 1, 2).float()  # 目标独热编码

    with torch.no_grad():
        noise = torch.clamp(torch.randn_like(volume) * 0.1, -0.2, 0.2)
        pred_t, feat_t = ema_model(volume + noise)
        pred_t_soft = torch.clamp(F.softmax(pred_t, dim=1), min=1e-7, max=1 - 1e-7)
        t_pseudo_labels = torch.argmax(pred_t_soft[labeled_bs:], dim=1)
        target_t_onehot = F.one_hot(label[labeled_bs:].long(), num_classes=args.num_classes).permute(0, 3, 1, 2).float()  # 目标独热编码


    # 修正前
    loss_ce_sl = F.cross_entropy(pred_s[:args.labeled_bs], label[:args.labeled_bs].long())
    loss_dice_sl = losses.DiceLoss(args.num_classes)(pred_s_soft[:args.labeled_bs],
                                                     label[:args.labeled_bs].unsqueeze(dim=1))

    loss_ce_tl = F.cross_entropy(pred_t[:args.labeled_bs], label[:args.labeled_bs].long())
    loss_dice_tl = losses.DiceLoss(args.num_classes)(pred_t_soft[:args.labeled_bs],
                                                     label[:args.labeled_bs].unsqueeze(dim=1))

    sup_loss = 0.5 * (loss_ce_sl + loss_s_bdice) + 0.5 * (loss_ce_tl + loss_t_bdice)

    # 权重
    w_consistency = get_current_weight(args.consistency, iter_num, args.consistency_rampup)
    w_contrast = get_current_weight(args.contrast_weight, iter_num, args.contrast_rampup)
    con_loss = g.InfoNCEWithDualViewLoss(
        temperature=0.1,
        gat_in_dim=32,  # 改
        gat_out_dim=128,  # 128
        num_samples=256,
        dropout=0.1
    ).cuda()

    # 损失初始化
    consistency_loss_u = torch.tensor(0.0).to(device)
    consistency_loss_l = torch.tensor(0.0).to(device)
    loss_contrast_l = torch.tensor(0.0).to(device)
    loss_contrast_u = torch.tensor(0.0).to(device)

    if iter_num >= args.stage1:
        # 一致性损失
        consistency_loss_u = torch.mean((pred_s_soft[labeled_bs:] - pred_t_soft[labeled_bs:]) ** 2)
        consistency_loss_l = torch.mean((pred_s_soft[:labeled_bs] - pred_t_soft[:labeled_bs]) ** 2)

        # 对比损失
        loss_contrast_l = con_loss(
            feat_s['up4'][:labeled_bs],
            feat_t['up4'][:labeled_bs],
            labels=label[:labeled_bs],
            pseudo_labels=label[:labeled_bs]
        )
        loss_contrast_u = con_loss(
            feat_s['up4'][labeled_bs:],
            feat_t['up4'][labeled_bs:],
            labels=t_pseudo_labels,
            pseudo_labels=t_pseudo_labels
        )

    # 不确定性修正
    loss_ce_s = torch.tensor(0.0).to(device)
    loss_ce_t = torch.tensor(0.0).to(device)
    loss_dice_s = torch.tensor(0.0).to(device)
    loss_dice_t = torch.tensor(0.0).to(device)
    vae_loss = torch.tensor(0.0).to(device)
    vae_loss_s = torch.tensor(0.0).to(device)
    vae_loss_t = torch.tensor(0.0).to(device)
    gan_loss = torch.tensor(0.0).to(device)
    sup_loss2 = torch.tensor(0.0).to(device)
    consistency_loss_u2 = torch.tensor(0.0).to(device)
    consistency_loss_l2 = torch.tensor(0.0).to(device)
    uncertainty_ratio_s = 0.0
    uncertainty_ratio_t = 0.0
    threshold_s = 0.0
    threshold_t = 0.0

    if iter_num >= args.stage2:
        frs_up = nn.functional.interpolate(
            model.fr,
            size=pred_s_soft.shape[2:],
            mode='bilinear',
            align_corners=True
        )
        frt_up = nn.functional.interpolate(
            ema_model.fr,
            size=pred_t_soft.shape[2:],
            mode='bilinear',
            align_corners=True
        )

        frs = torch.cat([frs_up, pred_s_soft], dim=1).to(device)
        frt = torch.cat([frt_up, pred_t_soft], dim=1).to(device)

        frs = model.cbam(frs)
        frt = ema_model.cbam(frt)

        fzs = model.fr
        fzt = ema_model.fr

        # 额外的数值检查
        if torch.isnan(frs).any() or torch.isinf(frs).any():
            print(f"[Warning] Invalid values in frs at iter {iter_num}")
            frs = torch.clamp(frs, min=-10.0, max=10.0)
            frs = torch.where(torch.isnan(frs), torch.zeros_like(frs), frs)
            frs = torch.where(torch.isinf(frs), torch.tensor(10.0, device=device), frs)

        if torch.isnan(frt).any() or torch.isinf(frt).any():
            print(f"[Warning] Invalid values in frt at iter {iter_num}")
            frt = torch.clamp(frt, min=-10.0, max=10.0)
            frt = torch.where(torch.isnan(frt), torch.zeros_like(frt), frt)
            frt = torch.where(torch.isinf(frt), torch.tensor(10.0, device=device), frt)

        # VAE-GAN生成掩码
        z_s, _ = vae_gan.encode(frs, fused_feat=fzs)
        gen_mask_s, vae_loss_s, _, _, _ = vae_gan(frs, fused_feat=fzs)
        gen_mask_s = torch.clamp(gen_mask_s, -5, 5)

        z_t, _ = vae_gan.encode(frt, fused_feat=fzt)
        gen_mask_t, vae_loss_t, _, _, _ = vae_gan(frt, fused_feat=fzt)
        gen_mask_t = torch.clamp(gen_mask_t, -5, 5)

        # 检查并处理VAE损失中的NaN
        if torch.isnan(vae_loss_s).any():
            print(f"[Warning] NaN in vae_loss_s at iter {iter_num}")
            vae_loss_s = torch.tensor(0.0, device=device)
        if torch.isnan(vae_loss_t).any():
            print(f"[Warning] NaN in vae_loss_t at iter {iter_num}")
            vae_loss_t = torch.tensor(0.0, device=device)

        vae_loss = (vae_loss_s + vae_loss_t) * 0.5

        if not torch.allclose(torch.sum(pred_t_soft, dim=1), torch.tensor(1.0, device=device), atol=1e-3):
            print(f"[Warning] EMA模型概率未归一化: sum={torch.sum(pred_t_soft, dim=1).mean().item()}")

        # 计算熵（提前计算，避免在函数内重复计算）
        entropy_s = -torch.sum(pred_s_soft * torch.log(pred_s_soft + 1e-8), dim=1)
        entropy_t = -torch.sum(pred_t_soft * torch.log(pred_t_soft + 1e-8), dim=1)

        # 动态阈值替换（传入iter_num和熵）
        pred_s_replaced_logits, mask_s, threshold_s = uncertainty_aware_replacement(
            pred_s,
            pred_s_soft,
            gen_mask_s,
            args,  # 含动态阈值参数
            iter_num,  # 当前迭代次数
            entropy=entropy_s  # 提前计算的熵
        )

        pred_s_replaced_soft = F.softmax(pred_s_replaced_logits, dim=1)
        pred_s_replaced_soft = torch.clamp(pred_s_replaced_soft, 1e-7, 1 - 1e-7)
        uncertainty_ratio_s = mask_s.sum().item() / mask_s.numel()

        with torch.no_grad():
            pred_t_replaced_logits, mask_t, threshold_t = uncertainty_aware_replacement(
                pred_t,
                pred_t_soft,
                gen_mask_t,
                args,
                iter_num,
                entropy=entropy_t
            )

            pred_t_replaced_soft = F.softmax(pred_t_replaced_logits, dim=1)
            pred_t_replaced_soft = torch.clamp(pred_t_replaced_soft, 1e-7, 1 - 1e-7)
            uncertainty_ratio_t = mask_t.sum().item() / mask_t.numel()

        # 检查输入图像
        if not (torch.all(volume >= 0) and torch.all(volume <= 1)):
            print(f"[Warning] 输入图像数据异常: min={volume.min()}, max={volume.max()}")
            volume = torch.clamp(volume, 0, 1)

        # GAN判别器输入
        input_gan_s = torch.cat([torch.clamp(pred_s_replaced_soft[:, 1:2, :, :], 0, 1), volume], dim=1)
        input_gan_t = torch.cat([torch.clamp(pred_t_replaced_soft[:, 1:2, :, :], 0, 1), volume], dim=1)

        # 检查判别器输入
        if torch.isnan(input_gan_s).any() or torch.isinf(input_gan_s).any():
            print(f"[Warning] Invalid values in input_gan_s at iter {iter_num}")
            input_gan_s = torch.clamp(input_gan_s, 0, 1)
            input_gan_s = torch.where(torch.isnan(input_gan_s), torch.zeros_like(input_gan_s), input_gan_s)

        if torch.isnan(input_gan_t).any() or torch.isinf(input_gan_t).any():
            print(f"[Warning] Invalid values in input_gan_t at iter {iter_num}")
            input_gan_t = torch.clamp(input_gan_t, 0, 1)
            input_gan_t = torch.where(torch.isnan(input_gan_t), torch.zeros_like(input_gan_t), input_gan_t)

        # GAN判别
        d_real = vae_gan.discriminator(
            torch.cat([label[:labeled_bs].float().unsqueeze(dim=1), volume[:labeled_bs]], dim=1))
        d_fake_s = vae_gan.discriminator(input_gan_s)
        d_fake_t = vae_gan.discriminator(input_gan_t)

        # 增强稳定性，添加更严格的范围限制
        d_real = torch.clamp(d_real, 1e-6, 1.0 - 1e-6)
        d_fake_s = torch.clamp(d_fake_s, 1e-6, 1.0 - 1e-6)
        d_fake_t = torch.clamp(d_fake_t, 1e-6, 1.0 - 1e-6)

        # 计算GAN损失
        gan_loss = -torch.log(d_real).mean() - torch.log(1 - d_fake_s).mean() - torch.log(1 - d_fake_t).mean()

        # 限制GAN损失范围
        gan_loss = torch.clamp(gan_loss, min=-100, max=100)

        # 不确定性修正损失
        loss_ce_s = F.cross_entropy(pred_s_replaced_logits[:labeled_bs], label[:labeled_bs].long())
        loss_dice_s = losses.DiceLoss(args.num_classes)(pred_s_replaced_soft[:labeled_bs],
                                                        label[:labeled_bs].unsqueeze(dim=1))

        with torch.no_grad():
            loss_ce_t = F.cross_entropy(pred_t_replaced_logits[:labeled_bs], label[:labeled_bs].long())
            loss_dice_t = losses.DiceLoss(args.num_classes)(pred_t_replaced_soft[:labeled_bs],
                                                            label[:labeled_bs].unsqueeze(dim=1))

        sup_loss2 = 0.5 * (loss_ce_s + loss_s_bdice2) + 0.5 * (loss_ce_t + loss_t_bdice2)

        # 一致性损失
        consistency_loss_u2 = torch.mean((pred_s_replaced_soft[labeled_bs:] - pred_t_replaced_soft[labeled_bs:]) ** 2)
        consistency_loss_l2 = torch.mean((pred_s_replaced_soft[:labeled_bs] - pred_t_replaced_soft[:labeled_bs]) ** 2)

    total_seg_loss = sup_loss + w_consistency * (consistency_loss_u + consistency_loss_l) + \
                     w_contrast * (loss_contrast_l + loss_contrast_u) + sup_loss2 + w_consistency * (
                                 consistency_loss_u2 + consistency_loss_l2)

    total_vae_gan_loss = vae_loss + gan_loss

    current_vae_weight = args.vae_gan_weight * min(1.0, iter_num / 1000)

    total_loss = total_seg_loss + current_vae_weight * total_vae_gan_loss

    total_loss = torch.clamp(total_loss, min=-1e4, max=1e4)

    # 优化
    optimizer_seg.zero_grad()
    optimizer_vae.zero_grad()

    with autocast():
        total_loss.backward()

    # 梯度裁剪
    generator_params = list(vae_gan.encoder.parameters()) + list(vae_gan.decoder.parameters())
    discriminator_params = vae_gan.discriminator.parameters()

    # 不同模块使用不同的梯度裁剪阈值
    torch.nn.utils.clip_grad_norm_(generator_params, max_norm=0.1)
    torch.nn.utils.clip_grad_norm_(discriminator_params, max_norm=0.1)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

    optimizer_seg.step()
    optimizer_vae.step()

    # 更新EMA模型
    update_ema_variables(model, ema_model, args.ema_decay, iter_num)

    if torch.isnan(vae_loss_s).any() or torch.isnan(vae_loss_t).any():
        print(f"[Warning] NaN in VAE loss at iter {iter_num}, skipping batch.")
        return (torch.tensor(0.0, device=device),) * 24

    return total_loss, total_seg_loss, total_vae_gan_loss, \
        sup_loss, loss_ce_sl, loss_dice_sl, loss_ce_tl, loss_dice_tl, consistency_loss_l, consistency_loss_u, loss_contrast_l, loss_contrast_u, \
        sup_loss2, loss_ce_s, loss_dice_s, loss_ce_t, loss_dice_t, vae_loss, vae_loss_s, vae_loss_t, gan_loss, consistency_loss_l2, consistency_loss_u2, \
        uncertainty_ratio_s, uncertainty_ratio_t, threshold_s, threshold_t


# 主训练函数
def train(args, snapshot_path):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global scaler
    scaler = GradScaler()

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    # 初始化模型
    model = create_model().to(device)
    ema_model = create_model(ema=True).to(device)
    vae_gan = create_vae_gan().to(device)

    # 检查模型参数中的NaN
    def check_nan_in_model(model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in {name}, reinitializing...")
                # 重新初始化有问题的参数
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.constant_(param, 0.0)

    # 优化器 - 使用更小的学习率和权重衰减
    optimizer_seg = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=1e-5)
    optimizer_vae = optim.Adam(
        vae_gan.parameters(),
        lr=args.base_lr * 0.0001,
        weight_decay=1e-5
    )

    # 数据集加载
    db_train = BaseDataSets(base_dir=args.root_path, split="train",
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size),
                            ])
                            )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total train is: {}, labeled_num is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs,
        unlabeled_idxs,
        args.batch_size,
        args.batch_size - args.labeled_bs
    )
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    # 日志和迭代设置
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iters/epoch".format(len(trainloader)))
    max_epoch = args.max_iterations // len(trainloader) + 1
    epoch_iter = len(trainloader)
    iterator = tqdm(range(max_epoch), ncols=70)
    iter_num = 0
    best_performance = 0.0

    for epoch_num in iterator:
        for batch in trainloader:
            # 检查输入数据是否有问题
            volume = batch['image'].to(device)
            label = batch['label'].to(device)

            volume = torch.clamp(volume, 0.0, 1.0)
            label = torch.clamp(label, 0, args.num_classes - 1)

            # 检查并处理NaN/Inf
            if torch.isnan(volume).any() or torch.isinf(volume).any():
                print(f"[Warning] Invalid values in input volume at iter {iter_num}")
                volume = torch.where(torch.isnan(volume) | torch.isinf(volume), torch.zeros_like(volume), volume)

            if torch.isnan(label).any() or torch.isinf(label).any():
                print(f"[Warning] Invalid values in label at iter {iter_num}")
                label = torch.where(torch.isnan(label) | torch.isinf(label), torch.zeros_like(label), label)

            # 执行训练步骤前检查模型参数
            check_nan_in_model(model)
            check_nan_in_model(vae_gan)

            # 执行训练步骤
            training_results = training_step(model, ema_model, vae_gan, volume, label, optimizer_seg, optimizer_vae, iter_num, args)

            # 解包训练结果
            (total_loss, total_seg_loss, total_vae_gan_loss,
             sup_loss, loss_ce_sl, loss_dice_sl, loss_ce_tl, loss_dice_tl,
             consistency_loss_l, consistency_loss_u, loss_contrast_l, loss_contrast_u,
             sup_loss2, loss_ce_s, loss_dice_s, loss_ce_t, loss_dice_t,
             vae_loss, vae_loss_s, vae_loss_t, gan_loss,
             consistency_loss_l2, consistency_loss_u2,
             uncertainty_ratio_s, uncertainty_ratio_t, threshold_s, threshold_t) = training_results

            # 检查是否有NaN损失，如果有则跳过本次迭代
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[Warning] NaN/Inf in total_loss at iter {iter_num}, skipping iteration")
                iter_num += 1
                continue

            # 学习率衰减
            lr_ = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            for pg in optimizer_seg.param_groups:
                pg['lr'] = lr_
            for pg in optimizer_vae.param_groups:
                pg['lr'] = lr_ * 0.1

            # 日志记录
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', total_loss, iter_num)
            writer.add_scalar('info/total_seg_loss', total_seg_loss, iter_num)
            writer.add_scalar('info/total_vae_gan_loss', total_vae_gan_loss, iter_num)

            writer.add_scalar('info/sup_loss', sup_loss, iter_num)
            writer.add_scalar('info/loss_ce_sl', loss_ce_sl, iter_num)
            writer.add_scalar('info/loss_dice_sl', loss_dice_sl, iter_num)
            writer.add_scalar('info/loss_ce_tl', loss_ce_tl, iter_num)
            writer.add_scalar('info/loss_dice_tl', loss_dice_tl, iter_num)
            writer.add_scalar('info/consistency_loss_l', loss_contrast_l, iter_num)
            writer.add_scalar('info/consistency_loss_u', loss_contrast_u, iter_num)

            # 不确定性修正
            writer.add_scalar('info/sup_loss2', sup_loss2, iter_num)
            writer.add_scalar('info/loss_ce_s', loss_ce_s, iter_num)
            writer.add_scalar('info/loss_dice_s', loss_dice_s, iter_num)
            writer.add_scalar('info/loss_ce_t', loss_ce_t, iter_num)
            writer.add_scalar('info/loss_dice_t', loss_dice_t, iter_num)
            writer.add_scalar('info/vae_loss', vae_loss, iter_num)
            writer.add_scalar('info/vae_loss_s', vae_loss_s, iter_num)
            writer.add_scalar('info/vae_loss_t', vae_loss_t, iter_num)
            writer.add_scalar('info/gan_loss', gan_loss, iter_num)
            writer.add_scalar('info/consistency_loss_l2', consistency_loss_l2, iter_num)
            writer.add_scalar('info/consistency_loss_u2', consistency_loss_u2, iter_num)

            writer.add_scalar('info/uncertainty_ratio_s', uncertainty_ratio_s, iter_num)
            writer.add_scalar('info/uncertainty_ratio_t', uncertainty_ratio_t, iter_num)

            writer.add_scalar('info/consistency_weight',
                              get_current_weight(args.consistency, iter_num, args.consistency_rampup), iter_num)
            writer.add_scalar('info/contrast_weight',
                              get_current_weight(args.contrast_weight, iter_num, args.contrast_rampup), iter_num)
            writer.add_scalar('info/dynamic_threshold_s', threshold_s, iter_num)
            writer.add_scalar('info/dynamic_threshold_t', threshold_t, iter_num)

            logging.info(
                f'epoch [{epoch_num + 1}/{max_epoch}]--iter {iter_num} : total_loss={total_loss.item():.4f}, total_seg_loss={total_seg_loss.item():.4f}, total_vae_gan_loss={total_vae_gan_loss.item():.4f}, '
                f'sup_loss={sup_loss.item():.4f}, sup_loss2={sup_loss2.item():.4f}, vae_loss={vae_loss.item():.4f}, gan_loss={gan_loss.item():.4f}'
            )

            # 可视化
            if iter_num % epoch_iter == 0:
                image = volume[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                with torch.no_grad():
                    pred, feat = model(image.unsqueeze(0))
                    pred_soft = F.softmax(pred, dim=1)
                    raw_outputs = torch.argmax(torch.softmax(pred, dim=1), dim=1, keepdim=True)

                # 可视化图像生成
                writer.add_image('train/Prediction', raw_outputs[0, ...] * 50, iter_num)
                writer.add_image('train/GroundTruth', labs, iter_num)

            # 验证和保存
            if iter_num > 0 and iter_num % epoch_iter == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=args.num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]  # dice
                mean_iou = np.mean(metric_list, axis=0)[1]
                mean_precision = np.mean(metric_list, axis=0)[2]
                mean_recall = np.mean(metric_list, axis=0)[3]
                mean_accuracy = np.mean(metric_list, axis=0)[4]
                mean_hd95 = np.mean(metric_list, axis=0)[5]
                mean_asd = np.mean(metric_list, axis=0)[6]

                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_iou', mean_iou, iter_num)
                writer.add_scalar('info/val_mean_precision', mean_precision, iter_num)
                writer.add_scalar('info/val_mean_recall', mean_recall, iter_num)
                writer.add_scalar('info/val_mean_accuracy', mean_accuracy, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                writer.add_scalar('info/val_mean_asd', mean_asd, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_iter_best = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_model_best = os.path.join(snapshot_path,
                                                   '{}_best_model.pth'.format(args.model))
                    # torch.save({
                    #     'model': model.state_dict(),
                    #     'vae_gan': vae_gan.state_dict(),
                    #     'optimizer_seg': optimizer_seg.state_dict(),
                    #     'optimizer_vae': optimizer_vae.state_dict()
                    # }, save_iter_best)

                    torch.save({
                        'model': model.state_dict(),
                        'vae_gan': vae_gan.state_dict()
                    }, save_model_best)
                    logging.info(f"Best model saved (Dice: {best_performance:.4f})")

                logging.info(
                    'epoch %d->iteration %d : mean_dice : %.4f | mean_iou : %.4f | mean_precision : %.4f | mean_recall : %.4f | mean_accuracy : %.4f | mean_hd95 : %.4f | mean_asd : %.4f'
                    % (epoch_num, iter_num, performance, mean_iou, mean_precision, mean_recall, mean_accuracy, mean_hd95, mean_asd))
                model.train()

            if iter_num >= args.max_iterations:
                iterator.close()
                break
            iter_num += 1

    # 最终保存
    final_save_path = os.path.join(snapshot_path, '{}_final.pth'.format(args.model))
    torch.save({
        'model': model.state_dict(),
        'vae_gan': vae_gan.state_dict()
    }, final_save_path)
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 保存路径
    snapshot_path = f"../model/{args.exp}_{args.labeled_num}_labeled/{args.model}/vae_gan"
    os.makedirs(snapshot_path, exist_ok=True)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', os.path.join(snapshot_path, 'code'), ignore=shutil.ignore_patterns('.git', '__pycache__'))

    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    scaler = GradScaler()
    train(args, snapshot_path)