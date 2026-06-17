import contextlib

import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def loss_vae_function(recon_x, target_x, mask, mu, logvar):
    eps = 1e-6
    # 收紧logvar范围，避免极端值
    logvar = torch.clamp(logvar, min=-5, max=5)  # 原范围[-10,2]可能过宽
    var = torch.exp(logvar).clamp(min=eps, max=1e3)  # 限制方差上限，避免过大
    # 限制mu的范围，避免mu²过大
    mu = torch.clamp(mu, min=-5, max=5)
    recon_x = recon_x.clamp(0, 1)

    # 增加通道维度
    if target_x.dim() == 3:
        target_x = target_x.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
    mask = mask.unsqueeze(1) if mask.dim() == 3 else mask  # 同样处理mask

    if mask.sum() == 0:
        return torch.tensor(0.0, device=mu.device), torch.tensor(0.0, device=mu.device), torch.tensor(0.0,
                                                                                                      device=mu.device)

    valid_elements = mask.sum() + eps
    MSE = F.mse_loss(recon_x * mask, target_x * mask, reduction='sum') / valid_elements

    # 稳定KL散度计算（避免var过小）
    kl_element = 1 + logvar - mu ** 2 - var
    kl_element = torch.clamp(kl_element, min=-100, max=100)  # 限制单个元素范围
    KLD = -0.5 * torch.sum(kl_element) / valid_elements

    # 检查是否有NaN
    if torch.isnan(KLD):
        KLD = torch.tensor(0.0, device=mu.device)  # 紧急规避NaN

    total_loss = MSE + 1e-4 * KLD
    # print(f"[DEBUG] recon_x: {recon_x.shape}, target_x: {target_x.shape}, mask: {mask.shape}")
    # print(f"[DEBUG] MSE={MSE.item():.4f}, KLD={KLD.item():.4f}, mask_sum={mask.sum().item()}")
    if torch.isnan(MSE) or torch.isinf(MSE):
        print(f"[NaN] MSE={MSE}, recon_x={recon_x.min().item()}, {recon_x.max().item()}")
    if torch.isnan(KLD) or torch.isinf(KLD):
        print(f"[NaN] KLD={KLD}, mu={mu.min().item()}, {mu.max().item()}")

    return total_loss, MSE, KLD


def vae_loss(recon_x, target_x, mask, mu, logvar):
    eps = 1e-8
    logvar = torch.clamp(logvar, min=-5, max=5)
    # 计算有效元素数（非零mask区域）
    valid_elements = mask.sum() + eps

    # 限制放大倍数
    scale = torch.clamp(mask.numel() / valid_elements, max=1000.0)
    # 重构损失
    recon_loss = F.mse_loss(recon_x * mask, target_x * mask, reduction='sum') / valid_elements

    # KL损失
    kl_element = 1 + logvar - mu.pow(2) - logvar.exp()
    kl_element = torch.clamp(kl_element, max=0)
    kl_loss = -0.5 * torch.sum(kl_element) / valid_elements  # 归一化
    kl_loss = torch.clamp(kl_loss, max=1.0)

    # 动态权重平衡
    kl_weight = torch.clamp(0.01 * kl_loss.detach(), min=eps, max=0.1)
    total_loss = 0.5 * recon_loss * scale + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        # 二分类特殊处理
        if self.n_classes == 1:
            inputs = inputs.sigmoid()
            target = target.float()
            intersection = (inputs * target).sum()
            union = inputs.sum() + target.sum()
            return 1 - (2. * intersection + 1e-5) / (union + 1e-5)

        # 多分类处理
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

# 新增 VAT2d（VAT：virtual adversarial training,虚拟对抗训练）
class softDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(softDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice
        return loss / self.n_classes

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    # pdb.set_trace()
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8  ###2-p length of vector
    return d


class VAT2d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT2d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(2)  # 2 is the number of classes

    def forward(self, model, x):  # 是否加入ema_model
        with torch.no_grad():
            pred = F.softmax(model(x)[0], dim=1)

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                # pred_hat = ema_model(x + self.xi * d)[0]
                pred_hat = model(x + self.xi * d)[0]
                logp_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(logp_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            # pred_hat = ema_model(x + r_adv)[0]
            pred_hat = model(x + r_adv)[0]
            logp_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(logp_hat, pred)
        return lds


class BoundaryAwareDiceLoss(torch.nn.Module):
    def __init__(self, num_classes, boundary_width=1, boundary_weight=2.0, smooth=1e-5):
        """
        带边界感知的Dice损失
        :param num_classes: 类别数（含背景）
        :param boundary_width: 边界宽度（像素数），通常设为1或2
        :param boundary_weight: 边界区域的权重（应大于1，突出边界重要性）
        :param smooth: 平滑项，避免除零
        """
        super().__init__()
        self.num_classes = num_classes
        self.boundary_width = boundary_width
        self.boundary_weight = boundary_weight
        self.smooth = smooth
        # 定义腐蚀操作的核（用于提取边界）
        self.kernel = torch.ones(
            (1, 1, 2 * boundary_width + 1, 2 * boundary_width + 1),
            dtype=torch.float32
        )

    def get_boundary_mask(self, mask):
        """
        从目标掩码中提取边界区域
        :param mask: 目标掩码，形状为 (B, C, H, W)，其中C为类别数（独热编码）
        :return: 边界掩码（1表示边界像素，0表示非边界），形状同mask
        """
        B, C, H, W = mask.shape
        boundary_masks = []

        for c in range(C):
            # 单类别掩码 (B, 1, H, W)
            single_mask = mask[:, c:c + 1, ...].float()
            # 腐蚀操作：缩小目标区域，得到内部区域
            eroded = F.conv2d(
                single_mask,
                self.kernel.to(mask.device),
                padding=self.boundary_width,
                groups=1
            )
            # 原始掩码 - 腐蚀后的掩码 = 边界区域（目标边缘）
            boundary = (single_mask - eroded) > 0.5  # 二值化
            boundary_masks.append(boundary.float())

        # 拼接所有类别的边界掩码 (B, C, H, W)
        return torch.cat(boundary_masks, dim=1)

    def forward(self, pred, target):
        """
        计算带边界感知的Dice损失
        :param pred: 模型输出的概率图，形状为 (B, C, H, W)（经softmax处理）
        :param target: 目标掩码，形状为 (B, C, H, W)（独热编码）
        :return: 加权后的总损失
        """
        # 1. 提取边界掩码
        boundary_mask = self.get_boundary_mask(target)
        # 非边界掩码 = 目标掩码 - 边界掩码（仅保留内部区域）
        inner_mask = target - boundary_mask

        # 2. 计算整体Dice损失（所有目标像素）
        intersection = torch.sum(pred * target, dim=(2, 3))
        union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
        dice_overall = (2. * intersection + self.smooth) / (union + self.smooth)

        # 3. 计算边界区域Dice损失（仅边界像素）
        intersection_boundary = torch.sum(pred * boundary_mask, dim=(2, 3))
        union_boundary = torch.sum(pred * boundary_mask, dim=(2, 3)) + torch.sum(boundary_mask, dim=(2, 3))
        dice_boundary = (2. * intersection_boundary + self.smooth) / (union_boundary + self.smooth)

        # 4. 计算内部区域Dice损失（非边界像素）
        intersection_inner = torch.sum(pred * inner_mask, dim=(2, 3))
        union_inner = torch.sum(pred * inner_mask, dim=(2, 3)) + torch.sum(inner_mask, dim=(2, 3))
        dice_inner = (2. * intersection_inner + self.smooth) / (union_inner + self.smooth)

        # 5. 加权融合：边界损失权重更高，增强边界关注度
        loss = 1 - (
                0.3 * dice_overall +  # 整体损失占比30%
                0.5 * self.boundary_weight * dice_boundary +  # 边界损失占比50%，并加权
                0.2 * dice_inner  # 内部损失占比20%
        )

        # 对所有类别和批次取平均
        return loss.mean()
