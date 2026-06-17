import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GATLayer(nn.Module):
    """图注意力层（保持不变，用于特征增强）"""
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):  # 0.1, 0.2
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

    @staticmethod
    def compute_attention(h, a, leakyrelu, dropout_layer):
        N = h.size()[0]
        h1 = h.unsqueeze(1).expand(-1, N, -1)
        h2 = h.unsqueeze(0).expand(N, -1, -1)
        a_input = torch.cat([h1, h2], dim=2)
        e = leakyrelu(a(a_input).squeeze(2))
        attention = F.softmax(e, dim=1)
        attention = dropout_layer(attention)
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)

    def forward(self, x):
        h = self.W(x)
        if self.training:
            h_prime = checkpoint(GATLayer.compute_attention, h, self.a, self.leakyrelu, self.dropout_layer)
        else:
            h_prime = GATLayer.compute_attention(h, self.a, self.leakyrelu, self.dropout_layer)
        return h_prime


class InfoNCEWithDualViewLoss(nn.Module):
    """基于双视图（s/t）的InfoNCE对比损失，按需求定义正/负样本"""
    def __init__(self, temperature=0.1, gat_in_dim=32, gat_out_dim=128, num_samples=256, dropout=0.1):
        super(InfoNCEWithDualViewLoss, self).__init__()
        self.temperature = temperature  # InfoNCE温度参数
        self.num_samples = num_samples  # 每个样本的采样点数
        self.gat = GATLayer(gat_in_dim, gat_out_dim, dropout=dropout)  # 图注意力层用于特征增强

    def entropy_based_sampling(self, features):
        """基于熵的特征采样（保持不变）"""
        b, c, h, w = features.shape
        features_reshaped = features.view(b, c, -1)  # (B, C, H*W)
        # 计算每个空间位置的熵（用sigmoid归一化后计算）
        prob = torch.sigmoid(features_reshaped)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)  # (B, H*W)
        # 按熵采样（取熵最高的点）
        samples = []
        indices = []
        for i in range(b):
            _, topk_indices = torch.topk(entropy[i], k=self.num_samples)
            sample = features_reshaped[i, :, topk_indices].transpose(0, 1)  # (num_samples, C)
            samples.append(sample)
            indices.append(topk_indices)
        return torch.stack(samples, dim=0), indices  # (B, num_samples, C), (B, num_samples)

    def get_corresponding_samples(self, features, indices):
        """根据索引获取对应位置的样本（保持不变）"""
        b, c, h, w = features.shape
        features_reshaped = features.view(b, c, -1)
        corresponding_samples = []
        for i in range(b):
            sample = features_reshaped[i, :, indices[i]].transpose(0, 1)
            corresponding_samples.append(sample)
        return torch.stack(corresponding_samples, dim=0)  # (B, num_samples, C)

    def get_label_samples(self, labels, indices):
        """获取采样点对应的标签（保持不变）"""
        b, h, w = labels.shape
        labels_reshaped = labels.view(b, -1)
        label_samples = []
        for i in range(b):
            label_samples.append(labels_reshaped[i, indices[i]])
        return torch.stack(label_samples, dim=0)  # (B, num_samples)

    def info_nce_loss(self, anchors, candidates, pos_mask):
        """
        计算InfoNCE损失
        Args:
            anchors: 锚点特征 (N, D)，N为锚点数量
            candidates: 候选特征 (M, D)，M为候选样本数量（包含正/负样本）
            pos_mask: 正样本掩码 (N, M)，1表示候选样本是锚点的正样本，0为负样本
        """
        # 计算锚点与候选样本的相似度（点积）
        sim = torch.matmul(anchors, candidates.T)  # (N, M)
        sim = sim / self.temperature  # 除以温度参数

        # 计算分子：所有正样本的exp(sim)之和
        exp_sim = torch.exp(sim)
        pos_sum = torch.sum(exp_sim * pos_mask, dim=1, keepdim=True)  # (N, 1)

        # 计算分母：所有候选样本的exp(sim)之和（包含正样本）
        total_sum = torch.sum(exp_sim, dim=1, keepdim=True)  # (N, 1)

        # InfoNCE损失：平均每个锚点的负对数概率
        loss = -torch.log(pos_sum / total_sum + 1e-8).mean()
        return loss

    def forward(self, feat_s, feat_t, labels=None, pseudo_labels=None):
        """
        Args:
            feat_s: 学生特征 (B, C, H, W)
            feat_t: 教师特征 (B, C, H, W)
            labels: 真实标签（用于s视图）(B, H, W)
            pseudo_labels: 伪标签（用于t视图）(B, H, W)
        """
        device = feat_s.device
        b = feat_s.shape[0]

        # 1. 采样：对学生特征采样，同步获取教师特征对应位置样本
        s_samples, indices = self.entropy_based_sampling(feat_s)  # (B, num_samples, C)
        t_samples = self.get_corresponding_samples(feat_t, indices)  # (B, num_samples, C)

        # 2. 获取标签：s视图用真实标签，t视图用伪标签
        s_labels = self.get_label_samples(labels, indices) if labels is not None else None  # (B, num_samples)
        t_labels = self.get_label_samples(pseudo_labels, indices) if pseudo_labels is not None else None  # (B, num_samples)

        # 3. GAT特征增强 + L2归一化
        s_flat = s_samples.view(-1, s_samples.shape[2])  # (B*num_samples, C)
        t_flat = t_samples.view(-1, t_samples.shape[2])  # (B*num_samples, C)

        # 使用GAT
        s_gat = F.normalize(self.gat(s_flat), p=2, dim=1)  # (B*num_samples, gat_out_dim)
        t_gat = F.normalize(self.gat(t_flat), p=2, dim=1)  # (B*num_samples, gat_out_dim)

        # w/o GAT
        # s_gat = F.normalize(s_flat, p=2, dim=1)  # (B*num_samples, gat_out_dim)
        # t_gat = F.normalize(t_flat, p=2, dim=1)  # (B*num_samples, gat_out_dim)

        # 4. 构建双视图的锚点和候选集（按批次处理）
        total_loss = 0.0
        for i in range(b):
            # 当前批次数据：s视图和t视图的特征及标签
            s_feat = s_gat[i*self.num_samples : (i+1)*self.num_samples]  # (num_samples, D)
            t_feat = t_gat[i*self.num_samples : (i+1)*self.num_samples]  # (num_samples, D)
            s_label = s_labels[i]  # (num_samples,)
            t_label = t_labels[i]  # (num_samples,)

            # 候选集：合并s和t的特征（所有可能的对比样本）
            candidates = torch.cat([s_feat, t_feat], dim=0)  # (2*num_samples, D)
            num_candidates = candidates.shape[0]

            # -------------------------- s视图损失（以s_feat为锚点）--------------------------
            # 正样本掩码：
            # 1. 与t_feat同位置的样本（s的第k个对应t的第k个）
            pos_mask_s_pos = torch.zeros((self.num_samples, num_candidates), device=device)
            pos_mask_s_pos[:, self.num_samples : ] = torch.eye(self.num_samples, device=device)  # t_feat区域的对角线为1

            # 2. 与s_label类别相同的样本（包括s_feat和t_feat中）
            pos_mask_s_label = torch.eq(
                s_label.unsqueeze(1),  # (num_samples, 1)
                torch.cat([s_label, t_label], dim=0).unsqueeze(0)  # (1, 2*num_samples)
            ).float().to(device)  # (num_samples, 2*num_samples)

            # 合并正样本掩码（逻辑或：满足任一条件即为正样本）
            pos_mask_s = (pos_mask_s_pos + pos_mask_s_label).clamp(0, 1)  # 避免重复计数（最多为1）

            # 排除自身作为正样本（s_feat中的第k个不与自身对比）
            pos_mask_s[:, :self.num_samples] -= torch.eye(self.num_samples, device=device)
            pos_mask_s = pos_mask_s.clamp(0, 1)

            # 计算s视图InfoNCE损失
            loss_s = self.info_nce_loss(anchors=s_feat, candidates=candidates, pos_mask=pos_mask_s)

            # -------------------------- t视图损失（以t_feat为锚点）--------------------------
            # 正样本掩码：
            # 1. 与s_feat同位置的样本（t的第k个对应s的第k个）
            pos_mask_t_pos = torch.zeros((self.num_samples, num_candidates), device=device)
            pos_mask_t_pos[:, :self.num_samples] = torch.eye(self.num_samples, device=device)  # s_feat区域的对角线为1

            # 2. 与t_label类别相同的样本（包括s_feat和t_feat中）
            pos_mask_t_label = torch.eq(
                t_label.unsqueeze(1),  # (num_samples, 1)
                torch.cat([s_label, t_label], dim=0).unsqueeze(0)  # (1, 2*num_samples)
            ).float().to(device)  # (num_samples, 2*num_samples)

            # 合并正样本掩码
            pos_mask_t = (pos_mask_t_pos + pos_mask_t_label).clamp(0, 1)

            # 排除自身作为正样本（t_feat中的第k个不与自身对比）
            pos_mask_t[:, self.num_samples:] -= torch.eye(self.num_samples, device=device)
            pos_mask_t = pos_mask_t.clamp(0, 1)

            # 计算t视图InfoNCE损失
            loss_t = self.info_nce_loss(anchors=t_feat, candidates=candidates, pos_mask=pos_mask_t)

            # 累加当前批次损失
            total_loss += (loss_s + loss_t) / 2  # 双视图损失平均

        return total_loss / b  # 平均所有批次损失


# 使用示例
if __name__ == "__main__":
    batch_size = 2
    feat_s = torch.randn(batch_size, 16, 512, 512)  # 学生特征
    feat_t = torch.randn(batch_size, 16, 512, 512)  # 教师特征
    labels = torch.randint(0, 10, (batch_size, 512, 512))  # 真实标签（s视图用）
    pseudo_labels = torch.randint(0, 10, (batch_size, 512, 512))  # 伪标签（t视图用）

    # 初始化损失函数
    loss_fn = InfoNCEWithDualViewLoss(temperature=0.07, num_samples=1024)

    # 计算损失
    loss = loss_fn(feat_s, feat_t, labels=labels, pseudo_labels=pseudo_labels)
    print(f"Dual-view InfoNCE loss: {loss.item()}")
