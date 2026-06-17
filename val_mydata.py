import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

"""
/media/dell/codes/Dqw/Paper/contrast/SSL4MIS/model/ISIC2017/Mean_Teacher_10_labeled/unet/
├── iter_1000_dice_0.7890.pth
├── iter_2000_dice_0.8901.pth
└── unet_best_model.pth
"""

# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         return dice, hd95
#     else:
#         return 0, 0
def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(int)
    gt = (gt > 0).astype(int)

    # 基础统计量计算
    tp = np.sum((pred == 1) & (gt == 1))  # True Positive
    tn = np.sum((pred == 0) & (gt == 0))  # True Negative
    fp = np.sum((pred == 1) & (gt == 0))  # False Positive
    fn = np.sum((pred == 0) & (gt == 1))  # False Negative

    # 计算指标
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0

    # HD95需要保持medpy计算
    try:
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
    except:
        hd95 = 0.0
        asd = 0.0

    return dice, iou, precision, recall, accuracy, hd95, asd


def test_single_volume(image, label, net, classes, patch_size=[512, 512]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # ==== 修改部分：移除切片循环，直接处理整张图像 ====
    x, y = image.shape[0], image.shape[1]

    # 调整图像尺寸到模型输入大小
    slice_resized = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
    input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()

    # 前向推理
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input_tensor), dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()

    # 还原到原始尺寸
    prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)

    # 计算指标
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[512, 512]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # ==== 修改部分：移除切片循环 ====
    x, y = image.shape[0], image.shape[1]

    # 调整图像尺寸到模型输入大小
    slice_resized = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
    input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()

    # 前向推理
    net.eval()
    with torch.no_grad():
        output_main, _, _, _ = net(input_tensor)
        prediction = torch.argmax(torch.softmax(output_main, dim=1), dim=1).squeeze(0)
        prediction = prediction.cpu().detach().numpy()

    # 还原到原始尺寸
    prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)

    # 计算指标
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

