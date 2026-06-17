import argparse
import os
import shutil
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
from medpy import metric
from tqdm import tqdm
from scipy.ndimage import binary_erosion, binary_dilation

# 导入网络工厂（根据实际情况调整）
# 其他导入
from networks.net_factory import net_factory

# DECSeg导入--初始化模型 net = net_factory处也要修改
# from networks.unet_decseg import net_factory

# CT
from networks.vision_transformer import SwinUnet as ViT_seg_CT
from config import get_config

# CCA-Seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg_CCA
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

"""
推理实验
"""

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='',  # 改
                    help='数据集根路径')  # 改
parser.add_argument('--dataset_type', type=str, default='3Dircadb',
                    choices=['ISIC2017', 'KvasirSEG', '3Dircadb', 'Bra2021'],
                    help='数据集类型（用于匹配文件名规则）')
parser.add_argument('--exp', type=str, default='MyModel', help='实验名称')  # 改
parser.add_argument('--model', type=str, default='unet_f', help='模型名称')  # 改
parser.add_argument('--model_pth', type=str,
                    default='',
                    help='模型名称')  # 改
parser.add_argument('--num_classes', type=int, default=2, help='网络输出通道数')
parser.add_argument('--labeled_num', type=int, default=10, help='带标签的数据量')
parser.add_argument('--patch_size', type=int, default=512, help='输入网络的图像尺寸')  # CT改224

# CT 模型参数
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# CCA-Seg 模型参数
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')  # R50-ViT-B_16
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')  #
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

# DECSeg 模型参数
parser.add_argument('--in_chns', type=int, default=1, help='input channel of network')
parser.add_argument('--sc', action='store_false', help='if Scale-enhanced consistency')
parser.add_argument('--cfa', action='store_false', help='if Cross-level Feature Aggregation')
parser.add_argument('--dcf', action='store_false', help='if Dual-scale Complementary Fusion')

args = parser.parse_args()
config = get_config(args)


def calculate_metric_percase(pred, gt):
    """计算单例的评估指标（使用代码二中的实现）"""
    pred = (pred > 0).astype(int)
    gt = (gt > 0).astype(int)

    print("pred shape:", pred.shape, "gt shape:", gt.shape)

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

    # HD95和ASD使用medpy计算
    try:
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
    except:
        hd95 = 0.0
        asd = 0.0

    return dice, iou, precision, recall, accuracy, hd95, asd


def get_file_path(folder, case, is_image=True, dataset_type=args.dataset_type):
    """
    多数据集通用的文件路径查找函数
    :param dataset_type: 数据集类型（ISIC2017/Bra2021）
    :param is_image: 是否为图像（True=图像，False=标签）
    """
    supported_formats = ['.png', '.jpg']  # 支持的文件格式

    # 1. 处理ISIC2017（无模态字段，直接用case拼接）
    if dataset_type == 'ISIC2017' or dataset_type == 'KvasirSEG' or dataset_type == '3Dircadb':
        for fmt in supported_formats:
            file_path = os.path.join(folder, f'{case}{fmt}')
            if os.path.exists(file_path):
                return file_path
        raise FileNotFoundError(f"ISIC数据集：{folder}中未找到{case}.png/.jpg")

    # 3. 处理BraTS2021（含模态字段_flair/_seg）
    elif dataset_type == 'Bra2021':
        parts = case.split('_')
        if len(parts) < 3:
            raise ValueError(f"BraTS2021的case格式错误：{case}（应为BraTS2021_<ID>_<slice>）")
        base_name = f"{parts[0]}_{parts[1]}"  # BraTS2021_00000
        slice_num = parts[2]  # 0072
        modality = "flair" if is_image else "seg"  # 图像=flair，标签=seg

        for fmt in supported_formats:
            file_name = f"{base_name}_{modality}_{slice_num}{fmt}"
            file_path = os.path.join(folder, file_name)
            if os.path.exists(file_path):
                return file_path
        raise FileNotFoundError(f"BraTS2021数据集：{folder}中未找到{base_name}_{modality}_{slice_num}.png/.jpg")

    else:
        raise ValueError(f"不支持的数据集类型：{dataset_type}（可选：ISIC2017/Bra2021）")


def infer_single_volume(case, net, test_save_path, FLAGS):
    """处理单张图像的推理过程"""
    # 读取图像和标签
    image_folder = os.path.join(FLAGS.root_path, 'Images')
    label_folder = os.path.join(FLAGS.root_path, 'Labels')
    img_path = get_file_path(
        folder=image_folder,
        case=case,
        is_image=True,
        dataset_type=FLAGS.dataset_type  # 新增：指定数据集类型
    )
    label_path = get_file_path(
        folder=label_folder,
        case=case,
        is_image=False,
        dataset_type=FLAGS.dataset_type  # 新增：指定数据集类型
    )

    # 读取图像并转为灰度（ISIC是RGB，模型通常用单通道输入）
    image = Image.open(img_path).convert('L')  # 转为灰度图
    label = Image.open(label_path).convert('L')  # 标签转为灰度图

    # 转为numpy数组
    image = np.array(image, dtype=np.float32) / 255.0  # 归一化到0-1
    label = np.array(label, dtype=np.float32)
    label = np.clip(label, 0, 255)  # 防止极端像素值
    label = (label > 128).astype(np.float32)  # 先转浮点确保精度
    label = np.round(label).astype(np.uint8)  # 明确取整

    # 获取原始尺寸
    x, y = image.shape[0], image.shape[1]

    image = np.expand_dims(image, axis=-1)  # 添加通道维度，形状变为 [H, W, 1]

    # 缩放到网络输入尺寸
    slice_resized = zoom(image, (FLAGS.patch_size / x, FLAGS.patch_size / y, 1), order=3)
    # DECSeg新增
    sized = round(FLAGS.patch_size * 0.5 / 32) * 32

    # 准备输入张量
    # input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()
    input_tensor = torch.from_numpy(slice_resized.transpose(2, 0, 1))  # [H,W,1]→[1,H,W]
    input_tensor = input_tensor.unsqueeze(0).float().cuda()  # →[1,1,512,512]（匹配模型输入）


    # 模型推理
    net.eval()
    with torch.no_grad():
        if FLAGS.model == 'unet_f':
            out_main = net(input_tensor)[0]
        elif FLAGS.model == 'unet_cct':
            out_main = net(input_tensor)[0]
        elif FLAGS.model == 'unet_urpc':
            out_main = net(input_tensor)[0]
        elif FLAGS.model == 'unet_bcp':
            out_main = net(input_tensor)[0]
        elif FLAGS.model == 'unet_cmmt':
            out_main = net(input_tensor)[0]
        elif FLAGS.model == 'unet_CGS':
            out_main = net(input_tensor)[0]
        elif FLAGS.model == 'unet_decseg':
            out_main = net(input_tensor, input_d_tensor)[0]
        else:
            out_main = net(input_tensor)
            # out_main = net(input_tensor)[0]  # CCA-Seg
        pred_soft = torch.softmax(out_main, dim=1)
        prediction = torch.argmax(pred_soft, dim=1).squeeze(0).cpu().numpy()

    # 缩放回原始尺寸
    prediction = zoom(prediction, (x / FLAGS.patch_size, y / FLAGS.patch_size), order=0)

    # 计算指标（针对每个类别）
    metric_list = []
    for i in range(1, FLAGS.num_classes):  # 从1开始因为0是背景
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # 保存结果为PNG
    # 原图需要归一化到0-255
    # 去除图像的通道维度，确保形状为(H, W)
    img_save = (image.squeeze(-1) * 255).astype(np.uint8)
    # 预测结果和标签转为0-255（便于可视化）
    pred_save = (prediction * 255).astype(np.uint8)
    label_save = (label * 255).astype(np.uint8)

    Image.fromarray(img_save).save(os.path.join(test_save_path, f'{case}_img.png'))
    Image.fromarray(pred_save).save(os.path.join(test_save_path, f'{case}_pred.png'))
    Image.fromarray(label_save).save(os.path.join(test_save_path, f'{case}_gt.png'))

    return metric_list


def Inference(FLAGS):
    """主推理函数"""
    # 读取验证集列表 val.txt/val.txt
    with open(os.path.join(FLAGS.root_path, 'val.txt'), 'r') as f:  # 改
        image_list = f.readlines()
    # 处理文件名，去除换行符和可能的扩展名
    image_list = sorted([item.strip() for item in image_list if item.strip()])

    # 设置模型和保存路径
    snapshot_path = FLAGS.model_pth
    test_save_path = f""  # 改

    # 创建保存目录
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path, exist_ok=True)

    # 初始化模型
    if FLAGS.exp == 'Bra2021/CT':  # 改
        net = ViT_seg_CT(config, img_size=args.patch_size, num_classes=args.num_classes)
    elif FLAGS.exp == '3Dircadb/CCA-Seg':  # 改
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        net = ViT_seg_CCA(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
    else:
        # 其他
        net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
        # DECSeg
        # net = net_factory(args.num_classes, args.in_chns, SC=args.sc, CFA=args.cfa, DCF=args.dcf)

    save_model_path = os.path.join(snapshot_path, f'{FLAGS.model}_best_model.pth')
    # 检查模型权重文件是否存在
    if not os.path.exists(save_model_path):
        raise FileNotFoundError(f"模型权重文件不存在：{save_model_path}")

    # MyModel
    # 1. 加载完整的权重字典（包含model、vae_gan、optimizer等）
    weight_dict = torch.load(save_model_path)
    # 2. 打印权重字典的键，帮助确认结构（可选，调试用）
    print("权重文件包含的键：", weight_dict.keys())
    # 3. 提取模型参数（核心：只取"model"键对应的参数，忽略vae_gan和optimizer）
    # （如果训练时保存的模型参数键不是"model"，需替换为实际键名，如"net"）

    if FLAGS.exp == 'Bra2021/Diffrect':  # 改
        model_state = weight_dict.get("state_dict", weight_dict)
    elif FLAGS.exp == 'Bra2021/CGS':  # 改
        model_state = weight_dict.get("state_dict", weight_dict)
    elif FLAGS.exp == 'KvasirSEG/DECSeg':  # 改
        model_state = weight_dict.get("state", weight_dict)  # 若没有"model"键，用整个字典（兼容不同保存方式）
    else:
        model_state = weight_dict.get("model", weight_dict)  # 若没有"model"键，用整个字典（兼容不同保存方式）

    # 4. 过滤模型不需要的键（若model_state中仍有意外键，可手动删除）
    unexpected_keys = ["vae_gan", "optimizer_seg", "optimizer_vae"]
    for key in unexpected_keys:
        if key in model_state:
            del model_state[key]
    # 5. 加载模型参数，strict=False：忽略模型中不存在的参数（避免冗余参数报错）
    net.load_state_dict(model_state, strict=False)

    print(f"已从 {save_model_path} 加载模型权重")
    net.cuda()
    net.eval()

    # 计算所有样本的平均指标
    total_metrics = np.zeros((FLAGS.num_classes - 1, 7))  # 排除背景类，每个类有7个指标
    for case in tqdm(image_list, desc="推理进度"):
        metric_list = infer_single_volume(case, net, test_save_path, FLAGS)
        total_metrics += np.array(metric_list)

    # 计算平均值
    avg_metrics = total_metrics / len(image_list)

    # 打印每个类别的平均指标
    # for i in range(FLAGS.num_classes - 1):
    #     print(f"类别 {i + 1} 平均指标:")
    #     print(f"Dice: {avg_metrics[i, 0]:.4f}, IoU: {avg_metrics[i, 1]:.4f}, "
    #           f"Precision: {avg_metrics[i, 2]:.4f}, Recall: {avg_metrics[i, 3]:.4f}, "
    #           f"Accuracy: {avg_metrics[i, 4]:.4f}, HD95: {avg_metrics[i, 5]:.4f}, "
    #           f"ASD: {avg_metrics[i, 6]:.4f}")

    return avg_metrics


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metrics = Inference(FLAGS)
    # 计算所有类别的总体平均
    overall_avg = np.mean(metrics, axis=0)
    print("\n所有类别总体平均指标:")
    print(f"Dice: {overall_avg[0]:.4f}, IoU: {overall_avg[1]:.4f}, "
          f"Precision: {overall_avg[2]:.4f}, Recall: {overall_avg[3]:.4f}, "
          f"Accuracy: {overall_avg[4]:.4f}, HD95: {overall_avg[5]:.4f}, "
          f"ASD: {overall_avg[6]:.4f}")
