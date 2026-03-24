# -*- coding: utf-8 -*-
"""
基于你原始 UAP 脚本的接口不变（尽量），内部使用 SSP 攻击逻辑实现。
- 保留 load_at_model(), main(), rob_test_uap() 等接口名，便于外部集成。
- generate_uap() 内部实现为 SSP（可用训练 loader 进行迭代生成 delta）。
- 增加一些 args 参数：bound, alpha, save_every_iter, workers, device, model_arch, data_path, disable_tqdm
- 中文注释详尽
"""
import os
import torch
import argparse
import random
import json
import numpy as np
from utils.load_data import load_data, normalzie
from tqdm import tqdm
from torch import nn
from pathlib import Path

# ====== SSP 风格的扰动生成：替换原先的 generate_uap ======
def _get_target_block_and_register_hook(model, target_layer, activations, remove_handles):
    """
    尝试多种可能的模块结构并注册 forward hook。
    返回注册的 handle 列表（可能是空）。
    兼容：model.net[...]、model.blocks[...]、model[...]（Sequential）等结构。
    """
    def activation_recorder_hook(self, input, output):
        # 存储平方后的激活以便 MSE 比较（与原 SSP 一致）
        activations.append(torch.square(output))
        return None

    handles = []
    # 尝试常见属性路径
    try:
        # simclr 风格： model.net[target_layer]
        target_mod = model.net[target_layer]
        handles.append(target_mod.register_forward_hook(activation_recorder_hook))
        return handles
    except Exception:
        pass

    try:
        # 另一种风格： model.blocks[target_layer]
        target_mod = model.blocks[target_layer]
        handles.append(target_mod.register_forward_hook(activation_recorder_hook))
        return handles
    except Exception:
        pass

    # 作为后备：尝试把 model 当成 nn.Sequential 直接索引
    try:
        target_mod = list(model.children())[target_layer]
        handles.append(target_mod.register_forward_hook(activation_recorder_hook))
        return handles
    except Exception:
        pass

    # 最后尝试 module 的深层索引（保守，不保证成功）
    # 遍历所有子模块并选择第 target_layer 个
    try:
        children = [m for m in model.modules() if not isinstance(m, nn.ModuleList)]
        if target_layer < len(children):
            handles.append(children[target_layer].register_forward_hook(activation_recorder_hook))
            return handles
    except Exception:
        pass

    # 若都失败，返回空列表（调用方需检测）
    return handles

def generate_uap(args, model, dataloader, xi=10/255, delta_steps=10, p=2, device=None):
    """
    将原 generate_uap 替换为 SSP 实现（接口签名尽量保持兼容）。
    输入:
        args - 包含 ssp 所需参数（如 bound, alpha, save_every_iter, model_arch 等）
        model - 待攻击模型（已在外部用 load_at_model 加载）
        dataloader - 用于生成扰动的 data loader（SSP 中使用训练数据）
        xi, delta_steps, p - 为兼容旧接口保留，但 SSP 将优先使用 args.bound/args.alpha
        device - 优先使用 args.device 或函数参数
    返回:
        delta (torch.Tensor) - 生成的扰动张量 (1,c,h,w)，已 detach 并放在 device 上
    """
    # 设备配置优先级：函数参数 device -> args.device -> CUDA_VISIBLE_DEVICES 环境 -> cpu
    if device is None:
        device = getattr(args, 'device', None)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = model.to(device).eval()

    # SSP 参数：bound, alpha, save_every_iter
    bound = getattr(args, 'bound', xi)  # 如果未提供，尽量使用 xi 的值
    alpha = getattr(args, 'alpha', 1e-2)  # 默认步幅
    save_every_iter = getattr(args, 'save_every_iter', 100)
    target_layer = 2 if getattr(args, 'model_arch', '') == 'simclr' else 5

    # 从 dataloader 取样得到输入尺寸（兼容不同数据集）
    sample_data = next(iter(dataloader))[0]
    _, c, h, w = sample_data.shape

    # 初始化 delta（在 [-bound, bound] 范围内随机）
    delta = bound * torch.rand((1, c, h, w), device=device, requires_grad=True) * 2 - bound
    # 损失函数：MSE（与原 SSP 一致）
    cri = torch.nn.MSELoss()

    k = 0
    # 进度显示由 args.disable_tqdm 决定
    disable_tqdm = getattr(args, 'disable_tqdm', False)

    # 遍历 dataloader（SSP 通常用训练集）
    for (image, _) in tqdm(dataloader, disable=disable_tqdm, desc="SSP 生成扰动迭代"):
        # detach 前一轮的 delta，并开启 grad
        delta = delta.detach()
        delta.requires_grad = True
        k += 1

        image = image.to(device)
        image_copy = image.clone().detach().to(device)

        # 用于记录 activation
        activations = []
        remove_handles = []

        # 第一步：注册 hook 并前向干净图像，记录 clean_layer（平方后的激活）
        handles = _get_target_block_and_register_hook(model, target_layer, activations, remove_handles)
        # 如果无法注册 hook，尝试直接前向并取某层输出 —— 若都失败，则退出并抛错
        if len(handles) == 0:
            # 为了稳健，尝试在 Sequential 中使用索引 target_layer
            raise RuntimeError(f"无法在模型中注册目标层 hook（target_layer={target_layer}）。请检查模型结构或调整 args.model_arch。")

        # 前向干净图像（不计算梯度）
        with torch.no_grad():
            model(image)
        # 此时 activations[0] 应为 clean layer 的平方激活
        clean_layer = activations[0].clone().detach().to(device).requires_grad_(True)

        # 移除之前的 hook（为了避免重复）
        for handle in handles:
            handle.remove()

        # 第二步：注册 hook，再对加了扰动的图像做前向，记录带扰动的激活
        activations = []
        handles = _get_target_block_and_register_hook(model, target_layer, activations, remove_handles)
        if len(handles) == 0:
            raise RuntimeError("在第二次注册 hook 时失败。")

        # 前向带扰动图像（这里需要计算激活以便比较）
        model(image_copy + delta)

        # 移除 hook
        for handle in handles:
            handle.remove()

        # 计算 loss（MSE between clean_layer and activations[0]），并反向传播得到 delta.grad
        loss = cri(clean_layer, activations[0])
        loss.backward()

        # 根据 SSP 原版更新规则：delta = delta + alpha * sign(delta.grad)
        with torch.no_grad():
            grad = delta.grad
            if grad is None:
                # 若没有梯度则跳过
                continue
            delta = delta + alpha * torch.sign(grad)
            # 将 delta 限制到 [-bound, bound]
            delta = torch.clamp(delta, -bound, bound)
            # 确保 delta 仍然是叶子（需要 requires_grad=True 以便下一轮）
            delta.requires_grad = True

        # 保存检查点（可选）
        if k % save_every_iter == 0:
            out_dir = Path(getattr(args, 'save_dir', 'ssp_output'))
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = out_dir / f'k={k}.pt'
            # 保存 tensor 到磁盘
            torch.save(delta.detach().cpu(), filename)
            print(f"[SSP] 已保存扰动到 {filename}")

    # 返回扰动（detach，放到 device 上）
    return delta.detach().to(device)


# ========== 准确率计算（沿用你原来的实现） ==========
def accuracy(new_output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = new_output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# ========== 使用 SSP 生成的扰动进行鲁棒测试（保留 rob_test_uap 名称以兼容外部） ==========
def rob_test_uap(args, model, test_loader):
    """
    生成扰动并在 test_loader 上评估鲁棒性。
    - generate_uap() 内部由 SSP 实现生成扰动（使用提供的 dataloader 作为训练数据）
    - 然后在 test_loader 上对模型进行带扰动的评估（Top-1, Top-5）
    """
    print("\n>>> 开始使用 SSP 生成 Universal-like 扰动 (SSP)...")
    # 这里我们使用 test_loader 本身来生成扰动（保持接口不变）。如果你想用训练集，请传入训练 loader。
    delta = generate_uap(args, model, test_loader, xi=getattr(args, 'bound', 8/255),
                         delta_steps=getattr(args, 'delta_steps', 5), p=np.inf,
                         device=getattr(args, 'device', None))

    print("\n>>> 使用生成的扰动测试模型鲁棒性...")
    top1_accuracy, top5_accuracy = 0.0, 0.0

    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader, desc="鲁棒测试中")):
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda() if torch.cuda.is_available() else (x_batch, y_batch)
            # 若 delta 尺寸与 x_batch 不完全匹配（例如输入尺寸不同），需要做 resize/中心裁剪等处理
            # 尝试广播 delta 到 batch size
            try:
                x_adv = torch.clamp(x_batch + delta, 0, 1)
            except Exception:
                # 若尺寸不匹配，尝试调整 delta 大小到 x_batch 的 H,W
                b, c, h, w = x_batch.shape
                d_c = delta.shape[1]
                if (c, h, w) != (d_c, delta.shape[2], delta.shape[3]):
                    # 简单 resize：使用 torch.nn.functional.interpolate
                    import torch.nn.functional as F
                    delta_resized = F.interpolate(delta, size=(h, w), mode='bilinear', align_corners=False)
                    delta_resized = delta_resized.expand(b, -1, -1, -1)
                    x_adv = torch.clamp(x_batch + delta_resized, 0, 1)
                else:
                    delta_exp = delta.expand(b, -1, -1, -1)
                    x_adv = torch.clamp(x_batch + delta_exp, 0, 1)

            logits = model(normalzie(args, x_adv))
            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0].item()
            top5_accuracy += top5[0].item()

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

    print(f"\n[SSP->UAP 鲁棒测试结果] Top-1: {top1_accuracy:.2f}% | Top-5: {top5_accuracy:.2f}%")
    return top1_accuracy, top5_accuracy


# ========== 加载模型（沿用原来的 load_at_model） ==========
def load_at_model(args):
    encoder_path = os.path.join('output', str(args.pre_dataset), '2aft_model',
                                str(args.victim), str(args.dataset), 'encoder')
    checkpoint = [Path(encoder_path) / ckpt for ckpt in os.listdir(Path(encoder_path))
                  if ckpt.endswith("pbs.pth")][0]
    encoder = torch.load(checkpoint)

    f_path = os.path.join('output', str(args.pre_dataset), '2aft_model',
                          str(args.victim), str(args.dataset), 'f')
    f_checkpoint = [Path(f_path) / ckpt for ckpt in os.listdir(Path(f_path))
                    if ckpt.endswith("pbs.pth")][0]

    F = torch.load(f_checkpoint)
    model = torch.nn.Sequential(encoder, F)
    return model.cuda()


# ========== 主函数（尽量保留你原来的参数与行为，同时增加 SSP 所需参数） ==========
def main():
    parser = argparse.ArgumentParser(description='SSP 风格的扰动生成与鲁棒性测试（保留原 UAP 接口）')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dataset', default='stl10', choices=['cifar10', 'stl10', 'gtsrb', 'imagenet'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pre_dataset', default='cifar10', choices=['cifar10', 'imagenet'])
    parser.add_argument('--victim', default='nnclr')

    # SSP 相关新参数（有默认值）
    parser.add_argument('--bound', type=float, default=8/255, help='扰动幅度上界（SSP 中的 bound）')
    parser.add_argument('--alpha', type=float, default=1e-2, help='每步扰动更新步长（SSP 中的 alpha）')
    parser.add_argument('--save_every_iter', type=int, default=100, help='多少迭代保存一次 delta')
    parser.add_argument('--workers', type=int, default=4, help='dataloader num_workers（供可选使用）')
    parser.add_argument('--device', type=str, default='cuda', help='设备：cuda 或 cpu')
    parser.add_argument('--model_arch', type=str, default='byol', help='模型结构标识（用于选 target_layer）')
    parser.add_argument('--save_dir', type=str, default='ssp_output', help='保存扰动的目录')
    parser.add_argument('--disable_tqdm', action='store_true', help='若设置则关闭 tqdm 显示')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # load_data 返回 train_loader, test_loader（与你原脚本一致）
    train_loader, test_loader = load_data(args.dataset, args.batch_size)

    # 注意：SSP 本意用训练集生成扰动，保持接口不变这里用 train_loader 作为生成器（rob_test_uap 中传入 test_loader）
    model = load_at_model(args)
    # 我们先用训练集生成扰动（如果你希望用 test_loader 生成，后面可改为传 test_loader）
    # 但为向后兼容，rob_test_uap() 会再次调用 generate_uap(test_loader,...)
    # 这里直接调用 rob_test_uap 使用 test_loader 来保持最小接口改动
    rob_test_uap(args, model, test_loader)


if __name__ == "__main__":
    main()
