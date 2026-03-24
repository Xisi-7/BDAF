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

mean_std = {'cifar10': [(0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)]}

# ========== 辅助：根据 target_layer 返回待注册的 module ==========
def get_target_module_from_model(model, target_layer):
    """
    尝试从 model[0] 中智能提取出用于注册 hook 的模块。
    支持以下情况（优先顺序）：
      - model[0].blocks[target_layer]
      - model[0].net[target_layer]
      - model[0].layer1/layer2/layer3/layer4（ResNet 常见）
      - model[0].conv1
    返回 (module, chosen_name)；若无法找到则抛出 ValueError。
    """
    encoder = model[0]  # 你的 model 是 Sequential(encoder, classifier)

    # 1) blocks（某些实现）
    if hasattr(encoder, 'blocks'):
        blocks = getattr(encoder, 'blocks')
        try:
            mod = blocks[target_layer]
            return mod, f'blocks[{target_layer}]'
        except Exception:
            pass

    # 2) net（某些包装器，如 SimCLR 的包装）
    if hasattr(encoder, 'net'):
        net = getattr(encoder, 'net')
        try:
            mod = net[target_layer]
            return mod, f'net[{target_layer}]'
        except Exception:
            pass

    # 3) ResNet 常见的 layer1..layer4 或 conv1
    # 映射：0->conv1, 1->layer1, 2->layer2, 3->layer3, 4->layer4
    try:
        if target_layer == 0 and hasattr(encoder, 'conv1'):
            return getattr(encoder, 'conv1'), 'conv1'
        layer_name = f'layer{target_layer}' if target_layer >= 1 else 'layer1'
        # 如果 target_layer 比较大超出范围，回退到 layer4（最后一层）
        if hasattr(encoder, layer_name):
            return getattr(encoder, layer_name), layer_name
    except Exception:
        pass

    # 4) 尝试按 index 访问 encoder.children()
    try:
        children = list(encoder.children())
        if 0 <= target_layer < len(children):
            return children[target_layer], f'children[{target_layer}]'
    except Exception:
        pass

    # 最终无法识别则报错
    raise ValueError(f"无法从 model[0] 中识别 target_layer={target_layer} 对应的模块，请检查模型结构或调整 target_layer。")


# ========== L4A_ugs 攻击实现 ==========
def generate_l4a_ugs(args, model, dataloader):
    """
    使用 L4A_ugs 方法生成通用扰动 (Universal Gaussian Stimulus)
    已增强：自动识别 target 层并注册 hook，包含异常处理与提示。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()

    bound = args.bound if hasattr(args, 'bound') else 10/255
    step_size = args.alpha if hasattr(args, 'alpha') else 1/255
    lamuda = getattr(args, 'lamuda', 0.5)
    target_layer = getattr(args, 'target_layer', 1)
    mean_std_type = getattr(args, 'mean_std', 'imagenet')

    # 初始化 delta，使用数据加载器的样例形状
    try:
        x_sample = next(iter(dataloader))[0]
    except StopIteration:
        raise RuntimeError("数据加载器为空，请检查 dataloader 是否正确。")
    _, c, h, w = x_sample.shape
    delta = torch.rand(1, c, h, w, device=device) * 2 * bound - bound
    delta.requires_grad = True

    k = 0
    for (image, _) in tqdm(dataloader, desc="生成 L4A_ugs 通用扰动"):
        model.zero_grad()
        activations = []
        remove_handles = []

        # 注册 hook 获取中间层激活特征
        def activation_recorder_hook(self, input, output):
            # 保存平方后的激活，保持原有实现逻辑
            try:
                activations.append(torch.square(output))
            except Exception:
                # 若 output 不是 tensor（谨防某些包装返回 tuple），尝试取第一个元素
                if isinstance(output, (list, tuple)):
                    activations.append(torch.square(output[0]))
                else:
                    raise
            return None

        # 获取待注册的 module（带友好提示）
        try:
            target_module, chosen_name = get_target_module_from_model(model, target_layer)
        except ValueError as e:
            # 提醒用户并退出（或回退到 layer4）
            print("⚠️ get_target_module_from_model 出错：", e)
            print("尝试回退到 ResNet 的 layer4 注册 hook。")
            try:
                target_module = getattr(model[0], 'layer4')
                chosen_name = 'layer4'
            except Exception:
                raise RuntimeError("无法回退到 layer4，请手动检查模型结构。")

        # 注册 hook
        handle = target_module.register_forward_hook(activation_recorder_hook)
        remove_handles.append(handle)
        # 打印一次信息（每个 batch 不重复打印，减少日志噪声）
        if k == 0:
            print(f"✅ 在模块 {chosen_name} 上注册了 forward hook（target_layer={target_layer}）。")

        # detach 并重置 grad
        delta = delta.detach().requires_grad_()

        # 生成随机高斯噪声图片
        dataplus = torch.randn(8, 3, h, w, device=device)
        if mean_std_type == 'cifar10':
            mean, std = mean_std['cifar10']
            # mean 和 std 是 tuples
            for i in range(3):
                dataplus[:, i, :, :] = dataplus[:, i, :, :] * std[i] + mean[i]
        else:
            # uniform 模式，兼容原来随机 mean/std 的想法
            std_rand = torch.rand(3, device=device) * 0.1 + 0.05
            mean_rand = torch.rand(3, device=device) * 0.6 + 0.2
            for i in range(3):
                dataplus[:, i, :, :] = dataplus[:, i, :, :] * std_rand[i] + mean_rand[i]

        dataplus.requires_grad = False
        image = image.to(device)
        data = torch.cat((dataplus, image), dim=0)

        # 前向传播计算激活损失
        # 注意： activations 列表会在 hook 被触发后填充
        model(data + delta)

        if len(activations) == 0:
            # 如果没有激活被记录，提供调试信息并报错
            for h_ in remove_handles:
                h_.remove()
            raise RuntimeError("未捕获到任何中间激活（activations 为空）。请确认 hook 注册的层正确，或检查模型前向是否执行。")

        # activations[0] 的第一部分对应 dataplus（前 8 张），后面对应真实图片
        # 要确保 activations[0] 的 batch 维度 >= 8
        act0 = activations[0]
        if act0.size(0) < 9:
            # 较小的 batch 时可能不满足索引要求，使用切分策略：尽量以一半划分
            mid = act0.size(0) // 2
            part1 = act0[:mid]
            part2 = act0[mid:]
            loss = torch.mean(part1) + torch.mean(part2) * lamuda
        else:
            loss = torch.mean(act0[0:8]) + torch.mean(act0[8:]) * lamuda

        # 更新 delta：FGSM-like
        loss.backward()
        # 检查 delta.grad 是否为 None（可能因 detach 问题）
        if delta.grad is None:
            # 清理 hook 后抛错，便于用户定位
            for h_ in remove_handles:
                h_.remove()
            raise RuntimeError("delta.grad 为 None，无法进行更新。请确认 delta.requires_grad=True 且 loss.backward() 正常。")

        delta = delta + step_size * delta.grad.sign()
        delta.clamp_(-bound, bound)

        k += 1
        # 卸载 hook
        for h_ in remove_handles:
            h_.remove()

    print(f"✅ L4A_ugs 扰动生成完成，最终范数 = {delta.abs().max():.4f}")
    return delta.detach()


# ========== 准确率计算 ==========
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


# ========== 使用 L4A_ugs 进行鲁棒测试 ==========
def rob_test_l4a_ugs(args, model, test_loader):
    print("\n>>> 开始生成 L4A_ugs 通用扰动...")
    delta = generate_l4a_ugs(args, model, test_loader)

    print("\n>>> 使用 L4A_ugs 测试模型鲁棒性...")
    top1_accuracy, top5_accuracy = 0, 0

    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader, desc="鲁棒测试中")):
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            # 如果 delta 是 1xCxHxW，会自动广播到 batch
            x_adv = torch.clamp(x_batch + delta, 0, 1)
            logits = model(normalzie(args, x_adv))
            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

    print(f"\n[L4A_ugs 鲁棒测试结果] Top-1: {top1_accuracy:.2f}% | Top-5: {top5_accuracy:.2f}%")
    return top1_accuracy.item(), top5_accuracy.item()


# ========== 加载模型 ==========
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


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description='L4A_ugs 鲁棒性测试')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dataset', default='stl10', choices=['cifar10', 'stl10', 'gtsrb', 'imagenet','animals10'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pre_dataset', default='cifar10', choices=['cifar10', 'imagenet'])
    parser.add_argument('--victim', default='nnclr')

    # L4A_ugs 特有参数
    parser.add_argument('--alpha', type=float, default=1/255)
    parser.add_argument('--bound', type=float, default=10/255)
    parser.add_argument('--lamuda', type=float, default=0.5)
    parser.add_argument('--target_layer', type=int, default=1, help="0->conv1,1->layer1,2->layer2,3->layer3,4->layer4")
    parser.add_argument('--mean_std', default='cifar10', choices=['cifar10', 'uniform'])
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_loader, test_loader = load_data(args.dataset, args.batch_size)
    model = load_at_model(args)
    rob_test_l4a_ugs(args, model, test_loader)


if __name__ == "__main__":
    main()
