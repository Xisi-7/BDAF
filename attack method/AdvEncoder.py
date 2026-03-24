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

# ========== 使用 UAP 进行鲁棒测试 ==========
def rob_test_uap(args, model, test_loader):
    print("\n>>> 开始生成 Universal Adversarial Perturbation (UAP)...")
    #delta = generate_uap(args, model, test_loader, xi=8/255, delta_steps=5, p=np.inf, device='cuda')

    print("\n>>> 使用 UAP 测试模型鲁棒性...")
    top1_accuracy, top5_accuracy = 0, 0
    uap_load_path = os.path.join('advencoder','output', str(args.pre_dataset), 'uap_results', 'gan_per', str(args.victim),
                                     str(args.dataset), 'nce', str(args.eps))
    uap_path = [Path(uap_load_path) / ckpt for ckpt in os.listdir(Path(uap_load_path)) if ckpt.endswith("20.pt")][0]
    UAP = torch.load(uap_path).cuda()
    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader, desc="鲁棒测试中")):
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            x_adv = torch.clamp(x_batch + UAP, 0, 1)
            logits = model(normalzie(args, x_adv))
            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

    print(f"\n[UAP鲁棒测试结果] Top-1: {top1_accuracy:.2f}% | Top-5: {top5_accuracy:.2f}%")
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
    parser = argparse.ArgumentParser(description='UAP 鲁棒性测试')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--dataset', default='stl10', choices=['cifar10', 'stl10', 'gtsrb', 'imagenet','animals10'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pre_dataset', default='cifar10', choices=['cifar10', 'imagenet'])
    parser.add_argument('--victim', default='nnclr')
    parser.add_argument('--eps', type=int, default=10)
    args = parser.parse_args()
    args.eps = args.eps / 255
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_loader, test_loader = load_data(args.dataset, args.batch_size)
    model = load_at_model(args)
    rob_test_uap(args, model, test_loader)


if __name__ == "__main__":
    main()
