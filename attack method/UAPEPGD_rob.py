import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.load_data import load_data, normalzie


# =========================
# Top-k 准确率
# =========================
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# =========================
# 加载模型
# =========================
def load_at_model(args):
    encoder_path = Path(
        'output', str(args.pre_dataset), '2aft_model',
        str(args.victim), str(args.dataset), 'encoder'
    )
    encoder_ckpt = [p for p in encoder_path.iterdir() if p.name.endswith("pbs.pth")][0]
    encoder = torch.load(encoder_ckpt)

    f_path = Path(
        'output', str(args.pre_dataset), '2aft_model',
        str(args.victim), str(args.dataset), 'f'
    )
    f_ckpt = [p for p in f_path.iterdir() if p.name.endswith("pbs.pth")][0]
    head = torch.load(f_ckpt)

    model = torch.nn.Sequential(encoder, head)
    return model.cuda().eval()


# =========================
# EPGD 内部增量
# =========================
def get_delta(model, image_copy, clean_label, bound, alpha):
    v = torch.zeros_like(image_copy)
    beta = 0.5
    gamma = 1e-5

    delta0 = (bound * torch.rand_like(image_copy) * 2 - bound).requires_grad_(True)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(10):
        adv_pre = F.normalize(model(image_copy + delta0), dim=1)
        loss = criterion(adv_pre, clean_label)
        loss.backward()

        with torch.no_grad():
            v = beta * v + gamma * (image_copy + delta0) + delta0.grad
            delta0 += alpha * v.sign()
            delta0.clamp_(-bound, bound)

        delta0 = delta0.detach().requires_grad_(True)

    return delta0.detach()


# =========================
# UAP-EPGD 生成
# =========================
def uapepgd(model, args, train_loader):
    device = args.device
    bound = args.bound

    delta = bound * torch.rand((1, 3, 64, 64), device=device) * 2 - bound

    uap_dir = Path(
        'output', str(args.pre_dataset), 'uap_results',
        str(args.victim), str(args.dataset)
    )
    uap_dir.mkdir(parents=True, exist_ok=True)

    for k, (image, _) in enumerate(tqdm(train_loader, desc="UAP-EPGD"), start=1):
        image = image[:1].to(device)

        with torch.no_grad():
            clean_pre = F.normalize(model(image), dim=1)
            clean_label = clean_pre.argmax(dim=1)

            adv_pre = F.normalize(model(image + delta), dim=1)
            adv_label = adv_pre.argmax(dim=1)

        if (adv_label == clean_label).all():
            image_copy = image + delta
            alpha = 0.002 if k < 10000 else 0.0002
            delta0 = get_delta(model, image_copy, clean_label, bound, alpha)
            delta = torch.clamp(delta + delta0, -bound, bound)

        if k >= args.max_iter:
            break

    final_path = uap_dir / f'uap_final_k{k}.pt'
    torch.save(delta.detach().cpu(), final_path)
    print(f"[✓] Final UAP saved: {final_path}")


# =========================
# UAP 鲁棒测试
# =========================
def rob_test_uap(args, model, test_loader):
    device = args.device

    uap_dir = Path(
        'output', str(args.pre_dataset), 'uap_results',
        str(args.victim), str(args.dataset)
    )

    uap_files = sorted(
        uap_dir.glob("uap_final_k*.pt"),
        key=lambda x: int(x.stem.split("k")[-1])
    )
    assert len(uap_files) > 0, f"UAP 目录为空: {uap_dir}"

    uap_path = uap_files[-1]
    print(f"[✓] Load UAP from {uap_path}")

    UAP = torch.load(uap_path).to(device)

    top1_sum, top5_sum, total = 0, 0, 0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="UAP Robust Test"):
            x, y = x.to(device), y.to(device)

            x_adv = torch.clamp(x + UAP, 0, 1)
            logits = model(normalzie(args, x_adv))

            top1, top5 = accuracy(logits, y, topk=(1, 5))
            bs = x.size(0)

            top1_sum += top1.item() * bs
            top5_sum += top5.item() * bs
            total += bs

    print(f"\n[Result] Top-1: {top1_sum / total:.2f}% | Top-5: {top5_sum / total:.2f}%")


# =========================
# 参数
# =========================
def get_args():
    parser = argparse.ArgumentParser("UAP-EPGD + Robust Test")

    parser.add_argument('--gpu', default='1')
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--dataset', default='stl10',
                        choices=['animals10', 'stl10', 'gtsrb', 'imagenet'])
    parser.add_argument('--pre_dataset', default='cifar10',
                        choices=['cifar10', 'imagenet'])
    parser.add_argument('--victim', default='nnclr')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--bound', type=float, default=0.05)
    parser.add_argument('--max_iter', type=int, default=20000)

    parser.add_argument('--eps', type=int, default=10)

    return parser.parse_args()


# =========================
# main
# =========================
def main():
    args = get_args()
    args.eps = args.eps / 255

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = load_at_model(args)

    # ✅ 只加载一次数据
    train_loader, test_loader = load_data(args.dataset, args.batch_size)

    # 1️⃣ 生成 UAP
    uapepgd(model, args, train_loader)

    # 2️⃣ 鲁棒测试
    rob_test_uap(args, model, test_loader)


if __name__ == "__main__":
    main()
