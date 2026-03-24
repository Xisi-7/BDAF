import os
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from utils.load_data import load_data, normalzie


# ========== Top-k 准确率计算 ==========
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


# ========== 干净样本（Clean）测试 ==========
def clean_test(args, model, test_loader):
    print("\n>>> 开始干净样本（Clean）测试...")
    model.eval()

    top1_accuracy, top5_accuracy = 0, 0

    with torch.no_grad():
        for counter, (x_batch, y_batch) in enumerate(
                tqdm(test_loader, desc="干净样本测试中")):
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            # 直接前向，不加任何扰动
            logits = model(normalzie(args, x_batch))

            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

    print(f"[Clean Test] Top-1: {top1_accuracy:.2f}% | Top-5: {top5_accuracy:.2f}%")
    return top1_accuracy.item(), top5_accuracy.item()


# ========== 加载模型 ==========
def load_at_model(args):
    encoder_path = os.path.join(
        'output', str(args.pre_dataset), '2aft_model',
        str(args.victim), str(args.dataset), 'encoder'
    )
    encoder_ckpt = [
        Path(encoder_path) / ckpt
        for ckpt in os.listdir(Path(encoder_path))
        if ckpt.endswith("pbs.pth")
    ][0]
    encoder = torch.load(encoder_ckpt)

    f_path = os.path.join(
        'output', str(args.pre_dataset), '2aft_model',
        str(args.victim), str(args.dataset), 'f'
    )
    f_ckpt = [
        Path(f_path) / ckpt
        for ckpt in os.listdir(Path(f_path))
        if ckpt.endswith("pbs.pth")
    ][0]
    F = torch.load(f_ckpt)

    model = torch.nn.Sequential(encoder, F)
    return model.cuda()


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description='Clean Accuracy 测试')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dataset', default='stl10',
                        choices=['cifar10', 'stl10', 'gtsrb', 'imagenet'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pre_dataset', default='cifar10',
                        choices=['cifar10', 'imagenet'])
    parser.add_argument('--victim', default='nnclr')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_loader, test_loader = load_data(args.dataset, args.batch_size)
    model = load_at_model(args)

    # 只进行干净样本测试
    clean_test(args, model, test_loader)


if __name__ == "__main__":
    main()
