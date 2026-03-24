import os
import torch
import argparse
import random
import json
import numpy as np
from utils.load_data import load_data, normalzie
from utils.predict import make_print_to_file, test,rob_test
from copy import deepcopy
from tqdm import tqdm
from utils.drc import layer_robustness_contribution
from torch import nn
from pathlib import Path


def arg_parse():
    parser = argparse.ArgumentParser(description='Train downstream models using of the pre-trained encoder')
    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    parser.add_argument('--dataset', default='stl10', choices=['cifar10', 'stl10', 'gtsrb', 'animals10'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--save', default='True')
    parser.add_argument('--pre_dataset', default='cifar10', choices=['cifar10'])
    parser.add_argument('--victim', default='byol',
                        choices=['byol', 'dino', 'mocov2plus','nnclr', 'ressl', 'swav', 'wmse'])
    args = parser.parse_args()
    return args


def train(args, model, dataloader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for i, (inputs, targets) in enumerate(tqdm(dataloader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(normalzie(args, inputs))
        loss = criterion(outputs, targets)
        train_loss += loss.item() * targets.size(0)

        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss / total, correct / total * 100, model


def load_at_model(args):
    encoder_path = os.path.join('output', str(args.pre_dataset), 'aft_model',
                                str(args.victim), str(args.dataset), 'encoder')
    checkpoint = [Path(encoder_path) / ckpt for ckpt in os.listdir(Path(encoder_path))
                  if ckpt.endswith("last.pth")][0]
    encoder = torch.load(checkpoint)

    f_path = os.path.join('output', str(args.pre_dataset), 'aft_model',
                          str(args.victim), str(args.dataset), 'f')
    f_checkpoint = [Path(f_path) / ckpt for ckpt in os.listdir(Path(f_path))
                    if ckpt.endswith("last.pth")][0]

    F = torch.load(f_checkpoint)
    model = torch.nn.Sequential(encoder, F)
    return model


def main():
    args = arg_parse()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_printoptions(profile="full")
    torch.cuda.synchronize()

    # Logging
    log_save_path = os.path.join('output', str(args.pre_dataset), 'log',
                                 'down_test', "2aft_model", str(args.victim),
                                 str(args.dataset))
    os.makedirs(log_save_path, exist_ok=True)
    now_time = make_print_to_file(path=log_save_path)

    # Dump args
    with open(os.path.join(log_save_path, 'args.json'), 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    uap_save_path_e = os.path.join('output', str(args.pre_dataset), '2aft_model',
                                   str(args.victim), str(args.dataset), 'encoder')
    uap_save_path_f = os.path.join('output', str(args.pre_dataset), '2aft_model',
                                   str(args.victim), str(args.dataset), 'f')
    os.makedirs(uap_save_path_e, exist_ok=True)
    os.makedirs(uap_save_path_f, exist_ok=True)

    # load data
    train_loader, test_loader = load_data(args.dataset, args.batch_size)

    print('Day: %s, Target encoder:%s, Downstream task:%s' % (now_time, args.victim, args.dataset))
    print("######################################  Test Attack! ######################################")

    print('==> Phase 1: Adversarial Fine-Tuning...')
    model = load_at_model(args)

    clean_acc_t1, clean_acc_t5 = test(args, model, test_loader)
    print('Clean downstream accuracy: %.4f%%' % (clean_acc_t1))

    print('==> Phase 2: Standard Fine-Tuning...')
    sorted_layer_robustness_contribution, non_robust_cd_layer_min_top10p = layer_robustness_contribution(
        deepcopy(model), epsilon=0.1)

    # Dump non_robust_cd_layer & value
    with open(os.path.join(log_save_path, 'layer.json'), 'w') as fid:
        json.dump(sorted_layer_robustness_contribution, fid, indent=2)

    all_param_names = [name for name, _ in model.named_parameters()]

    # Buffer-guided Robustness-Aware Layer Role Assignment Strategy

    selected_indices = []
    for n_key in non_robust_cd_layer_min_top10p:
        for idx, name in enumerate(all_param_names):
            if n_key in name:
                selected_indices.append(idx)

    expanded_indices = set()
    for idx in selected_indices:
        expanded_indices.add(idx)
        if idx - 1 >= 0:
            expanded_indices.add(idx - 1)
        if idx + 1 < len(all_param_names):
            expanded_indices.add(idx + 1)

    # Differentiated Learning Rate Fine-tuning

    # Role Aware Learning Rate

    base_core_lr = 1e-3
    base_neighbor_lr = 1e-4
    decay = 0.95
    param_groups = []

    # Layer-wise learning rate decay

    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx in expanded_indices:
            param.requires_grad = True
            depth_decay = decay ** (len(all_param_names) - 1 - idx)
            if idx in selected_indices:
                lr = base_core_lr * depth_decay
            else:
                lr = base_neighbor_lr * depth_decay
            param_groups.append({"params": [param], "lr": lr})
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(param_groups, weight_decay=8e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Robustness–Accuracy Trade-Off Score-based Model Selection
    best_tos_gap = float('inf')
    best_tos_clean = 0
    best_tos_rob = 0

    best_clean_acc = 0
    best_clean_rob = 0
    for epoch in range(args.epochs):
        print(f"==> Epoch {epoch + 1}/{args.epochs}")
        print("==> Training...")
        criterion = nn.CrossEntropyLoss()
        train_loss, train_acc, model = train(args, model, train_loader, optimizer, criterion)

        scheduler.step()

        print(f"==> Train loss: {train_loss:.4f}, train acc: {train_acc:.2f}%")
        clean_acc_t1, clean_acc_t5 = test(args, model, test_loader)
        rob_acc_t1, rob_acc_t5 = rob_test(args, model, test_loader)
        print('Epoch [%d/%d], Top1 train acc: %.4f, Top1 test acc: %.4f,Top1  rob test acc: %.4f'
              % (epoch + 1, args.epochs, train_acc, clean_acc_t1, rob_acc_t1))

        gap = abs(clean_acc_t1 - rob_acc_t1)

        save_tos = False

        if gap < best_tos_gap:
            save_tos = True

        elif (clean_acc_t1 > best_tos_clean) and (rob_acc_t1 > best_tos_rob):
            save_tos = True

        if args.save == 'True' and save_tos:
            best_tos_gap = gap
            best_tos_clean = clean_acc_t1
            best_tos_rob = rob_acc_t1

            torch.save(
                model[0],
                f'{uap_save_path_e}/{args.victim}_{args.pre_dataset}_{args.dataset}_tos.pth'
            )
            torch.save(
                model[1],
                f'{uap_save_path_f}/{args.victim}_{args.pre_dataset}_{args.dataset}_tos.pth'
            )

        if clean_acc_t1 > best_clean_acc:
            best_clean_acc = clean_acc_t1
            best_clean_rob = rob_acc_t1

            if args.save == 'True':
                torch.save(
                    model[0],
                    f'{uap_save_path_e}/{args.victim}_{args.pre_dataset}_{args.dataset}_bestclean.pth'
                )
                torch.save(
                    model[1],
                    f'{uap_save_path_f}/{args.victim}_{args.pre_dataset}_{args.dataset}_bestclean.pth'
                )
    print("\n================== Training Summary ==================")
    print(
            f"Best TOS  -> Clean: {best_tos_clean:.4f}, "
            f"Rob: {best_tos_rob:.4f}, "
            f"Gap: {best_tos_gap:.4f}"
        )
    print(
            f"BestClean -> Clean: {best_clean_acc:.4f}, "
            f"Rob: {best_clean_rob:.4f}"
        )
    print("=====================================================\n")


if __name__ == "__main__":
    main()
