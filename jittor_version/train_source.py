import os
import shutil
import numpy as np
import jittor as jt
import jittor.nn as nn
from tqdm import tqdm

from model.SFUniDA import SFUniDA, load_pytorch_weights
from dataset.dataset import SFUniDADataset
from config.model_config import build_args
from utils.net_utils import (set_logger, set_random_seed, get_device,
                              compute_h_score, CrossEntropyLabelSmooth)


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
    return optimizer


def train(args, model, dataloader, criterion, optimizer, epoch_idx=0.0):
    model.train()
    loss_stack = []

    iter_idx = epoch_idx * len(dataloader)
    iter_max = args.epochs * len(dataloader)

    for batch_idx, (imgs_train, _, imgs_label, _) in enumerate(dataloader):
        iter_idx += 1

        _, pred_cls = model(imgs_train, apply_softmax=True)
        imgs_onehot_label = jt.zeros_like(pred_cls)
        _scatter_src = jt.ones([imgs_label.shape[0], 1]).float32()
        imgs_onehot_label.scatter_(1, imgs_label.unsqueeze(1).int32(), _scatter_src)

        loss = criterion(pred_cls, imgs_onehot_label)

        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.step(loss)

        loss_stack.append(loss.item())

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch_idx} [{batch_idx+1}/{len(dataloader)}] loss={loss.item():.4f}")

    train_loss = np.mean(loss_stack)
    return train_loss


def test(args, model, dataloader, src_flg=True):
    model.eval()
    gt_label_stack = []
    pred_cls_stack = []

    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0

    for _, imgs_test, imgs_label, _ in dataloader:
        _, pred_cls = model(imgs_test, apply_softmax=True)
        gt_label_stack.append(imgs_label.numpy())
        pred_cls_stack.append(pred_cls.numpy())

    gt_label_all = np.concatenate(gt_label_stack, axis=0)
    pred_cls_all = np.concatenate(pred_cls_stack, axis=0)

    h_score, known_acc, unknown_acc, per_cls_acc = compute_h_score(
        args, class_list, gt_label_all, pred_cls_all, open_flg, open_thresh=0.50)

    return h_score, known_acc, unknown_acc, per_cls_acc


def main(args):
    device = get_device()
    print(f"Using device: {device}")
    this_dir = os.path.dirname(os.path.abspath(__file__))

    model = SFUniDA(args)

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        save_dir = os.path.dirname(args.checkpoint)
        # Load Jittor checkpoint
        if args.checkpoint.endswith('.pkl'):
            model.load(args.checkpoint)
        else:
            # Load PyTorch checkpoint
            load_pytorch_weights(model, args.checkpoint)
    else:
        save_dir = os.path.join(this_dir, "checkpoints", args.dataset,
                                "source_{}".format(args.s_idx),
                                "source_{}_{}".format(args.source_train_type, args.target_label_type))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    args.save_dir = save_dir
    logger = set_logger(args, log_name="log_source_training.txt")

    params_group = []
    for k, v in model.backbone_layer.named_parameters():
        params_group += [{"params": [v], 'lr': args.lr * 0.1}]
    for k, v in model.feat_embed_layer.named_parameters():
        params_group += [{"params": [v], 'lr': args.lr}]
    for k, v in model.class_layer.named_parameters():
        params_group += [{"params": [v], 'lr': args.lr}]

    optimizer = jt.optim.SGD(params_group, lr=args.lr, momentum=0.9,
                              weight_decay=1e-3, nesterov=True)
    optimizer = op_copy(optimizer)

    source_data_list = open(os.path.join(args.source_data_dir, "image_unida_list.txt"), "r").readlines()
    source_dataset = SFUniDADataset(args, args.source_data_dir, source_data_list, d_type="source",
                                     preload_flg=True, batch_size=args.batch_size,
                                     shuffle=True, num_workers=0, drop_last=True)

    target_dataloader_list = []
    for idx in range(len(args.target_domain_dir_list)):
        target_data_dir = args.target_domain_dir_list[idx]
        target_data_list = open(os.path.join(target_data_dir, "image_unida_list.txt"), "r").readlines()
        target_dataset = SFUniDADataset(args, target_data_dir, target_data_list, d_type="target",
                                         preload_flg=False, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0, drop_last=False)
        target_dataloader_list.append(target_dataset)

    if args.source_train_type == "smooth":
        criterion = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1, reduction=True)
    elif args.source_train_type == "vanilla":
        criterion = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.0, reduction=True)
    else:
        raise ValueError("Unknown source_train_type:", args.source_train_type)

    notation_str = "\n=================================================\n"
    notation_str += "    START TRAINING ON THE SOURCE:{} == {}         \n".format(args.s_idx, args.target_label_type)
    notation_str += "================================================="
    logger.info(notation_str)

    for epoch_idx in range(args.epochs):
        train_loss = train(args, model, source_dataset, criterion, optimizer, epoch_idx)
        logger.info("Epoch:{}/{} train_loss:{:.3f}".format(epoch_idx, args.epochs, train_loss))

        if epoch_idx % 1 == 0:
            source_h_score, source_known_acc, source_unknown_acc, src_per_cls_acc = test(
                args, model, source_dataset, src_flg=True)
            logger.info("EVALUATE ON SOURCE: H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(
                source_h_score, source_known_acc, source_unknown_acc))

        checkpoint_file = "latest_source_checkpoint.pkl"
        model.save(os.path.join(save_dir, checkpoint_file))

    for idx_i, item in enumerate(args.target_domain_list):
        notation_str = "\n=================================================\n"
        notation_str += "        EVALUATE ON THE TARGET:{}                  \n".format(item)
        notation_str += "================================================="
        logger.info(notation_str)

        hscore, knownacc, unknownacc, _ = test(args, model, target_dataloader_list[idx_i], src_flg=False)
        logger.info("H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownACC:{:.3f}".format(hscore, knownacc, unknownacc))


if __name__ == "__main__":
    args = build_args()
    set_random_seed(args.seed)
    main(args)
