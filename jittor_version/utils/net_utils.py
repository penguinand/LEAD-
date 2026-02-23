import os
import sys
import logging
import random
import numpy as np
import jittor as jt
import jittor.nn as nn


def get_device():
    """Jittor handles device automatically; return string for compatibility"""
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        return "cuda"
    return "cpu"


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * jt.log(input_ + epsilon)
    entropy = jt.sum(entropy, dim=1)
    return entropy


def log_args(args):
    s = "\n==========================================\n"
    s += ("python" + " ".join(sys.argv) + "\n")
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    s += "==========================================\n"
    return s


def set_logger(args, log_name="train_log.txt"):
    log_format = "%(asctime)s [%(levelname)s] - %(message)s"
    logger = logging.getLogger(log_name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    if args.test:
        file_handler = logging.FileHandler(os.path.join(args.save_dir, log_name), mode="a")
        file_format = logging.Formatter("%(message)s")
    else:
        file_handler = logging.FileHandler(os.path.join(args.save_dir, log_name), mode="w")
        file_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_format)

    terminal_handler = logging.StreamHandler()
    terminal_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    terminal_handler.setLevel(logging.INFO)
    terminal_handler.setFormatter(terminal_format)

    logger.addHandler(file_handler)
    logger.addHandler(terminal_handler)
    if not args.test:
        logger.debug(log_args(args))
    return logger


def compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flag=True, open_thresh=0.5, pred_unc_all=None):
    """Compute H-score using numpy arrays.
    gt_label_all: numpy array [N]
    pred_cls_all: numpy array [N, C]
    """
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros_like(per_class_num)
    pred_label_all = np.argmax(pred_cls_all, axis=1)  # [N]

    if open_flag:
        cls_num = pred_cls_all.shape[1]
        if pred_unc_all is None:
            epsilon = 1e-5
            entropy = -pred_cls_all * np.log(pred_cls_all + epsilon)
            entropy = np.sum(entropy, axis=1)
            pred_unc_all = entropy / np.log(cls_num)

        unc_idx = np.where(pred_unc_all > open_thresh)[0]
        pred_label_all[unc_idx] = cls_num

    for i, label in enumerate(class_list):
        label_idx = np.where(gt_label_all == label)[0]
        correct_idx = np.where(pred_label_all[label_idx] == label)[0]
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))

    per_class_acc = per_class_correct / (per_class_num + 1e-5)

    if open_flag:
        known_acc = per_class_acc[:-1].mean()
        unknown_acc = per_class_acc[-1]
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
    else:
        known_acc = per_class_correct.sum() / (per_class_num.sum() + 1e-5)
        unknown_acc = 0.0
        h_score = 0.0

    return h_score, known_acc, unknown_acc, per_class_acc


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction

    def execute(self, inputs, targets, applied_softmax=True):
        if applied_softmax:
            log_probs = jt.log(inputs + 1e-10)
        else:
            log_probs = jt.nn.log_softmax(inputs, dim=-1)

        if len(targets.shape) == 1 or inputs.shape != targets.shape:
            targets_oh = jt.zeros_like(inputs)
            _scatter_src = jt.ones([targets.shape[0], 1]).float32()
            targets_oh.scatter_(1, targets.unsqueeze(1).int32(), _scatter_src)
            targets = targets_oh

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss
