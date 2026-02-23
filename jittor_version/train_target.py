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
                              compute_h_score, Entropy)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
    return optimizer


best_score = 0.0
best_coeff = 1.0


def obtain_LEAD_pseudo_labels(args, model, dataloader, epoch_idx=0.0):
    model.eval()
    pred_cls_bank = []
    gt_label_bank = []
    embed_feat_bank = []

    class_list = args.target_class_list
    args.logger.info("Generating offline feat_decomposition based pseudo labels...")

    with jt.no_grad():
        for _, imgs_test, imgs_label, _ in dataloader:
            embed_feat, pred_cls = model(imgs_test, apply_softmax=True)
            pred_cls_bank.append(pred_cls.numpy())
            embed_feat_bank.append(embed_feat.numpy())
            gt_label_bank.append(imgs_label.numpy())

    pred_cls_bank = np.concatenate(pred_cls_bank, axis=0)  # [N, C]
    gt_label_bank = np.concatenate(gt_label_bank, axis=0)  # [N]
    embed_feat_bank = np.concatenate(embed_feat_bank, axis=0)  # [N, D]
    embed_feat_bank = embed_feat_bank / (np.linalg.norm(embed_feat_bank, axis=1, keepdims=True) + 1e-8)

    global best_score, best_coeff

    # C_t estimation
    if epoch_idx % 10 == 0:
        args.logger.info("Performing C_t estimation...")
        coeff_list = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
        embed_feat_cpu = embed_feat_bank.copy()

        if args.dataset == "VisDA":
            np.random.seed(2021)
            data_size = embed_feat_cpu.shape[0]
            sample_idxs = np.random.choice(data_size, data_size // 3, replace=False)
            embed_feat_cpu = embed_feat_cpu[sample_idxs, :]

        embed_feat_cpu = TSNE(n_components=2, init="pca", random_state=0).fit_transform(embed_feat_cpu)

        best_score = 0.0
        for coeff in coeff_list:
            Ct = max(int(args.class_num * coeff), 2)
            kmeans = KMeans(n_clusters=Ct, random_state=0).fit(embed_feat_cpu)
            cluster_labels = kmeans.labels_
            sil_score = silhouette_score(embed_feat_cpu, cluster_labels)
            if sil_score > best_score:
                best_score = sil_score
                best_coeff = coeff
        args.logger.info("Performing C_t estimation Done!")

    Ct = int(args.class_num * best_coeff)
    data_num = pred_cls_bank.shape[0]
    pos_topk_num = int(data_num / Ct)

    global known_space_basis, unknown_space_basis
    if epoch_idx == 0.0:
        # SVD on classifier weights
        src_cls_feat = model.class_layer.fc.weight.numpy()  # [C, D]
        u, s, vt = np.linalg.svd(src_cls_feat.T, full_matrices=True)
        main_r = args.class_num
        known_space_basis = u[:, :main_r].T  # [C, D]
        known_space_basis = known_space_basis / (np.linalg.norm(known_space_basis, axis=-1, keepdims=True) + 1e-8)
        unknown_space_basis = u[:, main_r:].T  # [D-C, D]
        unknown_space_basis = unknown_space_basis / (np.linalg.norm(unknown_space_basis, axis=-1, keepdims=True) + 1e-8)

    # Projections
    known_proj_cords = embed_feat_bank @ known_space_basis.T  # [N, C]
    unknown_proj_cords = embed_feat_bank @ unknown_space_basis.T  # [N, D-C]

    known_proj_norm = np.linalg.norm(known_proj_cords, axis=-1)  # [N]
    unknown_proj_norm = np.linalg.norm(unknown_proj_cords, axis=-1)  # [N]

    # GMM on unknown norms
    unknown_space_norm_gm = GaussianMixture(n_components=2, random_state=0).fit(unknown_proj_norm.reshape(-1, 1))
    gaussian_two_mus = unknown_space_norm_gm.means_.squeeze()
    gaussian_mu1 = np.min(gaussian_two_mus)
    gaussian_mu2 = np.max(gaussian_two_mus)

    # Target prototype construction
    sorted_pred_cls_idxs = np.argsort(-pred_cls_bank, axis=0)  # descending
    pos_topk_idxs = sorted_pred_cls_idxs[:pos_topk_num, :].T  # [C, topk]

    tar_pos_feat_proto = np.zeros((args.class_num, args.embed_feat_dim))
    for c in range(args.class_num):
        tar_pos_feat_proto[c] = np.mean(embed_feat_bank[pos_topk_idxs[c]], axis=0)
    tar_pos_feat_proto = tar_pos_feat_proto / (np.linalg.norm(tar_pos_feat_proto, axis=-1, keepdims=True) + 1e-8)

    # Source anchors
    src_cls_weight = model.class_layer.fc.weight.numpy()  # [C, D]
    src_pos_feat_proto = src_cls_weight / (np.linalg.norm(src_cls_weight, axis=-1, keepdims=True) + 1e-8)

    tar_psd_pos_feat_simi = np.clip(embed_feat_bank @ tar_pos_feat_proto.T, 0.0, None)  # [N, C]
    src_psd_pos_feat_simi = np.clip(embed_feat_bank @ src_pos_feat_proto.T, 0.0, None)  # [N, C]

    per_sample_fuse_common_score = np.sqrt(
        (1.0 - np.exp(-tar_psd_pos_feat_simi)) * np.exp(src_psd_pos_feat_simi - 1.0))

    # Per-class norm prior
    per_cls_norm_prior = np.zeros(args.class_num)
    for c in range(args.class_num):
        per_cls_norm_prior[c] = np.mean(unknown_proj_norm[pos_topk_idxs[c]])

    # Instance-level decision boundaries
    per_sample_per_cls_thresh = np.tile(per_cls_norm_prior, (data_num, 1))  # [N, C]
    per_cls_thresh_gap = np.clip(gaussian_mu2 - per_cls_norm_prior, 0.0, None)  # [C]
    per_sample_per_cls_thresh = per_sample_per_cls_thresh + per_sample_fuse_common_score * per_cls_thresh_gap[None, :]

    # Obtain pseudo-labels
    psd_label = np.argmax(per_sample_fuse_common_score, axis=-1)
    psd_label_weight = np.ones_like(psd_label, dtype=np.float32)
    psd_label_oh = psd_label.copy()

    for i in range(args.class_num):
        label_idxs = np.where(psd_label == i)[0]
        alpha = 1e-4
        unknown_mask = unknown_proj_norm[label_idxs] >= per_sample_per_cls_thresh[label_idxs, i]
        psd_label[label_idxs[unknown_mask]] = args.class_num

        psd_label_weight[label_idxs] = 1.0 - np.power(
            (1 + (unknown_proj_norm[label_idxs] - per_sample_per_cls_thresh[label_idxs, i]) ** 2 / alpha),
            -(alpha + 1.) / 2.)

        unkown_idxs = unknown_proj_norm[label_idxs] >= gaussian_mu2
        known_idxs = unknown_proj_norm[label_idxs] < per_cls_norm_prior[i]
        psd_label_weight[label_idxs[unkown_idxs]] = 1.0
        psd_label_weight[label_idxs[known_idxs]] = 1.0

    psd_unknown_flg = (psd_label == args.class_num)  # [N]
    psd_known_flg = (psd_label != args.class_num)

    # One-hot pseudo labels
    psd_label_onehot = np.zeros_like(pred_cls_bank)
    psd_label_onehot[np.arange(data_num), psd_label_oh] = 1.0
    psd_label_onehot[psd_unknown_flg, :] = 1.0
    psd_label_onehot = psd_label_onehot / (np.sum(psd_label_onehot, axis=-1, keepdims=True) + 1e-5)

    # Log accuracy
    per_class_num = np.zeros(len(class_list))
    pre_class_num = np.zeros_like(per_class_num)
    per_class_correct = np.zeros_like(per_class_num)
    for i, label in enumerate(class_list):
        label_idx = np.where(gt_label_bank == label)[0]
        correct_idx = np.where(psd_label[label_idx] == label)[0]
        pre_class_num[i] = float(len(np.where(psd_label == label)[0]))
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))
    per_class_acc = per_class_correct / (per_class_num + 1e-5)

    args.logger.info("PSD AVG ACC:\t" + "{:.3f}".format(np.mean(per_class_acc)))
    args.logger.info("PSD PER ACC:\t" + "\t".join(["{:.3f}".format(item) for item in per_class_acc]))

    return (jt.array(psd_label_onehot.astype(np.float32)),
            jt.array(pred_cls_bank.astype(np.float32)),
            jt.array(embed_feat_bank.astype(np.float32)),
            jt.array(known_space_basis.astype(np.float32)),
            jt.array(unknown_space_basis.astype(np.float32)),
            jt.array(psd_unknown_flg.astype(np.int32)),
            jt.array(psd_label_weight.astype(np.float32)))


known_space_basis = None
unknown_space_basis = None


def train_target(args, model, train_dataloader, test_dataloader, optimizer, epoch_idx=0.0):
    model.eval()
    (psd_label_onehot_bank, pred_cls_bank, embed_feat_bank,
     known_basis, unknown_basis, psd_unknown_bank,
     psd_label_weight_bank) = obtain_LEAD_pseudo_labels(args, model, test_dataloader, epoch_idx=epoch_idx)

    model.train()
    local_KNN = 4
    all_pred_loss_stack = []
    psd_pred_loss_stack = []
    knn_pred_loss_stack = []
    reg_pred_loss_stack = []

    iter_idx = epoch_idx * len(train_dataloader)
    iter_max = args.epochs * len(train_dataloader)

    for batch_idx, (imgs_train, _, imgs_label, imgs_idx) in enumerate(train_dataloader):
        iter_idx += 1

        psd_label = psd_label_onehot_bank[imgs_idx]  # [B, C]
        psd_weight = psd_label_weight_bank[imgs_idx].unsqueeze(1)  # [B, 1]

        embed_feat, pred_cls = model(imgs_train, apply_softmax=True)

        # L_ce
        psd_pred_loss = jt.sum(-psd_label * psd_weight * jt.log(pred_cls + 1e-5), dim=-1).mean()

        # L_reg
        embed_feat_norm = embed_feat / (jt.norm(embed_feat, p=2, dim=-1, keepdim=True) + 1e-8)
        knfeat_proj_cords = jt.einsum("nd, cd -> nc", embed_feat_norm, known_basis)
        knfeat_proj_norms = jt.norm(knfeat_proj_cords, p=2, dim=-1, keepdim=True)

        unfeat_proj_cords = jt.einsum("nd, cd -> nc", embed_feat_norm, unknown_basis)
        unfeat_proj_norms = jt.norm(unfeat_proj_cords, p=2, dim=-1, keepdim=True)

        feat_proj_norms = jt.concat([knfeat_proj_norms, unfeat_proj_norms], dim=-1)
        feat_proj_probs = nn.softmax(feat_proj_norms, dim=-1)

        psd_unknown_flg = psd_unknown_bank[imgs_idx].long()
        psd_unknown_oh = jt.zeros_like(feat_proj_norms)
        _scatter_src = jt.ones([psd_unknown_flg.shape[0], 1]).float32()
        psd_unknown_oh.scatter_(1, psd_unknown_flg.unsqueeze(1).int32(), _scatter_src)
        feat_reg_loss = jt.sum(-psd_unknown_oh * psd_weight * jt.log(feat_proj_probs + 1e-5), dim=-1).mean()

        # L_con
        with jt.no_grad():
            feat_dist = jt.einsum("bd, nd -> bn", embed_feat_norm, embed_feat_bank)
            _, nn_feat_idx = feat_dist.topk(local_KNN + 1, dim=-1)
            nn_feat_idx_0 = nn_feat_idx[:, 1:]
            # Gather KNN predictions
            nn_preds = []
            for k in range(local_KNN):
                nn_preds.append(pred_cls_bank[nn_feat_idx_0[:, k]])
            nn_pred_cls = jt.stack(nn_preds, dim=1).mean(dim=1)
            # Update banks
            pred_cls_bank[imgs_idx] = pred_cls.detach()
            embed_feat_bank[imgs_idx] = embed_feat_norm.detach()

        knn_pred_loss = jt.sum(-nn_pred_cls * jt.log(pred_cls + 1e-5), dim=-1).mean()

        loss = args.lam_psd * psd_pred_loss + feat_reg_loss + knn_pred_loss
        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.step(loss)

        all_pred_loss_stack.append(loss.item())
        psd_pred_loss_stack.append(psd_pred_loss.item())
        knn_pred_loss_stack.append(knn_pred_loss.item())
        reg_pred_loss_stack.append(feat_reg_loss.item())

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch_idx} [{batch_idx+1}/{len(train_dataloader)}] loss={loss.item():.4f}")

    train_loss_dict = {
        "all_pred_loss": np.mean(all_pred_loss_stack),
        "psd_pred_loss": np.mean(psd_pred_loss_stack),
        "con_pred_loss": np.mean(knn_pred_loss_stack),
        "reg_pred_loss": np.mean(reg_pred_loss_stack),
    }
    return train_loss_dict


def test(args, model, dataloader, src_flg=False):
    model.eval()
    gt_label_stack = []
    pred_cls_stack = []

    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0

    with jt.no_grad():
        for _, imgs_test, imgs_label, _ in dataloader:
            _, pred_cls = model(imgs_test, apply_softmax=True)
            gt_label_stack.append(imgs_label.numpy())
            pred_cls_stack.append(pred_cls.numpy())

    gt_label_all = np.concatenate(gt_label_stack, axis=0)
    pred_cls_all = np.concatenate(pred_cls_stack, axis=0)

    h_score, known_acc, unknown_acc, _ = compute_h_score(
        args, class_list, gt_label_all, pred_cls_all, open_flg, open_thresh=args.w_0)
    return h_score, known_acc, unknown_acc


def main(args):
    device = get_device()
    print(f"Using device: {device}")
    this_dir = os.path.dirname(os.path.abspath(__file__))

    model = SFUniDA(args)

    if args.note is None:
        save_dir = os.path.join(this_dir, "checkpoints", args.dataset,
                                "s_{}_t_{}".format(args.s_idx, args.t_idx),
                                args.target_label_type,
                                "{}_psd_{}".format(args.source_train_type, args.lam_psd))
    else:
        save_dir = os.path.join(this_dir, "checkpoints", args.dataset,
                                "s_{}_t_{}".format(args.s_idx, args.t_idx),
                                args.target_label_type,
                                "{}_psd_{}_{}".format(args.source_train_type, args.lam_psd, args.note))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    args.logger = set_logger(args, log_name="log_target_training.txt")

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        if args.checkpoint.endswith('.pkl'):
            model.load(args.checkpoint)
        else:
            load_pytorch_weights(model, args.checkpoint)
    else:
        print(args.checkpoint)
        raise ValueError("YOU MUST SET THE APPROPRIATE SOURCE CHECKPOINT FOR TARGET MODEL ADAPTATION!!!")

    shutil.copy(os.path.join(this_dir, "train_target.py"), os.path.join(args.save_dir, "train_target.py"))
    shutil.copy(os.path.join(this_dir, "utils/net_utils.py"), os.path.join(args.save_dir, "net_utils.py"))

    param_group = []
    for k, v in model.backbone_layer.named_parameters():
        param_group += [{'params': [v], 'lr': args.lr * 0.1}]
    for k, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': [v], 'lr': args.lr}]
    for k, v in model.class_layer.named_parameters():
        v.stop_grad()

    optimizer = jt.optim.SGD(param_group, lr=args.lr, momentum=0.9,
                              weight_decay=1e-3, nesterov=True)
    optimizer = op_copy(optimizer)

    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()

    target_train_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target",
                                           preload_flg=True, batch_size=args.batch_size,
                                           shuffle=True, num_workers=0, drop_last=True)
    target_test_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target",
                                          preload_flg=True, batch_size=args.batch_size * 2,
                                          shuffle=False, num_workers=0, drop_last=False)

    notation_str = "\n=======================================================\n"
    notation_str += "   START TRAINING ON THE TARGET:{} BASED ON SOURCE:{}  \n".format(args.t_idx, args.s_idx)
    notation_str += "======================================================="
    args.logger.info(notation_str)

    best_h_score = 0.0
    best_known_acc = 0.0
    best_unknown_acc = 0.0

    for epoch_idx in range(args.epochs):
        loss_dict = train_target(args, model, target_train_dataset, target_test_dataset, optimizer, epoch_idx)
        args.logger.info("Epoch: {}/{}, train_all_loss:{:.3f}, "
                         "train_psd_loss:{:.3f}, train_reg_loss:{:.3f}, train_con_loss:{:.3f}".format(
            epoch_idx + 1, args.epochs,
            loss_dict["all_pred_loss"], loss_dict["psd_pred_loss"],
            loss_dict["reg_pred_loss"], loss_dict["con_pred_loss"]))

        hscore, knownacc, unknownacc = test(args, model, target_test_dataset, src_flg=False)
        args.logger.info("Current: H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(
            hscore, knownacc, unknownacc))

        if args.target_label_type == 'PDA' or args.target_label_type == 'CLDA':
            if knownacc >= best_known_acc:
                best_h_score = hscore
                best_known_acc = knownacc
                best_unknown_acc = unknownacc
        else:
            if hscore >= best_h_score:
                best_h_score = hscore
                best_known_acc = knownacc
                best_unknown_acc = unknownacc

        args.logger.info("Best   : H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(
            best_h_score, best_known_acc, best_unknown_acc))


if __name__ == "__main__":
    args = build_args()
    set_random_seed(args.seed)

    # Try PyTorch checkpoint first, then Jittor checkpoint
    pt_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "checkpoints", args.dataset, "source_{}".format(args.s_idx),
                           "source_{}_{}".format(args.source_train_type, args.target_label_type),
                           "latest_source_checkpoint.pth")
    jt_ckpt = os.path.join("checkpoints", args.dataset, "source_{}".format(args.s_idx),
                           "source_{}_{}".format(args.source_train_type, args.target_label_type),
                           "latest_source_checkpoint.pkl")

    if os.path.isfile(pt_ckpt):
        args.checkpoint = pt_ckpt
    elif os.path.isfile(jt_ckpt):
        args.checkpoint = jt_ckpt
    else:
        args.checkpoint = pt_ckpt  # will raise error

    args.reset = False
    main(args)
