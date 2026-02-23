import jittor as jt
import jittor.nn as nn
from jittor import init, models
import numpy as np


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            init.zero_(m.bias)
    elif classname.find('BatchNorm') != -1:
        init.gauss_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            init.zero_(m.bias)
    elif classname.find('Linear') != -1:
        init.xavier_gauss_(m.weight)
        if m.bias is not None:
            init.zero_(m.bias)


class ResBase(nn.Module):
    def __init__(self, res_name):
        super().__init__()
        if res_name == "resnet50":
            model_resnet = models.resnet50(pretrained=True)
        elif res_name == "resnet101":
            model_resnet = models.resnet101(pretrained=True)
        else:
            raise ValueError(f"Unknown ResNet arch: {res_name}")

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.backbone_feat_dim = model_resnet.fc.in_features

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        return x


class Embedding(nn.Module):
    def __init__(self, feature_dim, embed_dim=256, type="ori"):
        super().__init__()
        self.bn = nn.BatchNorm1d(embed_dim, affine=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, embed_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def execute(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class WeightNormLinear(nn.Module):
    """Manual weight normalization since Jittor lacks nn.weight_norm."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_v = jt.randn(out_features, in_features)
        self.weight_g = jt.ones(out_features, 1)
        self.bias = jt.zeros(out_features)
        # Initialize
        init.xavier_gauss_(self.weight_v)
        # Compute initial g
        norms = jt.norm(self.weight_v, p=2, dim=1, keepdim=True)
        self.weight_g = norms.detach()

    @property
    def weight(self):
        """Compute weight = g * v / ||v||"""
        norms = jt.norm(self.weight_v, p=2, dim=1, keepdim=True)
        return self.weight_g * self.weight_v / (norms + 1e-8)

    def execute(self, x):
        w = self.weight
        return nn.matmul_transpose(x, w) + self.bias


class Classifier(nn.Module):
    def __init__(self, embed_dim, class_num, type="linear"):
        super().__init__()
        self.type = type
        if type == 'wn':
            self.fc = WeightNormLinear(embed_dim, class_num)
        else:
            self.fc = nn.Linear(embed_dim, class_num)
            self.fc.apply(init_weights)

    def execute(self, x):
        x = self.fc(x)
        return x


class SFUniDA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone_arch = args.backbone_arch
        self.embed_feat_dim = args.embed_feat_dim
        self.class_num = args.class_num

        if "resnet" in self.backbone_arch:
            self.backbone_layer = ResBase(self.backbone_arch)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_arch}")

        self.backbone_feat_dim = self.backbone_layer.backbone_feat_dim
        self.feat_embed_layer = Embedding(self.backbone_feat_dim, self.embed_feat_dim, type="bn")
        self.class_layer = Classifier(self.embed_feat_dim, class_num=self.class_num, type="wn")

    def get_embed_feat(self, input_imgs):
        backbone_feat = self.backbone_layer(input_imgs)
        embed_feat = self.feat_embed_layer(backbone_feat)
        return embed_feat

    def execute(self, input_imgs, apply_softmax=True):
        backbone_feat = self.backbone_layer(input_imgs)
        embed_feat = self.feat_embed_layer(backbone_feat)
        cls_out = self.class_layer(embed_feat)
        if apply_softmax:
            cls_out = nn.softmax(cls_out, dim=1)
        return embed_feat, cls_out


def load_pytorch_weights(model, pytorch_ckpt_path):
    """Load PyTorch checkpoint weights into Jittor model.
    Handles weight_norm parameter mapping. Skips num_batches_tracked."""
    import torch
    ckpt = torch.load(pytorch_ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict']

    jt_params = {k: v for k, v in model.named_parameters()}
    loaded, skipped = 0, 0

    for pt_key, pt_val in state_dict.items():
        if 'num_batches_tracked' in pt_key:
            skipped += 1
            continue

        np_val = pt_val.cpu().numpy()

        if pt_key in jt_params:
            if jt_params[pt_key].shape == list(np_val.shape):
                jt_params[pt_key].assign(jt.array(np_val))
                loaded += 1
            else:
                print(f"  Shape mismatch for {pt_key}: jt={jt_params[pt_key].shape} vs pt={np_val.shape}")
                skipped += 1
        else:
            # Try direct attribute access for non-parameter state
            try:
                parts = pt_key.split('.')
                obj = model
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                attr_name = parts[-1]
                if hasattr(obj, attr_name):
                    param = getattr(obj, attr_name)
                    if isinstance(param, jt.Var):
                        param.assign(jt.array(np_val))
                        loaded += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    print(f"Loaded {loaded} params from PyTorch checkpoint, skipped {skipped} ({pytorch_ckpt_path})")
