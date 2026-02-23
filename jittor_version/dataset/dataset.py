import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import jittor as jt
from jittor.dataset import Dataset
import jittor.transform as transforms


def train_transform(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.ImageNormalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    ])


def test_transform(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.ImageNormalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    ])


class SFUniDADataset(Dataset):
    def __init__(self, args, data_dir, data_list, d_type, preload_flg=True,
                 batch_size=64, shuffle=False, num_workers=0, drop_last=False):
        super().__init__()

        self.d_type = d_type
        self.dataset = args.dataset
        self.preload_flg = preload_flg

        self.shared_class_num = args.shared_class_num
        self.source_private_class_num = args.source_private_class_num
        self.target_private_class_num = args.target_private_class_num

        self.shared_classes = [i for i in range(args.shared_class_num)]
        self.source_private_classes = [i + args.shared_class_num for i in range(args.source_private_class_num)]

        if args.dataset == "Office" and args.target_label_type == "OSDA":
            self.target_private_classes = [i + args.shared_class_num + args.source_private_class_num + 10
                                           for i in range(args.target_private_class_num)]
        else:
            self.target_private_classes = [i + args.shared_class_num + args.source_private_class_num
                                           for i in range(args.target_private_class_num)]

        self.source_classes = self.shared_classes + self.source_private_classes
        self.target_classes = self.shared_classes + self.target_private_classes

        self.data_dir = data_dir
        self.data_list_raw = [item.strip().split() for item in data_list]

        if self.d_type == "source":
            self.data_list_raw = [item for item in self.data_list_raw if int(item[1]) in self.source_classes]
        else:
            self.data_list_raw = [item for item in self.data_list_raw if int(item[1]) in self.target_classes]

        self.pre_loading()

        self.train_trans = train_transform()
        self.test_trans = test_transform()

        self.set_attrs(
            batch_size=batch_size,
            total_len=len(self.data_list_raw),
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last
        )

    def pre_loading(self):
        if "Office" in self.dataset and self.preload_flg:
            resize_trans = transforms.Resize((256, 256))
            print("Dataset Pre-Loading Started ....")
            self.img_list = []
            for item in tqdm(self.data_list_raw, ncols=60):
                img = Image.open(os.path.join(self.data_dir, item[0])).convert("RGB")
                img = resize_trans(img)
                self.img_list.append(img)
            print("Dataset Pre-Loading Done!")

    def __getitem__(self, img_idx):
        img_f, img_label = self.data_list_raw[img_idx]
        if "Office" in self.dataset and self.preload_flg:
            img = self.img_list[img_idx]
        else:
            img = Image.open(os.path.join(self.data_dir, img_f)).convert("RGB")

        if self.d_type == "source":
            img_label = int(img_label)
        else:
            img_label = int(img_label) if int(img_label) in self.source_classes else len(self.source_classes)

        img_train = self.train_trans(img)
        img_test = self.test_trans(img)

        return img_train, img_test, img_label, img_idx
