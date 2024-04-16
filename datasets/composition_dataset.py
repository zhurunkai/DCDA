from itertools import product

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    RandomPerspective,
    RandomRotation,
    Resize,
    ToTensor,
)
from torchvision.transforms.transforms import RandomResizedCrop
import random

from utils.core import find_index

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def transform_image(split="train", imagenet=False):
    if imagenet:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform


class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = "%s/%s" % (self.img_dir, img)
        img = Image.open(file).convert("RGB")
        return img


class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split="compositional-split-natural",
            open_world=False,
            imagenet=False,
            config=None
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world
        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + "/images/")

        (
            self.attrs,
            self.objs,
            self.pairs,
            self.train_pairs,
            self.val_pairs,
            self.test_pairs,
        ) = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == "train":
            self.data = self.train_data
        elif self.phase == "val":
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print(
            "# train pairs: %d | # val pairs: %d | # test pairs: %d"
            % (len(self.train_pairs), len(self.val_pairs), len(self.test_pairs))
        )
        print(
            "# train images: %d | # val images: %d | # test images: %d"
            % (len(self.train_data), len(self.val_data), len(self.test_data))
        )

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )
        self.train_in_pairs_index = []
        for train_pair in self.train_pairs:
            for index, pair in enumerate(self.pairs):
                if train_pair == pair:
                    self.train_in_pairs_index.append(index)
        if len(self.train_in_pairs_index) != len(self.train_pairs):
            raise (
                f"The length of train_in-pairs_index is {len(self.train_in_pairs_index)}, the length of train_pair is {len(self.train_pairs)}"
            )

        attr2train_pair = {}
        obj2train_pair = {}
        attr_idx2train_pair_idx = [[] for _ in self.attrs]
        obj_idx2train_pair_idx = [[] for _ in self.objs]
        for index, pair in enumerate(self.train_pairs):
            if pair[0] in attr2train_pair.keys():
                attr2train_pair[pair[0]].append(pair)
            else:
                attr2train_pair[pair[0]] = [pair]
            if pair[1] in obj2train_pair.keys():
                obj2train_pair[pair[1]].append(pair)
            else:
                obj2train_pair[pair[1]] = [pair]
            attr_idx2train_pair_idx[find_index(self.attrs, pair[0])].append(index)
            obj_idx2train_pair_idx[find_index(self.objs, pair[1])].append(index)
        self.attr2train_pair = attr2train_pair
        self.obj2train_pair = obj2train_pair
        self.attr_idx2train_pair_idx = attr_idx2train_pair_idx
        self.obj_idx2train_pair_idx = obj_idx2train_pair_idx

        self.get_concept2data()

        reciprocal_frequency_for_train_pairs = [1.0 for _ in self.train_pairs]
        for train_pair, pair2data in self.train_pair2data.items():
            reciprocal_frequency_for_train_pairs[
                find_index(self.train_pairs, train_pair)
            ] = 1.0 / len(pair2data)
        self.reciprocal_frequency_for_train_pairs = torch.tensor(
            reciprocal_frequency_for_train_pairs
        )

        train_pair_idx2init_attr_percent = []
        train_pair_idx2init_obj_percent = []
        for index, train_pair in enumerate(self.train_pairs):
            attr, obj = train_pair
            train_pair_idxs_attr_indice = torch.tensor(
                self.attr_idx2train_pair_idx[self.attr2idx[attr]]
            )
            train_pair_idxs_obj_indice = torch.tensor(
                self.obj_idx2train_pair_idx[self.obj2idx[obj]]
            )
            train_pair_idxs_attr_percent = torch.nn.functional.softmax(
                self.reciprocal_frequency_for_train_pairs[train_pair_idxs_attr_indice],
                dim=0,
            )
            train_pair_idxs_obj_percent = torch.nn.functional.softmax(
                self.reciprocal_frequency_for_train_pairs[train_pair_idxs_obj_indice],
                dim=0,
            )
            train_pair_idx2init_attr_percent.append(train_pair_idxs_attr_percent)
            train_pair_idx2init_obj_percent.append(train_pair_idxs_obj_percent)
        self.final_attr_percent = train_pair_idx2init_attr_percent
        self.final_obj_percent = train_pair_idx2init_obj_percent

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.0

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for a, o in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for a, o in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

        pairs2attr_idx = []
        pairs2obj_idx = []
        for attr, obj in self.pairs:
            pairs2attr_idx.append(self.attr2idx[attr])
            pairs2obj_idx.append(self.obj2idx[obj])
        self.pairs2attr_idx = torch.tensor(pairs2attr_idx)
        self.pairs2obj_idx = torch.tensor(pairs2obj_idx)

        train_pairs2attr_idx = []
        train_pairs2obj_idx = []
        for attr, obj in self.train_pairs:
            train_pairs2attr_idx.append(self.attr2idx[attr])
            train_pairs2obj_idx.append(self.obj2idx[obj])
        self.train_pairs2attr_idx = torch.tensor(train_pairs2attr_idx)
        self.train_pairs2obj_idx = torch.tensor(train_pairs2obj_idx)

    def get_concept2data(self):
        obj2data = {}
        attr2data = {}
        train_pair2data = {}
        for i, data_item in enumerate(self.train_data):
            new_data_item = [data_item[0], i]
            if data_item[1] in attr2data.keys():
                attr2data[data_item[1]].append(new_data_item)
            else:
                attr2data[data_item[1]] = [new_data_item]
            if data_item[2] in obj2data.keys():
                obj2data[data_item[2]].append(new_data_item)
            else:
                obj2data[data_item[2]] = [new_data_item]
            if (data_item[1], data_item[2]) in train_pair2data.keys():
                train_pair2data[(data_item[1], data_item[2])].append([data_item[0], i])
            else:
                train_pair2data[(data_item[1], data_item[2])] = [new_data_item]
        self.obj2data = obj2data
        self.attr2data = attr2data
        self.train_pair2data = train_pair2data

    def get_split_info(self):
        data = torch.load(self.root + "/metadata_{}.t7".format(self.split))
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = (
                instance["image"],
                instance["attr"],
                instance["obj"],
                instance["set"],
            )

            if attr == "NA" or (attr, obj) not in self.pairs or settype == "NA":
                continue

            data_i = [image, attr, obj]
            if settype == "train":
                train_data.append(data_i)
            elif settype == "val":
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, "r") as f:
                pairs = f.read().strip().split("\n")
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            "%s/%s/train_pairs.txt" % (self.root, self.split)
        )
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            "%s/%s/val_pairs.txt" % (self.root, self.split)
        )
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            "%s/%s/test_pairs.txt" % (self.root, self.split)
        )

        all_attrs, all_objs = sorted(list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
            list(set(tr_objs + vl_objs + ts_objs))
        )
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == "train":
            pair = (attr, obj)
            train_pair_idx = self.train_pair_to_idx[pair]
            attr_index = torch.multinomial(self.final_attr_percent[train_pair_idx], 1)[
                0
            ]
            obj_index = torch.multinomial(self.final_obj_percent[train_pair_idx], 1)[0]
            same_attr_train_pair_idx = self.attr_idx2train_pair_idx[
                self.attr2idx[attr]
            ][attr_index]
            same_obj_train_pair_idx = self.obj_idx2train_pair_idx[self.obj2idx[obj]][
                obj_index
            ]
            new_attr = random.choice(
                self.train_pair2data[self.train_pairs[same_attr_train_pair_idx]]
            )
            while len(self.train_pair2data[self.train_pairs[same_attr_train_pair_idx]]) != 1 and new_attr[1] == index:
                new_attr = random.choice(
                    self.train_pair2data[self.train_pairs[same_attr_train_pair_idx]]
                )
            new_obj = random.choice(
                self.train_pair2data[self.train_pairs[same_obj_train_pair_idx]]
            )
            while len(self.train_pair2data[self.train_pairs[same_obj_train_pair_idx]]) != 1 and new_obj[1] == index:
                new_obj = random.choice(
                    self.train_pair2data[self.train_pairs[same_obj_train_pair_idx]]
                )
            attr_img = self.loader(new_attr[0])
            attr_img = self.transform(attr_img)
            obj_img = self.loader(new_obj[0])
            obj_img = self.transform(obj_img)
            data = [
                img,
                self.attr2idx[attr],
                self.obj2idx[obj],
                self.pair2idx[(attr, obj)],
                self.train_pair_to_idx[(attr, obj)],
                attr_img,
                obj_img,
            ]
        else:
            data = [
                img,
                self.attr2idx[attr],
                self.obj2idx[obj],
                self.pair2idx[(attr, obj)],
            ]

        return data

    def __len__(self):
        return len(self.data)
