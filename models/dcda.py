import os

import clip
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from datasets.composition_dataset import CompositionDataset
from flags import DATA_FOLDER
from models.automatic_weighted_loss import AutomaticWeightedLoss
from models.mlp import MLP
from models.vadapter import VAdapter
from .clip_components import (
    ResidualAttentionBlock,
    VisionTransformer,
    Transformer,
    LayerNorm,
)
from .gcn import GCN

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class ClipImageEncoder(nn.Module):
    def __init__(
            self,
            image_resolution: int,
            vision_layers: int,
            vision_width: int,
            vision_patch_size: int,
            vision_embed_dim: int,
            config
    ):
        super().__init__()
        if config is not None:
            if not config.vision_share:
                self.attr_attn_adapter = None
                self.obj_attn_adapter = None
            else:
                self.attr_attn_adapter = VAdapter(vision_width, vision_width // 64)
                self.obj_attn_adapter = VAdapter(vision_width, vision_width // 64)
        else:
            self.attr_attn_adapter = VAdapter(vision_width, vision_width // 64)
            self.obj_attn_adapter = VAdapter(vision_width, vision_width // 64)
        vision_heads = vision_width // 64
        self.model = VisionTransformer(
            input_resolution=image_resolution,
            layers=vision_layers,
            width=vision_width,
            patch_size=vision_patch_size,
            heads=vision_heads,
            output_dim=vision_embed_dim,
            config=config,
            visual_attr_adapter=None,
            visual_obj_adapter=None
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.model.positional_embedding, std=0.01)
        proj_std = (self.model.transformer.width ** -0.5) * (
                (2 * self.model.transformer.layers) ** -0.5
        )
        attn_std = self.model.transformer.width ** -0.5
        fc_std = (2 * self.model.transformer.width) ** -0.5
        for block in self.model.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        nn.init.normal_(self.model.proj, std=self.model.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.model.conv1.weight.dtype

    def forward(self, image):
        return self.model(image)


class ClipTextEncoder(nn.Module):
    def __init__(
            self,
            context_length: int,
            vocab_size: int,
            text_width: int,
            text_heads: int,
            text_layers: int,
            text_embed_dim: int,
            config=None,
            dataset=None,
    ):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.l_adapter = LAdapter(config, dataset, text_width)
        self.context_length = context_length
        self.transformer = Transformer(
            width=text_width,
            layers=text_layers,
            heads=text_heads,
            attn_mask=self.build_attention_mask(),
            l_adapter=self.l_adapter,
            config=config
        )
        self.dtype = torch.float32
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, text_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, text_width)
        )
        self.ln_final = LayerNorm(text_width)
        self.text_projection = nn.Parameter(torch.empty(text_width, text_embed_dim))
        self.soft_embeddings = nn.Parameter(
            torch.rand(len(dataset.attrs) + len(dataset.objs), text_width).to(config.device)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.initialize_parameters()

    def init_clip_embeddings(self):
        all_attrs = self.dataset.attrs
        all_objs = self.dataset.objs
        objects = [obj.replace(".", " ").lower() for obj in all_objs]
        attributes = [attr.replace(".", " ").lower() for attr in all_attrs]
        tokenized = torch.cat(
            [
                clip.tokenize(tok, context_length=self.config.context_length)
                for tok in attributes + objects
            ]
        )
        orig_token_embedding = self.token_embedding(tokenized.to(self.config.device))
        soft_embeddings = torch.zeros(
            (len(attributes) + len(objects), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_embeddings[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)
        self.soft_embeddings = nn.Parameter(soft_embeddings.to(self.config.device))

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * (
                (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            if not isinstance(block, ResidualAttentionBlock):
                continue
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def set_soft_embeddings(self, se):
        if se.shape == self.soft_embeddings.shape:
            self.state_dict()["soft_embeddings"].copy_(se)
        else:
            raise RuntimeError(
                f"Error: Incorrect Soft Embedding Shape {se.shape}, Expecting {self.soft_embeddings.shape}!"
            )

    def construct_token_tensors(self, idx, type_phase, pair_nums):
        def get_token_ids(config):
            prompts = {
                "pair": "a photo of X X",
                "attr": "a photo of X object",
                "obj": "a photo of X",
            }
            token_ids = {}
            for type, prompt in prompts.items():
                token_ids_item = clip.tokenize(
                    [prompt],
                    context_length=config.context_length,
                ).to(config.device)
                token_ids[type] = token_ids_item
            return token_ids

        token_ids = get_token_ids(self.config)
        if type_phase == "evaluate":
            pair_token_ids = token_ids["pair"].repeat(len(self.dataset.pairs), 1)
        elif type_phase == "train":
            pair_token_ids = token_ids["pair"].repeat(len(self.dataset.train_pairs), 1)
        else:
            raise Exception("type_phase error")
        soft_embeddings = self.dropout(self.soft_embeddings)
        if self.config.multi_prompt_type:
            attr_idx = idx[: len(self.dataset.attrs), 0]
            attr_token_ids = token_ids["attr"].repeat(len(self.dataset.attrs), 1)
            attr_token_tensor = self.token_embedding(attr_token_ids)
            attr_eos_idx = int(token_ids["attr"][0].argmax())
            attr_token_tensor[:, attr_eos_idx - 2, :] = soft_embeddings[attr_idx]
            obj_idx = idx[
                      len(self.dataset.attrs): len(self.dataset.attrs)
                                               + len(self.dataset.objs),
                      0,
                      ]
            obj_token_ids = token_ids["obj"].repeat(len(self.dataset.objs), 1)
            obj_token_tensor = self.token_embedding(obj_token_ids)
            obj_eos_idx = int(token_ids["obj"][0].argmax())
            obj_token_tensor[:, obj_eos_idx - 1, :] = soft_embeddings[obj_idx]
            pair_attr_idx, pair_obj_idx = (
                idx[len(self.dataset.attrs) + len(self.dataset.objs):, 0],
                idx[len(self.dataset.attrs) + len(self.dataset.objs):, 1],
            )
        else:
            pair_attr_idx, pair_obj_idx = (
                idx[:, 0],
                idx[:, 1],
            )
        pair_token_tensor = self.token_embedding(pair_token_ids)
        pair_eos_idx = int(token_ids["pair"][0].argmax())
        pair_token_tensor[:, pair_eos_idx - 2, :] = soft_embeddings[pair_attr_idx]
        pair_token_tensor[:, pair_eos_idx - 1, :] = soft_embeddings[
            pair_obj_idx + len(self.dataset.attrs)
            ]
        if self.config.multi_prompt_type:
            token_tensor = torch.cat(
                [attr_token_tensor, obj_token_tensor, pair_token_tensor]
            )
            eot_indice = (
                    [attr_eos_idx] * len(self.dataset.attrs)
                    + [obj_eos_idx] * len(self.dataset.objs)
                    + [pair_eos_idx] * pair_nums
            )
        else:
            token_tensor = pair_token_tensor
            eot_indice = [pair_eos_idx] * pair_nums
        return token_tensor, eot_indice

    def forward_separate(self, idx):
        if not self.config.multi_prompt_type:
            if len(self.dataset.pairs) == len(idx):
                type_phase = "evaluate"
            elif len(self.dataset.train_pairs) == len(idx):
                type_phase = "train"
            else:
                raise Exception(
                    f"error,idx shape:{idx.shape}，pairs len:{len(self.dataset.pairs)}，train_pairs len:{len(self.dataset.train_pairs)}")
        else:
            if len(self.dataset.attrs) + len(self.dataset.objs) + len(
                    self.dataset.pairs
            ) == len(idx):
                type_phase = "evaluate"
            elif len(self.dataset.attrs) + len(self.dataset.objs) + len(
                    self.dataset.train_pairs
            ) == len(idx):
                type_phase = "train"
            else:
                raise Exception(
                    f"error,idx shape:{idx.shape}，pairs len:{len(self.dataset.pairs)}，train_pairs len:{len(self.dataset.train_pairs)}")
        pair_nums = (
            len(self.dataset.train_pairs)
            if type_phase == "train"
            else len(self.dataset.pairs)
        )
        x, eot_idx = self.construct_token_tensors(idx, type_phase, pair_nums)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer((x, torch.tensor(eot_idx).to(self.config.device)))
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), eot_idx] @ self.text_projection
        return x

    def forward_complete1(self, text):
        eot_idx = text.argmax(dim=-1)
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        return (x, eot_idx)

    def forward_complete_block(self, text):
        x, eot_idx = text
        for i in range(9):
            x, eot_idx = self.transformer.resblocks[i]((x, eot_idx))
        return (x, eot_idx)

    def forward_complete_attn(self, text, i):
        x, eot_idx = text
        x.add_(self.transformer.resblocks[i].attention(self.transformer.resblocks[i].ln_1(x)))
        return (x, eot_idx)

    def forward_complete_graph(self, text, i):
        x, eot_idx = text
        x.add_(self.transformer.resblocks[i].l_adapter(text)[0])
        return (x.cpu(), eot_idx)

    def forward_complete_mlp(self, text, i):
        x, eot_idx = text
        x.add_(self.transformer.resblocks[i].mlp(self.transformer.resblocks[i].ln_2(x)))
        return (x, eot_idx)

    def forward_complete_post(self, text):
        x, eot_idx = text
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), eot_idx] @ self.text_projection
        return x

    def forward_complete(self, text):
        eot_idx = text.argmax(dim=-1)
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer((x, eot_idx))
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), eot_idx] @ self.text_projection
        return x

    def forward(self, input):
        if self.config.embedding_type == "complete":
            return self.forward_complete(input)
        elif self.config.embedding_type == "separate":
            return self.forward_separate(input)


class LAdapter(nn.Module):
    def __init__(self, config, dataset: CompositionDataset, embed_dim):
        super().__init__()
        self.dtype = torch.float32
        self.config = config
        self.dataset = dataset
        self.num_attrs, self.num_objs, self.num_pairs = (
            len(dataset.attrs),
            len(dataset.objs),
            len(dataset.pairs),
        )
        self.pairs = dataset.pairs

        all_words = list(dataset.attrs) + list(dataset.objs)
        self.num_words = len(all_words)
        self.ln_1 = LayerNorm(embed_dim)
        if not config.graph_init:
            self.train_adj = self.adj_from_pairs(self.dataset.train_pairs)
            self.all_adj = self.adj_from_pairs(self.dataset.pairs)
            torch.save(
                {"all_adj": self.all_adj, "train_adj": self.train_adj},
                f"{DATA_FOLDER}/graph_init/{config.dataset}.t7",
            )
        else:
            graph = torch.load(config.graph_init)
            self.all_adj = graph["all_adj"]
            self.train_adj = graph["train_adj"]
        self.current_adj = self.train_adj if config.phase == "train" else self.all_adj
        self.gcn = GCN(self.current_adj, embed_dim, embed_dim, config.hidden_layers)

    def switch_adj(self, adj_key):
        adj_dict = {"all": self.all_adj, "train": self.train_adj}
        self.phase = "train" if adj_key == "train" else "test"
        self.gcn.set_adj(adj_dict[adj_key])

    def update_dict(self, wdict, row, col, data):
        wdict["row"].append(row)
        wdict["col"].append(col)
        wdict["data"].append(data)

    def adj_from_pairs(self, pairs):
        def edges_from_pairs(pairs):
            weight_dict = {"data": [], "row": [], "col": []}
            for i in range(self.num_words):
                self.update_dict(weight_dict, i, i, 1.0)
            for idx, (attr, obj) in enumerate(pairs):
                attr_idx, obj_idx = (
                    self.dataset.attr2idx[attr],
                    self.dataset.obj2idx[obj] + self.num_attrs,
                )
                self.update_dict(weight_dict, attr_idx, obj_idx, 1.0)
                self.update_dict(weight_dict, obj_idx, attr_idx, 1.0)
                node_id = idx + self.num_words
                self.update_dict(weight_dict, node_id, node_id, 1.0)
                self.update_dict(weight_dict, node_id, attr_idx, 1.0)
                self.update_dict(weight_dict, node_id, obj_idx, 1.0)
                self.update_dict(weight_dict, attr_idx, node_id, 1.0)
                self.update_dict(weight_dict, obj_idx, node_id, 1.0)
            return weight_dict

        edges = edges_from_pairs(pairs)
        adj = sp.csr_matrix(
            (edges["data"], (edges["row"], edges["col"])),
            shape=(len(pairs) + self.num_words, len(pairs) + self.num_words)
        )
        return adj

    def forward(self, input):
        embeddings, eot_idx = input
        eot_embeddings = embeddings[eot_idx, torch.arange(embeddings.shape[1])].clone()
        if not self.config.multi_prompt_type:
            # 1prompt
            attr_idx2pair_idx = self.dataset.all_attr_idx2pair_idx
            obj_idx2pair_idx = self.dataset.all_obj_idx2pair_idx
            attr_embeddings = []
            for i in range(len(self.dataset.attrs)):
                attr_embeddings.append(
                    torch.mean(
                        embeddings[eot_idx - 2, torch.arange(embeddings.shape[1])][
                            attr_idx2pair_idx[i]
                        ],
                        axis=0,
                    )
                )
            obj_embeddings = []
            for i in range(len(self.dataset.objs)):
                obj_embeddings.append(
                    torch.mean(
                        embeddings[eot_idx - 1, torch.arange(embeddings.shape[1])][
                            obj_idx2pair_idx[i]
                        ],
                        axis=0,
                    )
                )
            attr_embeddings, obj_embeddings = torch.stack(attr_embeddings), torch.stack(
                obj_embeddings
            )
            eot_embeddings = torch.cat(
                (attr_embeddings, obj_embeddings, eot_embeddings), 0
            )
            eot_embeddings = self.gcn(eot_embeddings)
            embeddings[eot_idx, torch.arange(embeddings.shape[1])] += eot_embeddings[
                                                                      len(self.dataset.attrs) + len(self.dataset.objs):
                                                                      ]
            return embeddings, eot_idx
        else:
            eot_embeddings = self.gcn(eot_embeddings)
            embeddings[eot_idx, torch.arange(embeddings.shape[1])] += eot_embeddings
            return embeddings, eot_idx


class DCDA(nn.Module):
    def __init__(
            self,
            config,
            dataset: CompositionDataset,
            image_resolution: int,
            vision_layers: int,
            vision_width: int,
            vision_patch_size: int,
            vision_embed_dim: int,
            context_length: int,
            vocab_size: int,
            text_width: int,
            text_heads: int,
            text_layers: int,
            text_embed_dim: int
    ) -> None:
        super().__init__()
        self.dtype = torch.float32
        self.config = config
        self.dataset = dataset
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.attr_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.counterpart_attr_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.counterpart_obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.attr_weight = nn.Parameter(torch.ones([]))
        self.obj_weight = nn.Parameter(torch.ones([]))
        self.pair_weight = nn.Parameter(torch.ones([]))
        self.visual_encoder = ClipImageEncoder(
            image_resolution,
            vision_layers,
            vision_width,
            vision_patch_size,
            vision_embed_dim,
            config
        )
        self.text_encoder = ClipTextEncoder(
            context_length,
            vocab_size,
            text_width,
            text_heads,
            text_layers,
            text_embed_dim,
            config,
            dataset,
        )
        self.img_dropout = nn.Dropout(p=0.5)
        self.attr_transform = nn.Linear(vision_embed_dim, vision_embed_dim)
        nn.init.xavier_uniform_(self.attr_transform.weight)
        nn.init.constant_(self.attr_transform.bias, 0.0)
        self.obj_transform = nn.Linear(vision_embed_dim, vision_embed_dim)
        nn.init.xavier_uniform_(self.obj_transform.weight)
        nn.init.constant_(self.obj_transform.bias, 0.0)
        self.automatic_weighted_loss = AutomaticWeightedLoss(3)
        self.text_fusion_mlp = MLP(text_width * 3, text_width, num_layers=2)

    def forward(self, image, text, attr_img, obj_img):
        img_self, img_attr, img_obj, img_same_attr, img_same_obj = self.visual_encoder([image, attr_img, obj_img])
        text_features = self.text_encoder(text)
        attr_features = text_features[: len(self.dataset.attrs), :]
        obj_features = text_features[
                       len(self.dataset.attrs): len(self.dataset.attrs)
                                                + len(self.dataset.objs),
                       :,
                       ]
        text_features = text_features[
                        len(self.dataset.attrs) + len(self.dataset.objs):, :
                        ]
        attr_features = attr_features / attr_features.norm(dim=1, keepdim=True)
        obj_features = obj_features / obj_features.norm(dim=1, keepdim=True)
        train_pairwise_attr_rep = attr_features[self.dataset.train_pairs2attr_idx]
        train_pairwise_obj_rep = obj_features[self.dataset.train_pairs2obj_idx]
        img_self = self.img_dropout(img_self)
        img_self = img_self / img_self.norm(
            dim=1, keepdim=True
        )
        img_attr = self.img_dropout(img_attr)
        img_attr = img_attr / img_attr.norm(
            dim=1, keepdim=True
        )
        img_obj = self.img_dropout(img_obj)
        img_obj = img_obj / img_obj.norm(
            dim=1, keepdim=True
        )
        text_features = text_features / text_features.norm(
            dim=1, keepdim=True
        )
        logit_scale = self.logit_scale.exp()
        logits_per_image = self.pair_weight * logit_scale * img_self @ text_features.t()
        if self.config.multi_prompt_type:
            attr_logit_scale = self.attr_logit_scale.exp()
            obj_logit_scale = self.obj_logit_scale.exp()
            counterpart_attr_logit_scale = self.counterpart_attr_logit_scale.exp()
            counterpart_obj_logit_scale = self.counterpart_obj_logit_scale.exp()
            attr_logits_per_image = attr_logit_scale * img_attr @ attr_features.t()
            counterpart_attr_logits_per_image = counterpart_attr_logit_scale * img_attr @ attr_features.t()
            obj_logits_per_image = obj_logit_scale * img_obj @ obj_features.t()
            counterpart_obj_logits_per_image = counterpart_obj_logit_scale * img_obj @ obj_features.t()
            logits_per_image += self.attr_weight * attr_logit_scale * img_attr @ train_pairwise_attr_rep.t() + self.obj_weight * obj_logit_scale * img_obj @ train_pairwise_obj_rep.t()
            return logits_per_image, attr_logits_per_image, obj_logits_per_image, counterpart_attr_logits_per_image, counterpart_obj_logits_per_image
        return logits_per_image, logits_per_image.t()
