import datetime
import json
import os
import pickle
import pprint

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader

from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import return_dataset_paths
from evaluate import main as evaluate_test
from flags import parser
from models.dcda import DCDA
from models.load_model import load_clip
from split_evaluate import main as split_evaluate_test
from utils import set_seed, path2optional_bool, str2bool, load_args
from utils.text_encode import all_type_separate, all_3type_complete

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.cuda.is_available()
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def epoch_evaluate(
        model: DCDA, config, epoch, epoch_train_losses
):
    model.eval()
    results = {}
    with torch.no_grad():
        model.text_encoder.l_adapter.switch_adj("all")
        if config.open_world:
            results, result_path = split_evaluate_test(config, model, is_train=True, epoch=epoch)
        else:
            results, result_path = evaluate_test(config, model, is_train=True, epoch=epoch)
        model.text_encoder.l_adapter.switch_adj("train")
        results["train_loss"] = np.mean(epoch_train_losses)
        results["weights0"] = [model.automatic_weighted_loss.params[0].item(),
                               0.5 / (model.automatic_weighted_loss.params[0].item() ** 2)]
        results["weights1"] = [model.automatic_weighted_loss.params[1].item(),
                               0.5 / (model.automatic_weighted_loss.params[1].item() ** 2)]
        results["weights2"] = [model.automatic_weighted_loss.params[2].item(),
                               0.5 / (model.automatic_weighted_loss.params[2].item() ** 2)]
        results["attr_weight"] = model.attr_weight.item()
        results["obj_weight"] = model.obj_weight.item()
        results["pair_weight"] = model.pair_weight.item()
        with open(result_path, "w+") as fp:
            json.dump(results, fp)
    model.train()
    return results


def train_model(
        model: DCDA, optimizer, train_dataset: CompositionDataset, config, scaler
):
    def printTimeStr(desc, epoch=None, iter=None):
        if epoch == config.start_epoch - 1 and iter < 3:
            print(desc + " ", datetime.datetime.now())

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    model.train()
    loss_fn = CrossEntropyLoss()
    tokenize_fn = (
        all_type_separate if config.embedding_type == "separate" else all_3type_complete
    )
    tokenize_res = tokenize_fn(train_dataset, "train", config)
    i = 0
    train_losses = []
    for i in range(config.start_epoch - 1, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )
        epoch_train_losses = []

        if config.epoch_switch:
            mode = config.switch_mode
            strategy = []
            if mode == 'pn':
                strategy = ["positive", "negtive"]
            elif mode == 'pnb':
                strategy = ["positive", "negative", "balance"]
            else:
                strategy = ["balance"]
            train_dataset.sample_strategy = strategy[i % len(strategy)]

        for bid, batch in enumerate(train_dataloader):
            if config.step_switch:
                mode = config.switch_mode
                strategy = []
                if mode == 'pn':
                    strategy = ["positive", "negtive"]
                elif mode == 'pnb':
                    strategy = ["positive", "negative", "balance"]
                else:
                    strategy = ["balance"]
                train_dataset.sample_strategy = strategy[i % len(strategy)]
            printTimeStr(f"iter {bid}", i, bid)
            (
                batch_img,
                batch_attr_target,
                batch_obj_target,
                batch_pair_target,
                attr_img,
                obj_img,
            ) = (
                batch[0].to(config.device),
                batch[1].to(config.device),
                batch[2].to(config.device),
                batch[4].to(config.device),
                batch[5].to(config.device),
                batch[6].to(config.device),
            )
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(batch_img, tokenize_res, attr_img, obj_img)
                loss = model.automatic_weighted_loss(
                    loss_fn(logits[0], batch_pair_target),
                    loss_fn(logits[1], batch_attr_target),
                    loss_fn(logits[2], batch_obj_target)
                )
                loss.requires_grad_(True)
                scaler.scale(loss).backward()
                if (bid + 1) % config.accu == 0:
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()
        progress_bar.close()
        progress_bar.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))
        epoch_evaluate(
            model, config, i + 1, epoch_train_losses
        )
        torch.save(
            model.state_dict(),
            os.path.join(config.save_path, config.experiment_name, f"epoch_{i + 1}_model.pt")
        )
    return model, optimizer


if __name__ == "__main__":
    config = parser.parse_args()
    load_args(config.config, config)
    config.save_path = path2optional_bool(config.save_path)
    config.load_model = path2optional_bool(config.load_model)
    config.use_soft_embeddings = path2optional_bool(config.use_soft_embeddings)
    config.train_soft_embeddings = str2bool(config.train_soft_embeddings)
    config.save_model = str2bool(config.save_model)
    config.multi_prompt_type = str2bool(config.multi_prompt_type)
    config.graph_init = path2optional_bool(config.graph_init)
    config.train_token_embeddings = str2bool(config.train_token_embeddings)
    config.vision_share = str2bool(config.vision_share)
    config.epoch_switch = str2bool(config.epoch_switch)
    config.step_switch = str2bool(config.step_switch)
    config.has_l_adapter = str2bool(config.step_switch)
    config.has_v_adapter = str2bool(config.step_switch)
    config.l_adapter_context = str2bool(config.step_switch)
    config.v_adapter_context = str2bool(config.step_switch)
    save_experiment_path = os.path.join(config.save_path, config.experiment_name)

    if not os.path.exists(save_experiment_path):
        os.makedirs(save_experiment_path)

        with open(
                os.path.join(config.save_path, config.experiment_name, "config.pkl"), "wb"
        ) as fp:
            pickle.dump(config, fp)
    set_seed(config.seed)
    print("training details")
    pprint.pprint(config)
    dataset_path = return_dataset_paths(config)[config.dataset]
    train_dataset = CompositionDataset(
        dataset_path, phase="train", split="compositional-split-natural", config=config
    )
    scaler = torch.cuda.amp.GradScaler()
    model, transform = load_clip(
        config.clip_model,
        config,
        train_dataset,
        jit=False,
        context_length=config.context_length,
        device=config.device,
    )
    if config.load_model:
        model.load_state_dict(torch.load(config.load_model))
    else:
        if config.use_soft_embeddings:
            model_parameters = torch.load(
                config.use_soft_embeddings, map_location=config.device
            )
            if "text_encoder.positional_embedding" in model_parameters:
                soft_embs = torch.load(
                    config.use_soft_embeddings, map_location=config.device
                )["text_encoder.soft_embeddings"]
            else:
                soft_embs = torch.load(
                    config.use_soft_embeddings, map_location=config.device
                )["soft_embeddings"]
            model.text_encoder.set_soft_embeddings(soft_embs)
            del model_parameters
            del soft_embs

        else:
            model.text_encoder.init_clip_embeddings()

    graph_parameters = []
    trainable_parameters = []
    token_embedding_parameters = []
    automatic_weighted_loss_parameters = []
    names = []
    keywords = ["norm", "projection", "logit_scale", "visual_attr_adapter", "visual_obj_adapter", "main_attr_weight",
                "counterpart_attr_weight", "main_obj_weight", "attr_proj", "obj_proj", "visual_encoder.model.proj",
                "fusion_mlp", "attr_weight", "obj_weight", "pair_weight"]
    if config.all_fine_tuning:
        keywords.append("visuai_encoder.model")
    for name, param in model.named_parameters():
        param.requires_grad = False
        if config.train_soft_embeddings and "soft" in name:
            param.requires_grad = True
            names.append(name)
            continue
        if "graph" in name:
            param.requires_grad = True
            graph_parameters.append(param)
            names.append(name)
            continue
        if any(keyword in name for keyword in keywords):
            param.requires_grad = True
            trainable_parameters.append(param)
            names.append(name)
            continue
        if "token" in name and config.train_token_embeddings and config.embedding_type == 'complete' and config.has_l_adapter:
            param.requires_grad = True
            token_embedding_parameters.append(param)
            names.append(name)
        if "automatic" in name:
            param.requires_grad = True
            automatic_weighted_loss_parameters.append(param)
    parameter_groups = []
    train_parameters = model.parameters()
    if config.train_soft_embeddings:
        parameter_groups.append(
            {
                "params": [model.text_encoder.soft_embeddings],
                "lr": config.lr,
                "weight_decay": 5e-5,
            }
        )
    if config.l_adater_layers != 0:
        parameter_groups.append(
            {"params": graph_parameters, "lr": config.lr, "weight_decay": 5e-5}
        )
    parameter_groups.append(
        {"params": trainable_parameters, "lr": config.lr, "weight_decay": 5e-5}
    )
    if config.train_token_embeddings and config.embedding_type == "complete":
        parameter_groups.append(
            {"params": token_embedding_parameters, "lr": config.lr, "weight_decay": 5e-5}
        )
    parameter_groups.append(
        {"params": automatic_weighted_loss_parameters, "lr": config.lr, "weight_decay": 5e-5}
    )
    optimizer = torch.optim.Adam(parameter_groups)

    print("finished")
