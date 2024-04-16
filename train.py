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
from utils.text_encode import all_type_separate, all_3type_complete
from utils.core import set_seed, path2optional_bool, str2bool, load_args
from utils.config_model import get_model_and_optimizer
from evaluate import main as evaluate_test
from flags import parser

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
if_save = False


def if_save_model(evaluate_base, evaluate_results, epoch=None):
    global if_save
    if_save = False
    if evaluate_results["val"]["AUC"] > evaluate_base["auc"]:
        evaluate_base["auc"] = evaluate_results["val"]["AUC"]
        if epoch is not None:
            evaluate_base["epoch"] = epoch
        if_save = True
    return if_save


def epoch_evaluate(
        model, config, epoch, epoch_train_losses, evaluate_base
):
    model.eval()
    with torch.no_grad():
        model.text_encoder.graph_adapter.switch_adj("all")
        results, result_path = evaluate_test(config, model, is_train=True, epoch=epoch)
        model.text_encoder.graph_adapter.switch_adj("train")
        if_save_model(evaluate_base, results, epoch)
        results["train_loss"] = np.mean(epoch_train_losses)
        results["evaluate_base"] = evaluate_base
        with open(result_path, "w+") as fp:
            json.dump(results, fp)
    model.train()
    return results


def train_model(
        model, optimizer, train_dataset, config, scaler
):
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    model.train()
    loss_fn = CrossEntropyLoss()
    tokenize_fn = (
        all_type_separate if config.embedding_type == "separate" else all_3type_complete
    )
    tokenize_res = tokenize_fn(train_dataset, "train", config)
    train_losses = []

    evaluate_base = {
        "best_seen": 0,
        "best_unseen": 0,
        "best_hm": 0,
        "auc": 0,
        "epoch": 0
    }

    for i in range(config.start_epoch - 1, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )
        epoch_train_losses = []

        for bid, batch in enumerate(train_dataloader):
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
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()
        progress_bar.close()
        progress_bar.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))

        evaluate_results = epoch_evaluate(
            model, config, i + 1, epoch_train_losses, evaluate_base
        )
        if (
                config.save_model
                and (i + 1) % config.save_every_n == 0
                and if_save_model(evaluate_base, evaluate_results)
        ):
            torch.save(
                model.state_dict(),
                os.path.join(str(config.save_path), config.experiment_name, f"epoch_{i + 1}_model.pt")
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
    config.v_adapter_context = str2bool(config.v_adapter_context)
    config.l_adapter_context = str2bool(config.l_adapter_context)
    config.has_v_adapter = str2bool(config.has_v_adapter)
    config.has_l_adapter = str2bool(config.has_l_adapter)
    config.graph_init = path2optional_bool(config.graph_init)
    config.train_token_embeddings = str2bool(config.train_token_embeddings)

    save_experiment_path = os.path.join(config.save_path, config.experiment_name)
    if not os.path.exists(save_experiment_path):
        os.makedirs(save_experiment_path)
        with open(os.path.join(config.save_path, config.experiment_name, "config.pkl"), "wb") as fp:
            pickle.dump(config, fp)
    # set the seed value
    set_seed(config.seed)
    pprint.pprint(config)
    dataset_path = return_dataset_paths(config)[config.dataset]
    train_dataset = CompositionDataset(dataset_path, phase="train", split="compositional-split-natural", config=config)
    scaler = torch.cuda.amp.GradScaler()
    model, optimizer = get_model_and_optimizer(config, train_dataset)
    if config.phase == "train":
        model, optimizer = train_model(model, optimizer, train_dataset, config, scaler)
        if config.save_model:
            torch.save(model.state_dict(),
                       os.path.join(str(config.save_path), config.experiment_name, "final_model.pt"))
