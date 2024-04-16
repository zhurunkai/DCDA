import copy
import json
import os
import numpy as np
import torch
from scipy.stats import hmean
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import return_dataset_paths
from utils.core import path2optional_bool, str2bool, load_args
from utils.text_encode import all_type_separate, all_3type_complete
from models.load_model import load_clip
from flags import parser

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class Evaluator:

    def __init__(self, dataset):
        self.dataset = dataset
        pairs = [
            (dataset.attr2idx[attr], dataset.obj2idx[obj])
            for attr, obj in dataset.pairs
        ]
        self.train_pairs = [
            (dataset.attr2idx[attr], dataset.obj2idx[obj])
            for attr, obj in dataset.train_pairs
        ]
        self.pairs = torch.LongTensor(pairs)

        # Mask over pairs that occur in closed world
        # Select set based on phase
        if dataset.phase == "train":
            print("Evaluating with train pairs")
            test_pair_set = set(dataset.train_pairs)
            test_pair_gt = set(dataset.train_pairs)
        elif dataset.phase == "val":
            print("Evaluating with validation pairs")
            test_pair_set = set(dataset.val_pairs + dataset.train_pairs)
            test_pair_gt = set(dataset.val_pairs)
        else:
            print("Evaluating with test pairs")
            test_pair_set = set(dataset.test_pairs + dataset.train_pairs)
            test_pair_gt = set(dataset.test_pairs)

        self.test_pair_dict = [
            (dataset.attr2idx[attr], dataset.obj2idx[obj]) for attr, obj in test_pair_gt
        ]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)

        # dict values are pair val, score, total
        for attr, obj in test_pair_gt:
            pair_val = dataset.pair2idx[(attr, obj)]
            key = (dataset.attr2idx[attr], dataset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        # open world
        if dataset.open_world:
            masks = [1 for _ in dataset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dataset.pairs]

        self.closed_mask = torch.BoolTensor(masks)

        # Mask of seen concepts
        seen_pair_set = set(dataset.train_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in dataset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        # Object specific mask over which pairs occur in the object oracle setting
        oracle_obj_mask = []
        for _obj in dataset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dataset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # Decide if the model under evaluation is a manifold model or not
        self.score_model = self.score_manifold_model

    # Generate mask for each settings, mask scores, and get prediction labels
    def generate_predictions(
            self, scores, obj_truth, bias=0.0, topk=1
    ):

        def get_pred_from_scores(_scores, topk):
            _, pair_pred = _scores.topk(
                topk, dim=1
            )
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(
                -1, topk
            ), self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(
            scores.shape[0], 1
        )
        scores[~mask] += bias

        results.update({"open": get_pred_from_scores(scores, topk)})
        results.update({"unbiased_open": get_pred_from_scores(orig_scores, topk)})

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({"closed": get_pred_from_scores(closed_scores, topk)})
        results.update(
            {"unbiased_closed": get_pred_from_scores(closed_orig_scores, topk)}
        )

        return results

    def score_clf_model(self, scores, obj_truth, topk=1):
        attr_pred, obj_pred = scores
        attr_pred, obj_pred, obj_truth = (
            attr_pred.to("cpu"),
            obj_pred.to("cpu"),
            obj_truth.to("cpu"),
        )
        attr_subset = attr_pred.index_select(1, self.pairs[:, 0])
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = attr_subset * obj_subset

        results = self.generate_predictions(scores, obj_truth)
        results["biased_scores"] = scores

        return results

    def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=1):
        scores = {k: v.to("cpu") for k, v in scores.items()}
        obj_truth = obj_truth

        # Gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr, obj)] for attr, obj in self.dataset.pairs], 1
        )
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results["scores"] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1):
        results = {}
        # Repeat mask along pairs dimension
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10

        # sort returns indices of k largest values
        _, pair_pred = closed_scores.topk(topk, dim=1)
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), self.pairs[
                                                                              pair_pred
                                                                          ][:, 1].view(-1, topk)

        results.update({"closed": (attr_pred, obj_pred)})
        return results

    def evaluate_predictions(
            self, predictions, attr_truth, obj_truth, pair_truth, allpred, topk=1
    ):
        attr_truth, obj_truth, pair_truth = (
            attr_truth.to("cpu"),
            obj_truth.to("cpu"),
            pair_truth.to("cpu"),
        )

        pairs = list(zip(list(attr_truth.numpy()), list(obj_truth.numpy())))

        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)

        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(unseen_ind)

        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            attr_match = attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk]
            obj_match = obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk]

            # Match of object pair
            match = (
                (attr_match * obj_match).any(1).float()
            )
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]

            seen_score, unseen_score = torch.ones(512, 5), torch.ones(512, 5)  # 待定 有啥用？

            return (
                attr_match,
                obj_match,
                match,
                seen_match,
                unseen_match,
                torch.Tensor(seen_score + unseen_score),
                torch.Tensor(seen_score),
                torch.Tensor(unseen_score),
            )

        def _add_to_dict(_scores, type_name, stats):
            base = [
                "_attr_match",
                "_obj_match",
                "_match",
                "_seen_match",
                "_unseen_match",
                "_ca",
                "_seen_ca",
                "_unseen_ca",
            ]
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        stats = dict()

        # Closed world
        closed_scores = _process(predictions["closed"])
        unbiased_closed = _process(predictions["unbiased_closed"])
        _add_to_dict(closed_scores, "closed", stats)
        _add_to_dict(unbiased_closed, "closed_ub", stats)

        # Calculating AUC
        scores = predictions["scores"]
        # getting score for each ground truth class
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][unseen_ind]

        # Getting top predicted score for these unseen classes
        max_seen_scores = predictions["scores"][unseen_ind][:, self.seen_mask].topk(
            topk, dim=1
        )[0][:, topk - 1]

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - correct_scores

        # Getting matched classes at max bias for diff
        unseen_matches = stats["closed_unseen_match"].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        # sorting these diffs
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]

        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats["closed_seen_match"].mean())
        unseen_match_max = float(stats["closed_unseen_match"].mean())
        seen_accuracy, unseen_accuracy = [], []

        base_scores = {k: v.to("cpu") for k, v in allpred.items()}
        obj_truth = obj_truth.to("cpu")

        # Gather scores for all relevant (a,o) pairs
        base_scores = torch.stack(
            [allpred[(attr, obj)] for attr, obj in self.dataset.pairs], 1
        )

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(scores, obj_truth, bias=bias, topk=topk)
            results = results["closed"]  # we only need biased
            results = _process(results)
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)
        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(
            unseen_accuracy
        )
        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        try:
            harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis=0)
        except BaseException:
            harmonic_mean = 0

        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats["biasterm"] = float(bias_term)
        stats["best_unseen"] = np.max(unseen_accuracy)
        stats["best_seen"] = np.max(seen_accuracy)
        stats["AUC"] = area
        stats["hm_unseen"] = unseen_accuracy[idx]
        stats["hm_seen"] = seen_accuracy[idx]
        stats["best_hm"] = max_hm
        return stats


def compute_representations(model, test_dataset, config):
    tokenize_fn = all_type_separate if config.embedding_type == "separate" else all_3type_complete
    tokenize_res = tokenize_fn(test_dataset, "all", config)
    rep = model.text_encoder(tokenize_res)

    attr_features = rep[: len(test_dataset.attrs), :]
    obj_features = rep[
                   len(test_dataset.attrs): len(test_dataset.attrs)
                                            + len(test_dataset.objs),
                   :,
                   ]
    rep = rep[
          len(test_dataset.attrs) + len(test_dataset.objs):, :
          ]

    attr_features = attr_features / attr_features.norm(dim=1, keepdim=True)
    obj_features = obj_features / obj_features.norm(dim=1, keepdim=True)
    rep = rep / rep.norm(dim=1, keepdim=True)
    return rep, attr_features, obj_features


def predict_logits(model, text_rep, dataset, config):
    device = config.device
    model.eval()
    all_attr_gt, all_obj_gt, all_pair_gt = (
        [],
        [],
        [],
    )
    pair_rep, attr_rep, obj_rep = text_rep
    dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False)
    all_logits = torch.Tensor()

    with torch.no_grad():
        for idx, data in tqdm(
                enumerate(dataloader), total=len(dataloader), desc="Testing"
        ):
            batch_img = data[0].to(device)
            batch_attr = batch_img.clone()
            batch_obj = batch_img.clone()
            img_self, img_attr, img_obj, img_same_attr, img_same_obj = model.visual_encoder(
                [batch_img, batch_attr, batch_obj])
            normalized_img = img_self / img_self.norm(dim=-1, keepdim=True)
            normalized_img_attr = img_attr / img_attr.norm(dim=-1, keepdim=True)
            normalized_img_obj = img_obj / img_obj.norm(dim=-1, keepdim=True)

            pairwise_attr_rep = attr_rep[dataset.pairs2attr_idx]
            pairwise_obj_rep = obj_rep[dataset.pairs2obj_idx]

            logits = model.pair_weight * model.logit_scale.exp() * normalized_img @ pair_rep.t() + model.attr_weight * model.attr_logit_scale.exp() * normalized_img_attr @ pairwise_attr_rep.t() + \
                     model.obj_weight * model.obj_logit_scale.exp() * normalized_img_obj @ pairwise_obj_rep.t()

            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
            logits = logits.cpu()
            all_logits = torch.cat([all_logits, logits], dim=0)

            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )
    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt


def threshold_with_feasibility(logits, seen_mask, threshold=None, feasiblity=None):
    score = copy.deepcopy(logits)
    # Note: Pairs are already aligned here
    mask = (feasiblity >= threshold).float()
    # score = score*mask + (1.-mask)*(-1.)
    score = score * (mask + seen_mask)

    return score


def test(
        test_dataset, evaluator, all_logits, all_attr_gt, all_obj_gt, all_pair_gt, config
):
    predictions = {
        pair_name: all_logits[:, i] for i, pair_name in enumerate(test_dataset.pairs)
    }
    all_pred = [predictions]

    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k] for i in range(len(all_pred))]
        ).float()

    results = evaluator.score_model(
        all_pred_dict, all_obj_gt.to(config.device), bias=config.bias, topk=config.topk
    )

    attr_acc = float(
        torch.mean((results["unbiased_closed"][0].squeeze(-1) == all_attr_gt).float())
    )
    obj_acc = float(
        torch.mean((results["unbiased_closed"][1].squeeze(-1) == all_obj_gt).float())
    )

    stats = evaluator.evaluate_predictions(
        results,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        all_pred_dict,
        topk=config.topk,
    )

    stats["attr_acc"] = attr_acc
    stats["obj_acc"] = obj_acc

    return stats


def main(config, model=None, is_train=False, epoch=None):
    dataset_path = return_dataset_paths(config)[config.dataset]
    print("loading validation dataset")
    val_dataset = CompositionDataset(
        dataset_path,
        phase="val",
        split="compositional-split-natural",
        open_world=config.open_world,
    )
    print("loading test dataset")
    test_dataset = CompositionDataset(
        dataset_path,
        phase="test",
        split="compositional-split-natural",
        open_world=config.open_world,
    )
    if not model:
        # totally evaluate
        model, transform = load_clip(
            config.clip_model,
            config,
            val_dataset,
            jit=False,
            context_length=config.context_length,
        )
        if config.eval_load_model:
            model.load_state_dict(torch.load(config.eval_load_model))
        else:
            if config.eval_use_soft_embeddings:
                model_parameters = torch.load(config.eval_use_soft_embeddings, map_location=config.device)
                soft_embs = torch.load(config.eval_use_soft_embeddings, map_location=config.device)[
                    "text_encoder.soft_embeddings"]
                model.text_encoder.set_soft_embeddings(soft_embs)
            else:
                model.text_encoder.init_clip_embeddings()
            model.to(config.device)

    val_text_rep = compute_representations(model, val_dataset, config)
    test_text_rep = compute_representations(model, test_dataset, config)

    print("evaluating on the validation set")
    evaluator = Evaluator(val_dataset)
    with torch.no_grad():
        all_logits_list, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits(
            model, val_text_rep, val_dataset, config
        )
        results = test(
            val_dataset,
            evaluator,
            all_logits_list,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config,
        )
    val_stats = copy.deepcopy(results)

    print("evaluating on the test set")
    with torch.no_grad():
        evaluator = Evaluator(test_dataset)
        all_logits_list, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits(
            model, test_text_rep, test_dataset, config
        )
        results = test(
            test_dataset,
            evaluator,
            all_logits_list,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config,
        )
    test_stats = copy.deepcopy(results)

    results = {
        "val": val_stats,
        "test": test_stats,
    }

    if epoch is not None:
        result_filename = f"{epoch}_closed.json"
    else:
        result_filename = f"evaluate_results.json"
    result_path = os.path.join(
        config.save_path, config.experiment_name, result_filename
    )

    if is_train:
        return results, result_path
    else:
        with open(result_path, "w+") as fp:
            json.dump(results, fp)

    print("done!")


if __name__ == "__main__":
    config = parser.parse_args()
    load_args(config.config, config)
    config.save_path = path2optional_bool(config.save_path)
    config.load_model = path2optional_bool(config.load_model)
    config.use_soft_embeddings = path2optional_bool(config.use_soft_embeddings)
    config.train_soft_embeddings = str2bool(config.train_soft_embeddings)
    config.save_model = str2bool(config.save_model)
    config.graph_init = path2optional_bool(config.graph_init)
    config.v_adapter_context = str2bool(config.v_adapter_context)
    config.l_adapter_context = str2bool(config.l_adapter_context)
    config.has_v_adapter = str2bool(config.has_v_adapter)
    config.has_l_adapter = str2bool(config.has_l_adapter)

    config.eval_load_model = path2optional_bool(config.eval_load_model)
    config.eval_use_soft_embeddings = path2optional_bool(config.eval_use_soft_embeddings)
    with torch.no_grad():
        main(config)
