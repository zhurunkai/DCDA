import torch
from models.load_model import load_clip


def get_model_and_optimizer(config, train_dataset):
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
            soft_embs = torch.load(config.use_soft_embeddings, map_location=config.device)[
                "text_encoder.soft_embeddings"]
            model.text_encoder.set_soft_embeddings(soft_embs)
        else:
            model.text_encoder.init_clip_embeddings()
    train_param_names = ["graph", "norm", "projection", "logit_scale", "visual_attr_adapter", "visual_obj_adapter",
                         "main_attr_weight",
                         "counterpart", "main_obj_weight", "fusion_weight", "attr_proj", "obj_proj",
                         "visual_encoder.model.proj",
                         "attr_weight", "obj_weight", "pair_weight"]
    graph_parameters = []
    trainable_parameters = []
    token_embedding_parameters = []
    for name, param in model.named_parameters():
        param.requires_grad = False
        if config.train_soft_embeddings and "soft" in name:
            param.requires_grad = True
            continue
        if "token" in name and config.train_token_embeddings and config.embedding_type == 'complete' and config.has_l_adapter:
            param.requires_grad = True
            token_embedding_parameters.append(param)
            continue
        train_flag = True
        for x in train_param_names:
            if train_flag or x in name:
                param.requires_grad = True
                trainable_parameters.append(param)
                train_flag = False
    parameter_groups = []
    if config.train_soft_embeddings:
        parameter_groups.append(
            {
                "params": [model.text_encoder.soft_embeddings],
                "lr": config.soft_emb_lr,
                "weight_decay": config.wd,
            }
        )
    if config.adapter_layers != 0:
        parameter_groups.append(
            {"params": graph_parameters, "lr": config.lr, "weight_decay": config.wd}
        )
    parameter_groups.append(
        {"params": trainable_parameters, "lr": config.lr, "weight_decay": config.wd}
    )
    if config.train_token_embeddings and config.embedding_type == "complete" and config.has_l_adapter:
        parameter_groups.append(
            {"params": token_embedding_parameters, "lr": config.lr, "weight_decay": config.wd}
        )
    optimizer = torch.optim.Adam(parameter_groups)
    return model, optimizer
