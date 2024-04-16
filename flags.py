import argparse

DATA_FOLDER = "./DATA_ROOT"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--experiment_name", help="name of the experiment", type=str, default="default_name"
)
parser.add_argument(
    "--dataset", help="name of the dataset", type=str, default="mit-states"
)
parser.add_argument(
    "--clip_model", help="clip model type", type=str, default="ViT-L/14"
)
parser.add_argument(
    "--context_length",
    help="sets the context length of the clip model",
    default=16,
    type=int,
)
parser.add_argument("--seed", help="seed value", default=3407, type=int)
parser.add_argument("--phase", help="phase", default="train", type=str)
parser.add_argument("--device", help="device", default="cuda", type=str)
parser.add_argument(
    "--load_model", help="directory for loading models", default="None", type=str
)
parser.add_argument(
    "--use_soft_embeddings",
    help="directory for soft embeddings",
    type=str,
    default="None",
)
parser.add_argument(
    "--train_soft_embeddings",
    type=str,
    default="False",
)
parser.add_argument(
    "--train_token_embeddings",
    type=str,
    default="True",
)
parser.add_argument(
    "--embedding_type",
    help="complete or separate",
    default="complete",
    type=str,
)
parser.add_argument("--lr", help="learning rate", type=float, default=5e-5)
parser.add_argument("--soft_emb_lr", help="learning rate", type=float, default=5e-6)
parser.add_argument("--wd", help="weight decay", type=float, default=5e-5)
parser.add_argument("--epochs", help="number of epochs", default=40, type=int)
parser.add_argument(
    "--train_batch_size", help="train batch size", default=256, type=int
)
parser.add_argument(
    "--save_path", help="save path", type=str, default="./data/mit-states"
)
parser.add_argument(
    "--save_every_n", help="saves the model every n epochs", default=1, type=int
)
parser.add_argument(
    "--save_model",
    help="indicate if you want to save the model state dict()",
    default="True",
    type=str,
)
parser.add_argument("--start_epoch", default=1, type=int)
parser.add_argument(
    "--dropout",
    help="add dropout",
    type=float,
    default=0.3,
)
parser.add_argument(
    "--graph_init",
    help="get dataset adj (all or train); else None",
    default="None",
    type=str,
)
parser.add_argument("--l_adapter_layers", default=3, type=int)
parser.add_argument("--v_adapter_layers", default=3, type=int)
parser.add_argument("--v_adapter_context", default="True", type=str)
parser.add_argument("--l_adapter_context", default="True", type=str)
parser.add_argument("--has_v_adapter", default="True", type=str)
parser.add_argument("--has_l_adapter", default="True", type=str)
parser.add_argument("--l_adapter_location", default="True", type=str)
parser.add_argument("--v_adapter_location", default="out", type=str)
parser.add_argument("--l_adapter_location", default="in", type=str)
parser.add_argument(
    "--hidden_layers", help="hidden_layers", default="d4096,d", type=str
)

parser.add_argument("--eval_batch_size", default=64, type=int)
parser.add_argument(
    "--bias",
    help="eval bias",
    type=float,
    default=1e3,
)
parser.add_argument(
    "--topk",
    help="eval topk",
    type=int,
    default=1,
)
parser.add_argument(
    "--eval_load_model", help="location for all model; else None", default="None", type=str
)
parser.add_argument(
    "--eval_use_soft_embeddings",
    help="directory for soft embeddings",
    type=str,
    default="None"
)
parser.add_argument("--threshold", type=float, help="optional threshold")
parser.add_argument(
    "--threshold_trials", type=int, default=50, help="how many threshold values to try"
)
parser.add_argument(
    "--open_world",
    help="evaluate on open world setup",
    action="store_true",
)
