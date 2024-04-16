from datasets.composition_dataset import CompositionDataset
import torch
from models import clip


def all_type_separate(dataset: CompositionDataset, type="train", config=None):
    prompt_idx_list = []
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    pairs = dataset.train_pairs if type == "train" else dataset.pairs
    if config.multi_prompt_type:
        for attr in dataset.attrs:
            prompt_idx_list.append([attr2idx[attr], 0])
        for obj in dataset.objs:
            prompt_idx_list.append([obj2idx[obj], 0])
    for pair in pairs:
        prompt_idx_list.append([attr2idx[pair[0]], obj2idx[pair[1]]])
    device = config.device if config else "cpu"
    prompt_idx_list = torch.tensor(prompt_idx_list).to(device)
    return prompt_idx_list


def tokenize(list, context_length):
    return clip.tokenize(list, context_length=context_length)


def clean_text(v):
    custom_map = {
        'Faux.Fur': 'fake fur',
        'Faux.Leather': 'fake leather',
        'Full.grain.leather': 'thick leather',
        'Hair.Calf': 'hairy leather',
        'Patent.Leather': 'shiny leather',
        'Boots.Ankle': 'ankle boots',
        'Boots.Knee.High': 'kneehigh boots',
        'Boots.Mid-Calf': 'midcalf boots',
        'Shoes.Boat.Shoes': 'boatshoes',
        'Shoes.Clogs.and.Mules': 'clogs shoes',
        'Shoes.Flats': 'flats shoes',
        'Shoes.Heels': 'heels',
        'Shoes.Loafers': 'loafers',
        'Shoes.Oxfords': 'oxford shoes',
        'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers',
        'traffic_light': 'traficlight',
        'trash_can': 'trashcan',
        'dry-erase_board': 'dry_erase_board',
        'black_and_white': 'black_white',
        'eiffel_tower': 'tower'
    }

    if v in custom_map:
        return custom_map[v]
    else:
        return v.lower()


def all_3type_complete(dataset, type="train", config=None):
    if type == 'train':
        pairs = [' '.join([clean_text(t) for t in c]) for c in dataset.train_pairs]
    else:
        pairs = [' '.join([clean_text(t) for t in c]) for c in dataset.pairs]

    prompt_list = ["a photo of a {} object".format(clean_text(c)) for c in dataset.attrs] + \
                  ["a photo of a {}".format(clean_text(c)) for c in dataset.objs] + \
                  [f"a photo of a {c}" for c in pairs]
    device = config.device if config else "cpu"
    tokenized_list = tokenize(prompt_list, config.context_length).to(device)
    return tokenized_list
