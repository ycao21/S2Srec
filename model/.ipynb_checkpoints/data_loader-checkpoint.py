from __future__ import print_function
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import torch

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def pad_set(idx_set, N=20):
    if len(idx_set) < N:
        padding = np.zeros(N - len(idx_set), dtype=np.int32)
        padded_set = np.append(idx_set, padding)
        return padded_set
    else:
        return idx_set


class ingredsetDataset(Dataset):
    def __init__(self, input_data, recipe_ingred_code):
        self.x = input_data
        self.recipe_ingred_map = recipe_ingred_code

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        pos_ing_concat_set = self.x[idx][0] + self.x[idx][1]
        pos_ln = len(pos_ing_concat_set)
        pos_ing_concat_set = pad_set(pos_ing_concat_set)

        neg_ing_concat_set = self.x[idx][0] + self.x[idx][2]
        neg_ln = len(neg_ing_concat_set)
        neg_ing_concat_set = pad_set(neg_ing_concat_set)

        recipe_pos = self.recipe_ingred_map.get(self.x[idx][4][0])
        recipe_pos_ln = len(recipe_pos)
        recipe_pos = pad_set(recipe_pos)

        recipe_neg = self.recipe_ingred_map.get(self.x[idx][5][0])
        recipe_neg_ln = len(recipe_neg)
        recipe_neg = pad_set(recipe_neg)

        return [torch.LongTensor(pos_ing_concat_set), pos_ln, torch.ones(1)], \
               [torch.LongTensor(neg_ing_concat_set), neg_ln, torch.zeros(1)], \
               [torch.LongTensor(recipe_pos), recipe_pos_ln], [torch.LongTensor(recipe_neg), recipe_neg_ln]

class clfDataset(Dataset):
    def __init__(self, input_data, recipe_ingred_code):
        self.x = input_data
        self.recipe_ingred_map = recipe_ingred_code

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        anchor_set = self.x[idx][0]
        anchor_set = pad_set(anchor_set)

        ingred_set = self.x[idx][0] + self.x[idx][1]
        set_ln = len(ingred_set)
        ingred_set = pad_set(ingred_set)

        target = self.x[idx][2]
        if len(self.x[idx]) == 4:
            recipe = self.x[idx][3]
        else:
            recipe = -1
        return torch.LongTensor(anchor_set), [torch.LongTensor(ingred_set), set_ln], torch.tensor(1), recipe


class recipeEmbDataset(Dataset):
    def __init__(self, recipe_index, recipe_ingred_code):
        self.y = recipe_index
        self.recipe_ingred_map = recipe_ingred_code

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        recipe = self.recipe_ingred_map.get(self.y[idx])
        recipe_ln = len(recipe)
        recipe = pad_set(recipe)

        return [self.y[idx], torch.LongTensor(recipe), recipe_ln]


class cartsetDataset(Dataset):
    def __init__(self, cart_index, recipe_index, cart_item_code, cart_item_idx_to_emb_idx, recipe_ingred_code):
        self.x = cart_index
        self.y = recipe_index
        self.cart_item_map = cart_item_code
        self.emb_idx_map = cart_item_idx_to_emb_idx
        self.recipe_ingred_map = recipe_ingred_code

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        cart_set_raw = self.cart_item_map.get(self.x[idx], 0)
        cart_set = [self.emb_idx_map.get(i, 0) for i in cart_set_raw]
        cart_set_ln = len(cart_set)
        cart_set = pad_set(cart_set)

        matched_recipe = self.recipe_ingred_map.get(self.y[idx])
        matched_recipe_ln = len(matched_recipe)
        matched_recipe = pad_set(matched_recipe)

        return [self.x[idx], torch.LongTensor(cart_set), cart_set_ln], \
               [torch.LongTensor(matched_recipe), matched_recipe_ln]