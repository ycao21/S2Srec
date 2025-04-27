from __future__ import print_function
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import torch
import random

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def pad_set(idx_set, pad_value, N=45):
    if len(idx_set) < N:
        padding = np.ones(N - len(idx_set), dtype=np.int32)*pad_value
        padded_set = np.append(idx_set, padding)
        return padded_set
    else:
        return idx_set


class ingredsetDataset(Dataset):
    def __init__(self, input_data):
        self.x = input_data

    def __len__(self):
        return len(self.x)
        # return len(self.sets)

    def __getitem__(self, idx):
        recipe_id = self.x[idx][0]
        ingredient_set = self.x[idx][1]
        return (recipe_id, ingredient_set)

import random
def collate_mask_n(batch, pad_id=0, N=45):
    """
    batch: list of ingredient‐ID lists
    pad_id: the ID to use for padding
    N: fixed max set size after padding
    Returns:
      inputs: LongTensor [B, N] with one element removed and the rest padded
      targets: LongTensor [B] with the masked‐out ingredient ID
      lengths: LongTensor [B] with the true (unpadded) length of each input
    """
    recipe, inputs, targets, lengths = [], [], [], []
    set_binary, set_binary_lengths, labels = [], [], []
    for recipe_s in batch:
        # print(S)
        # 1) choose one ingredient at random to be the "masked" target
        recipe_id = recipe_s[0]
        S = recipe_s[1].copy()
        # S.append(ingredient_dict['<stop>'])
        dropped = []
        for i in range(3):
            # for mlm
            tfidf_rank = [i[0] for i in sorted(recipe_tfidf[recipe_id].items(),
                                               key=lambda item: item[1],
                                               reverse=True)
                               if i[0] not in dropped]
            m = tfidf_rank[0]
            dropped.append(m)
            # 2) remove *one instance* of that ingredient
            S_minus = S.copy()
            S_minus.remove(m)
            # 3) shuffle the remaining set
            random.shuffle(S_minus)
            # 4) record its true length
            lengths.append(len(S_minus))
            # 5) pad/truncate to N
            padded = pad_set(S_minus, pad_id, N=N)
            inputs.append(padded)
            targets.append(m)
            recipe.append(recipe_id)

            # for clf
            if i == 0:
                n = 0
            else:
                n = random.randint(1, min(len(S)-2,3))
            m = random.sample(S, n)
            S_minus = S.copy()
            for i in m:
                S_minus.remove(i)
            padded = pad_set(S_minus, pad_id, N=N)
            set_binary.append(padded)
            set_binary_lengths.append(len(S_minus))
            if n == 0:
                labels.append(1)
            else:
                labels.append(0)


    # print(inputs, targets)
    return (torch.tensor(recipe),
        (torch.LongTensor(inputs),    # [B, N]
        torch.LongTensor(lengths),    # [B]
        torch.LongTensor(targets)),  # [B]
        (torch.LongTensor(set_binary),    # [B, N]
        torch.LongTensor(set_binary_lengths),    # [B]
        torch.FloatTensor(labels))
    )


def collate_mask_n_cpu(batch, leave_n_out=False):
    output = []
    for recipe_s in batch:
        # print(S)
        # 1) choose one ingredient at random to be the "masked" target
        S = recipe_s[1].copy()
        # S.append(ingredient_dict['<stop>'])
        for i in range(1):
            if leave_n_out:
                n = random.randint(1, min(3, len(S)-2))
            else:
                n = 1
            m = random.sample(S, n)
            # n = random.randint(1, min(3, len(S)-2))
            # 2) remove *one instance* of that ingredient
            S_minus = S.copy()
            for i in m:
                S_minus.remove(i)
            # 3) shuffle the remaining set
            random.shuffle(S_minus)
            # 4) record its true length
            lengths = len(S_minus)
            output.append((recipe_s[0], (S_minus, lengths, m)))

    return output