# !/usr/bin/env python
import re
import numpy as np
import string
import random
import pickle
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

from config import *


def words_preprocessing(text):
    text = text.lower()
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)
    word_tokens = nopunct.split(' ')
    lemmatizer = WordNetLemmatizer()
    word_lemma = [lemmatizer.lemmatize(w) for w in word_tokens]
    stop_words = set(stopwords.words('english')+['fresh','organic'])
    filtered_sentence = [w for w in word_lemma if not w in stop_words]
    return  ' '.join(filtered_sentence)


def get_tokenized(text):
    text = words_preprocessing(text)
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    segments_ids = [1] * len(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print(tokenized_text)
    return indexed_tokens, segments_ids


def get_BERTemb(model, tokens_tensor, segments_tensors):
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    token_vecs = encoded_layers[11][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding


def get_embeddings(text, model):
    indexed_tokens, segments_ids = get_tokenized(text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    emb_768 = get_BERTemb(model, tokens_tensor, segments_tensors)
    return emb_768.detach().numpy()

def generate_embedding(model, index_text_map_path, emb_dump_path):
    model.eval()

    with open(index_text_map_path, 'rb') as f:
        map = pickle.load(f)

    emb_dict = {}
    for k in tqdm(map.keys()):
        emb = get_embeddings(map[k], model)
        emb_dict[k] = emb

    with open(emb_dump_path, 'wb') as handle:
        pickle.dump(emb_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return emb_dict


def calculate_recipe_ingredient_dist(recipe_name_emb, ingred_emb, recipeidx2ingredidx, recipe_ingredient_dist_cos_path):
    recipe_item_dist = {}
    for i in tqdm(recipe_name_emb.keys()):
        dist_list = {}
        for j in recipeidx2ingredidx[i]:
            dist = np.dot(recipe_name_emb[i], ingred_emb[j]) / (np.linalg.norm(recipe_name_emb[i]) * np.linalg.norm(ingred_emb[j]))
            dist_list[j] = dist
        recipe_item_dist[i] = dist_list

    with open(recipe_ingredient_dist_cos_path, 'wb') as handle:
        pickle.dump(recipe_item_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return recipe_item_dist

def get_recipe_main_ingredient(common_ingredient, recipe_ingredient_dist_cos, ingred_idx2name_map, n_main, n_max_ingred, recipeidx2mainingredidx_map_path):
    recipe_main_ingredient = {}
    for i in tqdm(recipe_ingredient_dist_cos.keys()):
        result = {k: v for k, v in sorted(recipe_ingredient_dist_cos[i].items(), key=lambda item: item[1], reverse=True)
                  if ingred_idx2name_map[k] not in common_ingredient}
        m_i = list(result.keys())[:n_main]
        if len(m_i) >= n_main-1 and len(result.keys()) <= n_max_ingred:
            recipe_main_ingredient[i] = m_i

    with open(recipeidx2mainingredidx_map_path, 'wb') as handle:
        pickle.dump(recipe_main_ingredient, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

def prepare_training_set(recipe_main_ingredient, item_idx_list, ingred_idx_list, recipe_idx_list):

    incomplete_recipe_train = []
    clf_ingred_set_train = []
    incomplete_recipe_test = []
    clf_ingred_set_test = []

    n_train = round(len(recipe_main_ingredient.keys()) * 0.7)
    n = 0
    for r in tqdm(recipe_main_ingredient.keys()):
        for _ in range(3):
            random_recipe = random.sample(recipe_idx_list, 1)
            if random_recipe == r:
                random_recipe = random.sample(recipe_idx_list, 1)

            n_pick_ingred = random.choice([2, 3])
            picked_ingred = random.sample(recipe_main_ingredient[r], n_pick_ingred)

            unpicked_ingred = [i for i in recipe_main_ingredient[r] if i not in picked_ingred]

            n_pick_item = random.choice([2, 3, 4, 5, 6, 7, 8])
            random_item = random.sample(item_idx_list, n_pick_item)

            n_pick_ingred = random.choice([2, 3, 4, 5])
            random_ingred = random.sample(ingred_idx_list, n_pick_ingred)

            if n <= n_train:
                incomplete_recipe_train.append([picked_ingred, unpicked_ingred, random_ingred, random_item, [r, 1], [random_recipe[0], -1]])
                clf_ingred_set_train.append([picked_ingred, random_ingred, 0])
            else:
                incomplete_recipe_test.append([picked_ingred, unpicked_ingred, random_ingred, random_item, [r, 1], [random_recipe[0], -1]])
                clf_ingred_set_test.append([picked_ingred, random_ingred, 0])
        if n <= n_train:
            clf_ingred_set_train.append([picked_ingred, unpicked_ingred, 1, r])
        else:
            clf_ingred_set_test.append([picked_ingred, unpicked_ingred, 1, r])

        n += 1

    with open(triplet_train_set_path, 'wb') as handle:
        pickle.dump(incomplete_recipe_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(clf_train_set_path, 'wb') as handle:
        pickle.dump(clf_ingred_set_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(triplet_test_set_path, 'wb') as handle:
        pickle.dump(incomplete_recipe_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(clf_test_set_path, 'wb') as handle:
        pickle.dump(clf_ingred_set_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

def main():
    print('Preparing BERT...')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    print('Done')

    # === Prepare embeddings ===
    print('=== Prepare embeddings ===')
    # generate ingredient embeddings
    print('Generating ingredient embeddings...')
    ingred_emb = generate_embedding(model, ingredidx2name_map_path, ingred_emb_path)
    # generate item embeddings
    print('Generating item embeddings...')
    item_emb = generate_embedding(model, itemidx2name_map_path, item_emb_path)
    # generate recipe name embeddings
    print('Generating recipe name embeddings...')
    recipe_name_emb = generate_embedding(model, recipeidx2name_map_path, recipe_name_emb_path)


    # === Calculating main ingredients of recipes ===
    print('=== Calculating main ingredients of recipes ===')
    # calculate main ingredients for recipes
    print('Calculating cosine distance between recipe and ingredients...')
    with open(recipeidx2ingredidx_map_path, 'rb') as f:
        recipeidx2ingredidx_map = pickle.load(f)
    with open(ingredidx2name_map_path, 'rb') as f:
        ingredidx2name_map = pickle.load(f)
    recipe_ingredient_dist_cos = calculate_recipe_ingredient_dist(recipe_name_emb, ingred_emb, recipeidx2ingredidx_map, recipe_ingredient_dist_cos_path)
    print('Done.')

    print('Calculating main ingredients of recipes based on distance...')
    common_ingredient = ['salt', 'sugar', 'black pepper', 'water', 'oil']
    n_main = 5
    n_max_ingred = 8
    get_recipe_main_ingredient(common_ingredient, recipe_ingredient_dist_cos, ingredidx2name_map,  n_main, n_max_ingred, recipeidx2mainingredidx_map_path)
    print('Done.')

    # === Prepare training data ===
    print('=== Prepare training data ===')
    # generate training set
    print('Generating training data set...')
    with open(itemidx2name_map_path, 'rb') as f:
        itemidx2name_map = pickle.load(f)
    item_idx_list = list(itemidx2name_map.keys())
    with open(ingredidx2name_map_path, 'rb') as f:
        ingredidx2name_map = pickle.load(f)
    ingred_idx_list = list(ingredidx2name_map.keys())
    with open(recipeidx2mainingredidx_map_path, 'rb') as f:
        recipeidx2mainingredidx_map = pickle.load(f)
    recipe_idx_list = list(recipeidx2mainingredidx_map.keys())

    prepare_training_set(recipeidx2mainingredidx_map, item_idx_list, ingred_idx_list, recipe_idx_list)
    print('Done.')

    # prepare pretrained embedding
    print('Generating item/ingredients pretrained embedding...')
    with open(item_emb_path, 'rb') as f:
        item_emb = pickle.load(f)
    with open(ingred_emb_path, 'rb') as f:
        ingred_emb = pickle.load(f)

    idx_start = len(ingred_emb)
    embidx2itemidx_map = {}
    for i in itemidx2name_map.keys():
        embidx2itemidx_map[idx_start] = i
        idx_start += 1
    with open(embidx2itemidx_map_path, 'wb') as handle:
        pickle.dump(embidx2itemidx_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ingredient_emb_array = np.array(list(ingred_emb.values()))
    item_emb_array = np.array(list(item_emb.values()))
    emb_pretrained_array = np.concatenate([ingredient_emb_array, item_emb_array])
    emb_pretrained_array[0] = np.zeros([768])
    with open(emb_pretrained_path, 'wb') as handle:
        pickle.dump(emb_pretrained_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done.')

if __name__ == '__main__':
    main()