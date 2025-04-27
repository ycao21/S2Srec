import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from args import get_parser
from model.net import SetPredictor
from model.data_loader import get_default_device, DeviceDataLoader, ingredsetDataset, collate_mask_n, collate_mask_n_cpu
import pickle
from config import *

parser = get_parser()
opts = parser.parse_args()
device = get_default_device()

def main():

    with open('./data/recipe_title_dict.pkl', 'rb') as handle:
        recipe_title_dict = pickle.load(handle)

    with open('./data/tag_recipe_list.pkl', 'rb') as handle:
        tag_recipe_list = pickle.load(handle)

    with open('./data/ingredient_dict_frequent.pkl', 'rb') as handle:
        ingredient_dict = pickle.load(handle)
    print('number of ingredient:', len(ingredient_dict))
    ingredient_dict_reverse = {}
    for k, v in ingredient_dict.items():
        ingredient_dict_reverse[v] = k

    with open('./data/emb_dict_frequent.pkl', 'rb') as handle:
        emb_dict = pickle.load(handle)

    with open('./data/recipe_tfidf_frequent.pkl', 'rb') as handle:
        recipe_tfidf = pickle.load(handle)


    # Load the training data from the text file
    training_data_file = './data/training_data_frequent.txt'
    training_data = []

    with open(training_data_file, 'r') as f:
        for line in f:
            np_array = eval(line)
            training_data.append(np_array)

    print(f"Number of data points loaded: {len(training_data)}")

    train_set = training_data

    n = len(train_set)
    train_recipe_cart_data = train_set[:round(0.7*n)]
    train_ds = ingredsetDataset(train_recipe_cart_data)
    train_dl = DataLoader(train_ds, batch_size=opts.batch_size, collate_fn=collate_mask_n,shuffle=True)
    train_dl = DeviceDataLoader(train_dl, device)

    valid_recipe_cart_data = train_set[round(0.7*n):round(0.9*n)]
    valid_ds = ingredsetDataset(valid_recipe_cart_data)
    valid_dl = DataLoader(valid_ds, batch_size=opts.batch_size, collate_fn=collate_mask_n, shuffle=True)
    valid_dl = DeviceDataLoader(valid_dl, device)

    test_recipe_cart_data = train_set[round(0.9*n):]

    print('Perparing pretrained embeddings...')
    emb_pretrained = list(emb_dict.values())
    emb_pretrained_tensor = torch.tensor(emb_pretrained)

    model = SetPredictor(emb_pretrained_tensor)
    model.to(device)

    criterion = [nn.CrossEntropyLoss(), nn.BCELoss()]
    base_params = model.parameters()
    optimizer = torch.optim.Adam([{'params': base_params}], lr=opts.lr * opts.freeRecipe)

    if opts.resume:
        print('Resuming ...')
        if os.path.isfile(opts.resume):
            checkpoint = torch.load(opts.resume)
            opts.start_epoch = checkpoint['epoch']
            best_val = checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(opts.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opts.resume))
            best_val = float('inf')
    else:
        best_val = float('inf')
    valtrack = 0

    for epoch in range(opts.start_epoch, opts.epochs):
        print('=== Start training epoch {} ==='.format(epoch))
        loss = train(train_dl, model, criterion, optimizer, epoch, 0.8)

        val_loss1, val_loss2, val_acc, v_TPs, v_FPs, v_FNs, _ = valid_metrics(valid_dl, model)
        print(' -> valid loss1: {loss1:.5f}\t'
            ' valid loss2: {loss2:.5f}\t'
            ' valid acc: {acc:.5f}\t'
            ' valid prec: {prec:.5f}\t'
            ' valid recl: {recl:.5f}\t'.format(loss1=val_loss1.avg, loss2=val_loss2.avg, acc=val_acc.avg, prec=v_TPs/(v_TPs+v_FPs+1e-8), recl=v_TPs/(v_TPs+v_FNs+ 1e-8)))

        # evaluate on validation set on val freq
        if (epoch + 1) % opts.valfreq == 0 and epoch != 0:
            val_loss = loss
            # check patience
            if val_loss >= best_val:
                valtrack += 1
            else:
                    valtrack = 0

            if valtrack >= opts.patience:
                # change the learning rate accordingly
                adjust_learning_rate(optimizer, epoch, opts)
                valtrack = 0

            # save the best model
            is_best = val_loss < best_val
            best_val = min(val_loss, best_val)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val': best_val,
                'optimizer': optimizer.state_dict(),
                'curr_val': val_loss,
            }, is_best)

    test_data = collate_mask_n_cpu(test_recipe_cart_data)
    recipe_id, all_true_sets, all_pred_sets = test_metrics_n(test_data, model, max_pred=10, topk=10)
    evaluate_stepwise_predictions(all_true_sets, all_pred_sets, topk=3, max_pred=5)
    
def train(train_dl, model, criterion, optimizer, epoch, alpha):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    train_start = time.time()

    # check_gpu_memory()
    for i, (_, set_data, binary_data) in enumerate(train_dl):
        # measure data loading time

        data_time.update(time.time() - end)
        # check_gpu_memory()
        logits = model(set_data[0], set_data[1], mode='mlm')
        loss1 = criterion[0](logits, set_data[2])

        y_hat = model(binary_data[0], binary_data[1], mode='clf')
        loss2 = criterion[1](y_hat,  binary_data[2].unsqueeze(1))

        loss = alpha*loss1 + (1-alpha) * loss2
        # measure performance and record losses
        losses.update(loss.data, set_data[0].size(0))
        if i % 10 == 0:
          print('  batch loss', loss.data.cpu().numpy())

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # check_gpu_memory()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' Epoch: {ep}\t'
          ' Loss {loss.avg:.5f}\t'
          ' lr {lr}\t'
          ' run time {r_time}\t'.format(ep=epoch, loss=losses,
                                        lr=optimizer.param_groups[0]['lr'], r_time=time.time()-train_start))

    return loss.cpu().data.numpy()

def valid_metrics(valid_dl, model, criterion):
    model.eval()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    accuracies = AverageMeter()
    TPs = 0
    FPs = 0
    FNs = 0

    x_list = []
    y_list = []
    for i, (_, set_data, binary_data) in enumerate(valid_dl):
        # measure data loading time

        # check_gpu_memory()
        logits = model(set_data[0], set_data[1], mode='mlm')
        loss1 = criterion[0](logits, set_data[2])

        y_hat = model(binary_data[0], binary_data[1], mode='clf')
        loss2 = criterion[1](y_hat,  binary_data[2].unsqueeze(1))

        # measure performance and record losses
        losses1.update(loss1.data, set_data[0].size(0))
        losses2.update(loss2.data, binary_data[0].size(0))

        # classification accuracy
        y = binary_data[2].unsqueeze(1)
        clf_binary = y_hat.data.cpu().numpy() > 0.5
        correct = (clf_binary == y.data.cpu().numpy()).sum()
        accuracy = correct / clf_binary.shape[0]

        for i in range(len(y)):
            if y[i] == 1:
                if y_hat[i] > 0.5:
                    TPs += 1
                else:
                    FNs += 1
            elif y_hat[i] > 0.5:
                FPs += 1

        x_list.extend([binary_data[0].data.cpu().numpy()])
        y_list.extend([y.data.cpu().numpy(), y_hat.data.cpu().numpy(), clf_binary])

        # measure performance and record losses
        losses1.update(loss1.data, set_data[0].size(0))
        losses2.update(loss2.data, binary_data[0].size(0))
        accuracies.update(accuracy, binary_data[0].size(0))
    return losses1, losses2, accuracies, TPs, FPs, FNs, [x_list, y_list]

def save_checkpoint(state, is_best, model_name='settransformer', filename='checkpoint.pth.tar'):
    """
    Save model checkpoint.

    Args:
        state (dict): Model state to save.
        is_best (bool): If True, this is the best model so far.
        model_name (str): Name of the model (for organizing different baselines).
        filename (str): Default filename (not used if is_best is True).
    """
    # Customize filename with model_name, epoch, and validation score
    filename = '/content/drive/MyDrive/data_eval/{}_e{:03d}_v{:.3f}.pth'.format(
        model_name, state['epoch'], state['best_val']
    )

    if is_best:
        torch.save(state, filename)
        print('Saved checkpoint: {}'.format(filename))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, opts):
    """Switching between modalities"""
    # parameters corresponding to the rest of the network
    optimizer.param_groups[0]['lr'] = opts.lr * opts.freeRecipe
    print('Initial base params lr: %f' % optimizer.param_groups[0]['lr'])
    # after first modality change we set patience to 3
    opts.patience = 3


def evaluate_stepwise_predictions(true_sets, pred_stepwise, max_pred=5, topk=10):
    """
    Evaluate stepwise predictions using:
    - Cumulative Recall@k (union of top-k across all steps)
    - Final Recall@k (based on top-1 from each step)
    - Top-k Success Rate (any step has hit in top-k)

    Args:
        true_sets: List[Set[int]]
        pred_stepwise: List[List[Set[int]]] — per-step top-k predictions
        k: int — top-k to evaluate for step-level success and cumulative recall

    Returns:
        Dict with averaged metrics
    """
    cumulative_recall_list = []
    final_recall_list = []
    success_list = []
    true_lengths = []
    pred_lengths = []

    for gt, step_preds_raw in zip(true_sets, pred_stepwise):
        # print(gt)
        step_preds = step_preds_raw[:max_pred].copy()
        # print(step_preds)
        flat_preds = []
        # step_success = False
        top1_per_step = []

        for i in range(len(step_preds[0])):
            for s in step_preds:
                # print(s)
                if s[i] not in flat_preds:
                  flat_preds.append(s[i])
        # print(flat_preds)

        top1_per_step = flat_preds[:max_pred]
        # flat_preds = set(flat_preds[:topk])
        flat_preds = set(step_preds[0][:topk])
        final_preds = set(top1_per_step)
        # print('final_preds', final_preds)


        cumulative_recall = len(flat_preds & gt) / len(gt) if gt else 0.0
        final_recall = len(final_preds & gt) / len(gt) if gt else 0.0

        cumulative_recall_list.append(cumulative_recall)
        final_recall_list.append(final_recall)
        # success_list.append(1.0 if step_success else 0.0)

        true_lengths.append(len(gt))
        pred_lengths.append(len(step_preds))


    print("=== Stepwise Model ===")
    print("Cumulative Recall@{}: {:.4f}".format(topk, np.mean(cumulative_recall_list)))
    print("Final Recall (Top-1 per step): {:.4f}".format(np.mean(final_recall_list)))
    # print("Top-{} Success Rate: {:.4f}".format(topk, np.mean(success_list)))
    print("MSE of Set Length: {:.4f}".format(np.mean((np.array(true_lengths) - np.array(pred_lengths)) ** 2)))

    return    

from tqdm import tqdm

def test_metrics_n(test_data, model, max_pred=10, topk=1):
    model.eval()
    with torch.no_grad():

        all_true_sets = []
        all_pred_sets = []
        recipe_id = []
        sims_list = []
        for i, data in tqdm(enumerate(test_data)):
            initial_ingredients = data[1][0]
            target = data[1][2]
            set_pred = []

            current_seq = initial_ingredients.copy()
            for step in range(max_pred):
                # Append the mask token to indicate we want to predict the next ingredient.
                input_seq = pad_set(current_seq, 0, 45)
                input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
                seq_lengths = torch.tensor([len(input_seq)], dtype=torch.long, device=device)

                logits = model(input_tensor, seq_lengths, mode='mlm')  # shape [1, seq_len, vocab_size]

                # Extract the logits corresponding to the masked position (last position).

                next_logits = logits
                # Greedy decode: select the token with maximum probability.
                next_id_list = torch.topk(next_logits, 50).indices.cpu().numpy()[0]
                next_id_list_filtered = [i for i in next_id_list if i not in current_seq]

                if len(next_id_list_filtered) > 0:
                    next_id = [int(i) for i in next_id_list_filtered]
                    set_pred.append(next_id[:topk])
                    # set_pred.extend(next_id_list_filtered[:topk])
                else:
                    set_pred.append([0])

                # Otherwise, add the predicted ingredient to the sequence.
                current_seq.append(next_id[0])
                # print('Current step output:', current_seq , '\n')

                # If the model predicts <STOP>, break.
                stop_binary = model(input_tensor, seq_lengths, mode='clf').data.cpu().numpy()
                # print(stop_binary)
                if stop_binary > 0.5:
                    break

            all_true_sets.append(set(target))
            if len(set_pred) == 0:
                set_pred.append([0])
            all_pred_sets.append(set_pred)
            recipe_id.append(data[0])

    return recipe_id, all_true_sets, all_pred_sets

if __name__ == '__main__':
    main()