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
from model.net import ChefCart
from model.data_loader import get_default_device, DeviceDataLoader, ingredsetDataset, recipeEmbDataset, clfDataset
import pickle
from config import *

parser = get_parser()
opts = parser.parse_args()
device = get_default_device()

def main():
    print('batch size', opts.batch_size)

    with open(recipeidx2ingredidx_map_path, 'rb') as f:
        recipeidx2ingredidx_map = pickle.load(f)
    with open(recipeidx2mainingredidx_map_path, 'rb') as f:
        recipeidx2mainingredidx_map = pickle.load(f)
    with open(triplet_train_set_path, 'rb') as f:
        train_set = pickle.load(f)
    with open(triplet_test_set_path, 'rb') as f:
        test_set = pickle.load(f)
    with open(emb_pretrained_path, 'rb') as f:
        emb_pretrained = pickle.load(f)
    # with open(clf_train_set_path, 'rb') as f:
    #     infr_emb_set = pickle.load(f)
    with open(clf_test_set_path, 'rb') as f:
        clf_test_set = pickle.load(f)

    # === Prepare training and validation data ===
    print('=== Prepare training and validation data ===')

    print('Perparing data loader...')
    n = len(train_set)
    train_recipe_cart_data = [e for i, e in enumerate(train_set[:round(0.7*n)]) if i % 3 == 0]
    train_ds = ingredsetDataset(train_recipe_cart_data, recipeidx2mainingredidx_map)
    train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True)
    train_dl = DeviceDataLoader(train_dl, device)

    valid_recipe_cart_data = [e for i, e in enumerate(train_set[round(0.7*n):]) if i % 3 == 0]
    valid_ds = ingredsetDataset(valid_recipe_cart_data, recipeidx2mainingredidx_map)
    valid_dl = DataLoader(valid_ds, batch_size=opts.batch_size, shuffle=True)
    valid_dl = DeviceDataLoader(valid_dl, device)

    test_recipe_cart_data = [e for i, e in enumerate(test_set) if i % 3 == 0]
    test_ds = ingredsetDataset(test_recipe_cart_data, recipeidx2mainingredidx_map)
    test_dl = DataLoader(test_ds, batch_size=opts.batch_size, shuffle=True)
    test_dl = DeviceDataLoader(test_dl, device)

    print('Perparing pretrained embeddings...')
    emb_pretrained_tensor = torch.tensor(emb_pretrained)

    print('Initialize model parameter...')
    model = ChefCart(emb_pretrained_tensor)
    model.to(device)

    criterion = [nn.BCELoss().to(device), nn.TripletMarginLoss(margin=0.05, p=2, reduction='sum').to(device)]
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

    print('There are %d parameter groups' % len(optimizer.param_groups))
    print('Initial base params lr: %f' % optimizer.param_groups[0]['lr'])

    cudnn.benchmark = True

    # run epochs
    for epoch in range(opts.start_epoch, opts.epochs):
        print('=== Start training epoch {} ==='.format(epoch))
        loss = train(train_dl, model, criterion, optimizer, epoch, opts.alpha)

        val_loss, val_acc, v_TPs, v_FPs, v_FNs, _ = valid_metrics(valid_dl, model)
        print(' -> valid loss: {loss:.5f}\t'
              ' valid acc: {acc:.5f}\t'
              ' valid prec: {prec:.5f}\t'
              ' valid recl: {recl:.5f}\t'.format(loss=val_loss.avg, acc=val_acc.avg, prec=v_TPs/(v_TPs+v_FPs), recl=v_TPs/(v_TPs+v_FNs)))

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

            # print('** Validation: %f (best) - %d (valtrack)' % (best_val, valtrack))

    # # save embeding
    # print('=== Save recipe embeddings ===')
    # print('Saving the tuned recipe embeddings...')
    # recipe_index = list(recipeidx2mainingredidx_map.keys())[:100]
    # recipe_ds = recipeEmbDataset(recipe_index, recipeidx2mainingredidx_map)
    # recipe_dl = DataLoader(recipe_ds, batch_size=opts.batch_size, shuffle=False)
    # recipe_dl = DeviceDataLoader(recipe_dl, device)
    # save_recipe_embedding(recipe_dl, model)

    # calculate test accuracy
    test_loss, test_acc, t_TPs, t_FPs, t_FNs, pred = test_metrics(test_dl, model)
    print(' test loss: {loss:.5f}\t'
          ' test acc: {acc:.5f}\t'
          ' test prec: {prec:.5f}\t'
          ' test recl: {recl:.5f}\t'
          ' test f1: {f1:.5f}\t'.format(loss=test_loss.avg, acc=test_acc.avg, prec=t_TPs/(t_TPs+t_FPs), 
                                        recl=t_TPs/(t_TPs+t_FNs), f1=t_TPs/(t_TPs+(t_FPs+t_FNs)/2)))

    with open('./data/output/pred.pickle', 'wb') as f:
        pickle.dump(pred, f)
        print('pred dumped to {}'.format(trained_recipe_emb_path))
        
def train(train_dl, model, criterion, optimizer, epoch, alpha):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    train_start = time.time()
    for i, (ingre_set_pos, ingre_set_neg, recipe_pos, recipe_neg) in enumerate(train_dl):
        # measure data loading time

        data_time.update(time.time() - end)

        x_set = torch.cat((ingre_set_pos[0], ingre_set_neg[0]), 0)
        x_ln = torch.cat((ingre_set_pos[1], ingre_set_neg[1]), 0)
        y = torch.cat((ingre_set_pos[2], ingre_set_neg[2]), 0)
        y_hat = model(x_set, x_ln, clr=1)
        loss_BCE = criterion[0](y_hat, y)

        recipe_anc = model(ingre_set_pos[0], ingre_set_pos[1], clr=0)
        recipe_poc = model(recipe_pos[0], recipe_pos[1], clr=0)
        recipe_neg = model(recipe_neg[0], recipe_neg[1], clr=0)
        loss_triplet = criterion[1](recipe_anc, recipe_poc, recipe_neg)

        loss = alpha * loss_BCE + (1 - alpha) * loss_triplet / 10

        # measure performance and record losses
        losses.update(loss.data, ingre_set_pos[0].size(0))
        print('  batch loss', loss.data.cpu().numpy())

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' Epoch: {ep}\t'
          ' Loss {loss.avg:.5f}\t'
          ' lr {lr}\t'
          ' run time {r_time}\t'.format(ep=epoch, loss=losses,
                                        lr=optimizer.param_groups[0]['lr'], r_time=time.time()-train_start))

    return loss.cpu().data.numpy()


def valid_metrics(test_dl, model):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    TPs = 0
    FPs = 0
    FNs = 0

    x_list = []
    y_list = []
    for i, (ingre_set_pos, ingre_set_neg, _, _) in enumerate(test_dl):
        x_set = torch.cat((ingre_set_pos[0], ingre_set_neg[0]), 0)
        x_ln = torch.cat((ingre_set_pos[1], ingre_set_neg[1]), 0)
        y = torch.cat((ingre_set_pos[2], ingre_set_neg[2]), 0)
        y_hat = model(x_set, x_ln, clr=1)
        loss_func = nn.BCELoss()
        loss = loss_func(y_hat, y)

        # classification accuracy
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

        x_list.extend([ingre_set_pos[0].data.cpu().numpy(), ingre_set_neg[0].data.cpu().numpy()])
        y_list.extend([y.data.cpu().numpy(), y_hat.data.cpu().numpy(), clf_binary])

        losses.update(loss.data, ingre_set_pos[0].size(0))
        accuracies.update(accuracy, ingre_set_pos[0].size(0))
    return losses, accuracies, TPs, FPs, FNs, [x_list, y_list]

def test_metrics(test_dl, model):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    TPs = 0
    FPs = 0
    FNs = 0

    x_list = []
    y_list = []
    for i, (ingre_set_pos, ingre_set_neg, _, _) in enumerate(test_dl):
        x_set = torch.cat((ingre_set_pos[0], ingre_set_neg[0]), 0)
        x_ln = torch.cat((ingre_set_pos[1], ingre_set_neg[1]), 0)
        y = torch.cat((ingre_set_pos[2], ingre_set_neg[2]), 0)
        y_hat = model(x_set, x_ln, clr=1)
        loss_func = nn.BCELoss()
        loss = loss_func(y_hat, y)

        # classification accuracy
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

        x_list.extend([x_set, x_set[0].data.cpu().numpy()])
        y_list.extend([y.data.cpu().numpy(), y_hat.data.cpu().numpy(), clf_binary])

        losses.update(loss.data, ingre_set_pos[0].size(0))
        accuracies.update(accuracy, ingre_set_pos[0].size(0))

    return losses, accuracies, TPs, FPs, FNs, [x_list, y_list]


# save recipe embeddings
def save_recipe_embedding(data_loader, model):
    # switch to evaluate mode
    model.eval()

    recipe_emb_dict = {}
    print('start save the tuned embeddings')
    for i, recipe in enumerate(data_loader):
        output = model(recipe[1], recipe[2], clr=0).data.cpu().numpy()

        recipe_idx = recipe[0].data.cpu().numpy()
        recipe_dict = {}
        for i in range(len(output)):
            recipe_emb = output[i]
            recipe_dict[recipe_idx[i]] = recipe_emb
        recipe_emb_dict.update(recipe_dict)

    with open(trained_recipe_emb_path, 'wb') as f:
        pickle.dump(recipe_emb_dict, f)
        print('trained_recipe_emb dumped to {}'.format(trained_recipe_emb_path))
    return

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = opts.snapshots + 'model_e%03d_v-%.3f.pth' % (state['epoch'], state['best_val'])
    if is_best:
        torch.save(state, filename)
        print('save checkpoint %s' % filename)

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

if __name__ == '__main__':
    main()