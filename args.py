import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='reciptor parameters')

    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')

    # data
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--workers', default=30, type=int)

    # model
    # must be N*3 for triplet
    parser.add_argument('--batch_size', default=2000, type=int)
    parser.add_argument('--snapshots', default='./data/snapshots/', type=str)

    # im2recipe model
    parser.add_argument('--embDim', default=600, type=int)
    parser.add_argument('--ingDim', default=600, type=int)
    parser.add_argument('--wVecDim', default=768, type=int)

    # training 
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=5, type=int) # 720
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--alpha', default=0.8, type=float)
    parser.add_argument('--valfreq', default=4, type=int) # 10
    parser.add_argument('--patience', default=1, type=int)
    parser.add_argument('--freeRecipe', default=True, type=bool)

    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--save_tuned_embed', default=False, action="store_true")


    return parser