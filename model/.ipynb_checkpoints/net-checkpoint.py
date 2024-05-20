import torch.nn as nn

from args import get_parser
from model.blocks import *

parser = get_parser()
opts = parser.parse_args()


# normalization
def norm(input, p=1, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


class SetTransformer(nn.Module):
    def __init__(self, weight):
        """
        Arguments:
            in_dimension: an integer.  # 2
            out_dimension: an integer. # 5 * K
        """
        super(SetTransformer, self).__init__()
        in_dimension = opts.wVecDim
        out_dimension = opts.embDim

        self.embs = nn.Embedding.from_pretrained(weight)

        d = in_dimension
        m = 16  # number of inducing points
        h = 4  # number of heads
        k = 2  # number of seed vectors

        self.encoder = nn.Sequential(
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))
        )
        self.decoder = nn.Sequential(
            PoolingMultiheadAttention(d, k, h, RFF(d)),
            SetAttentionBlock(d, h, RFF(d)),
            SetAttentionBlock(d, h, RFF(d))
        )

        self.predictor = nn.Linear(k * d, out_dimension)

    def forward(self, x, sq_lengths):
        """
        Arguments:
            x: a float tensor with shape [batch, n, in_dimension].
        Returns:
            a float tensor with shape [batch, out_dimension].
        """
        x = self.embs(x) # shape [b, n, d]

        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        batch_max_len = sorted_len.cpu().numpy()[0]

        cut_x = x[:, :batch_max_len, :]

        x = self.encoder(cut_x)  # shape [batch, batch_max_len, d]
        x = self.decoder(x)  # shape [batch, k, d]

        b, k, d = x.shape
        x = x.view(b, k * d)

        y = self.predictor(x)
        return y


class ChefCart(nn.Module):
    def __init__(self, weight):
        super(ChefCart, self).__init__()

        self.ingre_embedding = nn.Sequential(
            nn.Linear(opts.ingDim, opts.embDim),
            nn.Tanh(),
        )
        self.ingSetTransformer_ = SetTransformer(weight)

        self.sigmoid = nn.Sigmoid()
        self.feedforward_clf = nn.Sequential(
            nn.Linear(opts.embDim, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x, sq_lengths, clr=0):
        output = self.ingSetTransformer_(x, sq_lengths)

        if clr == 1:
            output = self.sigmoid(self.feedforward_clf(output))
        else:
            output = self.ingre_embedding(output)
            output = norm(output)

        return output

#
# class CategoryClassification(nn.Module):
#     def __init__(self):
#         super(CategoryClassification, self).__init__()
#         self.dropout = nn.Dropout(opts.hidden_dropout_prob)
#         self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)
#
#     def forward(self, ingre_emb):
#         pooled_output = self.dropout(ingre_emb)
#         recipe_class = self.semantic_branch(pooled_output)
#         return recipe_class