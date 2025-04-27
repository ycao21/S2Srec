import torch.nn as nn

from args import get_parser
from model.blocks import *

parser = get_parser()
opts = parser.parse_args()


# normalization
def norm(input, p=1, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

class logisticSetTransformer(nn.Module):
    """
    A simple feed-forward alternative to the full SetTransformer.
    It uses:
      - Pretrained embeddings for ingredients.
      - A linear layer to project embeddings to the desired dimension.
      - A simple MLP applied per ingredient.
      - Average pooling to obtain a permutation invariant representation.
      - A final predictor projection.
    """
    def __init__(self, weight):
        super(logisticSetTransformer, self).__init__()
        in_dimension = opts.wVecDim
        out_dimension = opts.embDim

        self.embs = nn.Embedding.from_pretrained(
            weight,
            freeze=True,
            padding_idx=0
        )

        # Project the pretrained embeddings to a new space.
        # self.reduce_dim = nn.Linear(in_dimension, out_dimension)

        # A simple MLP applied independently to each ingredient embedding.
        self.mlp = nn.Sequential(
            nn.Linear(in_dimension, out_dimension),
            nn.ReLU()
        )

        # A final linear projection to get the set-level representation.
        self.predictor = nn.Linear(out_dimension, out_dimension)

    def forward(self, x, sq_lengths, mode='mlm'):
        """
        Args:
            x (LongTensor): Input tensor of ingredient indices with shape [batch, n].
            sq_lengths (Tensor): Tensor with shape [batch] containing the actual number
                                 of ingredients for each recipe (to correctly average pool).
            mode (str): The mode can be 'mlm' or 'clf'; here it doesn't change the backbone.
        Returns:
            Tensor: Set-level output tensor of shape [batch, out_dimension].
        """
        # Get the ingredient embeddings: [batch, n, wVecDim]
        x = self.embs(x)
        # Project to desired dimension: [batch, n, embDim]
        # x = self.reduce_dim(x)
        # Apply feed-forward network on each ingredient embedding.
        x = self.mlp(x)
        # Average pooling over the set (using provided sequence lengths for a correct average)
        # This gives a permutation invariant summary of the ingredient set.
        pooled = torch.sum(x, dim=1) / sq_lengths.view(-1, 1).type_as(x)
        # Final linear projection.
        y = self.predictor(pooled)
        return y


class LSTMSetTransformer(nn.Module):
    def __init__(self, weight):
        """
        A bidirectional LSTM alternative to the SetTransformer.

        Args:
            weight (Tensor): Pretrained embedding weights.
        """
        super(LSTMSetTransformer, self).__init__()
        in_dimension = opts.wVecDim       # Dimension of the ingredient embeddings.
        out_dimension = opts.embDim       # Desired output (hidden) dimension.

        # Pretrained ingredient embeddings (frozen).
        self.embs = nn.Embedding.from_pretrained(
            weight,
            freeze=True,
            padding_idx=0
        )

        # Bidirectional LSTM.
        # Since the LSTM is bidirectional, we use out_dimension//2 as hidden size so that the concatenated
        # output becomes out_dimension.
        self.lstm = nn.LSTM(
            input_size=in_dimension,
            hidden_size=out_dimension // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Final projection layer.
        self.predictor = nn.Linear(out_dimension, out_dimension)

    def forward(self, x, sq_lengths, mode='mlm'):
        """
        Forward pass of the LSTM-based backbone.

        Args:
            x (LongTensor): Input tensor of ingredient indices with shape [batch, seq_len].
            sq_lengths (Tensor): Tensor with the actual sequence lengths [batch].
            mode (str): Although not used to change operations here, kept for interface consistency.

        Returns:
            Tensor: A set-level representation of shape [batch, out_dimension].
        """
        # Lookup embeddings: [batch, seq_len, wVecDim]
        x = self.embs(x)

        # Pack the sequence to ignore padded positions.
        packed = nn.utils.rnn.pack_padded_sequence(x, sq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)

        # h_n has shape [num_directions, batch, hidden_dim].
        # For a bidirectional LSTM with one layer: h_n.shape = [2, batch, embDim/2].
        # Concatenate the hidden states from both directions.
        h_n_cat = torch.cat([h_n[0], h_n[1]], dim=1)  # Shape: [batch, embDim]

        # Final projection.
        y = self.predictor(h_n_cat)
        return y

class FFNNSetTransformer(nn.Module):
    """
    A simple feed-forward alternative to the full SetTransformer.
    It uses:
      - Pretrained embeddings for ingredients.
      - A linear layer to project embeddings to the desired dimension.
      - A simple MLP applied per ingredient.
      - Average pooling to obtain a permutation invariant representation.
      - A final predictor projection.
    """
    def __init__(self, weight):
        super(FFNNSetTransformer, self).__init__()
        in_dimension = opts.wVecDim
        out_dimension = opts.embDim

        self.embs = nn.Embedding.from_pretrained(
            weight,
            freeze=True,
            padding_idx=0
        )

        # Project the pretrained embeddings to a new space.
        self.reduce_dim = nn.Linear(in_dimension, out_dimension)

        # A simple MLP applied independently to each ingredient embedding.
        self.mlp = nn.Sequential(
            nn.Linear(out_dimension, out_dimension),
            nn.ReLU(),
            nn.Linear(out_dimension, out_dimension)
        )

        # A final linear projection to get the set-level representation.
        self.predictor = nn.Linear(out_dimension, out_dimension)

    def forward(self, x, sq_lengths, mode='mlm'):
        """
        Args:
            x (LongTensor): Input tensor of ingredient indices with shape [batch, n].
            sq_lengths (Tensor): Tensor with shape [batch] containing the actual number
                                 of ingredients for each recipe (to correctly average pool).
            mode (str): The mode can be 'mlm' or 'clf'; here it doesn't change the backbone.
        Returns:
            Tensor: Set-level output tensor of shape [batch, out_dimension].
        """
        # Get the ingredient embeddings: [batch, n, wVecDim]
        x = self.embs(x)
        # Project to desired dimension: [batch, n, embDim]
        x = self.reduce_dim(x)
        # Apply feed-forward network on each ingredient embedding.
        x = self.mlp(x)
        # Average pooling over the set (using provided sequence lengths for a correct average)
        # This gives a permutation invariant summary of the ingredient set.
        pooled = torch.sum(x, dim=1) / sq_lengths.view(-1, 1).type_as(x)
        # Final linear projection.
        y = self.predictor(pooled)
        return y

class SetEncoder(nn.Module):
    def __init__(self, weight):
        """
        Arguments:
            in_dimension: an integer.  # 2
            out_dimension: an integer. # 5 * K
        """
        super(SetEncoder, self).__init__()
        in_dimension = opts.wVecDim
        out_dimension = opts.embDim

        self.embs = nn.Embedding.from_pretrained(
            weight,
            freeze=True,
            padding_idx=0
            )

        d = out_dimension
        m = 16  # number of inducing points
        h = 4  # number of heads
        k = 2  # number of seed vectors

        self.reduce_dim = nn.Linear(in_dimension, out_dimension)

        self.encoder = nn.Sequential(
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d)),
            InducedSetAttentionBlock(d, m, h, RFF(d), RFF(d))
        )

        # self.decoder = nn.Sequential(
        #     PoolingMultiheadAttention(d, k, h, RFF(d)),
        #     SetAttentionBlock(d, h, RFF(d)),
        #     SetAttentionBlock(d, h, RFF(d))
        # )

        self.predictor = nn.Linear(out_dimension, out_dimension)

        # self.mask_query = nn.Parameter(torch.randn(1, 1, d))
        # self.mask_attn  = MultiheadAttentionBlock(d, h, RFF(d))
        # self.mlm_head = nn.Linear(d, opts.vocab_size)

    def forward(self, x, sq_lengths, mode='mlm'):
        """
        Arguments:
            x: a float tensor with shape [batch, n, in_dimension].
        Returns:
            a float tensor with shape [batch, out_dimension].
        """
        x = self.embs(x) # shape [b, n, d]
        x = self.reduce_dim(x)
        x = self.encoder(x)
        # x = self.decoder(x)
        # b, k, d = x.shape
        # x = x.view(b, k * d)
        x = x.mean(dim=1)
        y = self.predictor(x)
        return y
    


class ElementEncoder(nn.Module):
    def __init__(self, args=0):
        super(ElementEncoder, self).__init__()
        in_dimension = opts.wVecDim
        hidden_dim = opts.embDim

        self.mlp = nn.Sequential(nn.Linear(in_dimension, hidden_dim), # 300 x 300
                                 nn.Dropout(opts.dropout_rate),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.Dropout(opts.dropout_rate),
                                 nn.ReLU())

    def activate_shallow(self, X):

        return self.mlp(X)

    def activate(self, X, m):
        X = self.mlp(X)

        return X, m

    def forward(self, batch):
        batch['eQ'] = self.mlp(batch['xQ'])
        batch['eA'] = self.mlp(batch['xA'])

        return batch

class Kitchenette(nn.Module):
    def __init__(self, weight):
        """
        A “Kitchenette” next-item predictor with the same API as
        LogisticRegressionMLMPredictor:
          - Inputs:  x: LongTensor [batch, seq_len], sq_lengths: LongTensor [batch]
          - Output:  logits [batch, vocab_size]

        Args:
            weight (Tensor): pretrained embedding matrix (num_tokens × wVecDim)
            hidden_dim (int): hidden size for the feed‑forward encoder
            dropout_rate (float): dropout probability
            vocab_size (int): number of ingredient classes
        """
        super(Kitchenette, self).__init__()
        # 1) frozen pretrained embeddings

        self.embs = nn.Embedding.from_pretrained(
            weight, freeze=True, padding_idx=0
        )
        in_dimension = opts.wVecDim
        hidden_dim = opts.embDim

        self.encoder = ElementEncoder(opts)

        # 2) deep encoder on pooled embeddings
        self.deep_encoder = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Dropout(opts.dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(opts.dropout_rate),
            nn.ReLU()
        )
        # 3) final MLM head
        self.predictor = nn.Linear(hidden_dim**2+hidden_dim, opts.embDim)

    def forward(self, x, sq_lengths, mode='mlm'):

        # 1) embed + average‐pool to get [batch, wVecDim]
        emb = self.embs(x)  # [B, N, wVecDim]
        pooled = emb.sum(dim=1) / sq_lengths.view(-1,1).float()  # [B, wVecDim]

        # 2) package into Kitchenette’s expected batch‐dict
        batch = {
            'xQ': pooled,
            'xA': pooled
        }

        # 3) run through ElementEncoder
        batch = self.encoder(batch)
        B = pooled.size(0)

        # 4) compute the outer‐product feature wQ
        eQ, eA = batch['eQ'], batch['eA']            # both [B, hidden_dim]
        wQ = torch.bmm(eQ.unsqueeze(2), eA.unsqueeze(1))  # [B, hidden_dim, hidden_dim]
        wQ = wQ.view(B, -1)                          # [B, hidden_dim^2]

        # 5) deep encoder on the concatenated embeddings
        dQ = torch.cat([eQ, eA], dim=1)              # [B, 2*hidden_dim]
        dQ = self.deep_encoder(dQ)                   # [B, hidden_dim]

        # 6) final feature and prediction
        feat = torch.cat([wQ, dQ], dim=1)             # [B, hidden_dim^2 + hidden_dim]
        feat = self.predictor(feat)                 # [B, vocab_size]
        return feat    
    
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

        self.embs = nn.Embedding.from_pretrained(
            weight,
            freeze=True,
            padding_idx=0
            )

        d = out_dimension
        m = 16  # number of inducing points
        h = 4  # number of heads
        k = 2  # number of seed vectors

        self.reduce_dim = nn.Linear(in_dimension, out_dimension)

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

        # self.mask_query = nn.Parameter(torch.randn(1, 1, d))
        # self.mask_attn  = MultiheadAttentionBlock(d, h, RFF(d))
        # self.mlm_head = nn.Linear(d, opts.vocab_size)

    def forward(self, x, sq_lengths, mode='mlm'):
        """
        Arguments:
            x: a float tensor with shape [batch, n, in_dimension].
        Returns:
            a float tensor with shape [batch, out_dimension].
        """
        x = self.embs(x) # shape [b, n, d]
        x = self.reduce_dim(x)
        x = self.encoder(x)
        x = self.decoder(x)
        b, k, d = x.shape
        x = x.view(b, k * d)
        y = self.predictor(x)
        return y

class S2Srec2(nn.Module):
    def __init__(self, weight):
        super(S2Srec2, self).__init__()
        model_type = opts.model_type
        if model_type == 's2srec2':
            self.ingSetTransformer_ = SetTransformer(weight)
        elif model_type == 'logistic':
            self.ingSetTransformer_ = logisticSetTransformer(weight)
        elif model_type == 'bilstm':
            self.ingSetTransformer_ = LSTMSetTransformer(weight)
        elif model_type == 'ffnn':
            self.ingSetTransformer_ = FFNNSetTransformer(weight)
        elif model_type == 'kitchenette':
            self.ingSetTransformer_ = Kitchenette(weight)

        self.mlm_head = nn.Linear(opts.embDim, opts.vocab_size)

        self.clf_head = nn.Sequential(
            nn.Linear(opts.embDim, 108),
            nn.ReLU(),
            nn.Linear(108, 1),
            nn.Sigmoid()
        )

    def forward(self, x, sq_lengths, mode='mlm'):
        output = self.ingSetTransformer_(x, sq_lengths, mode=mode)

        if mode == "mlm":
            output = self.mlm_head(output)
        elif mode == "clf":
            output = self.clf_head(output)

        return output
