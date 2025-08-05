import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Dropout, Softmax, Linear, LayerNorm
from .MultiheadAttention import MultiheadAttention


class Embedding(nn.Module):
    def __init__(self, bev_in_channels, prior_in_channels, out_channels, img_size, patch_size, dropout_rate=0.1):
        super(Embedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size[0]

        self.grid_size = (int(img_size[0] / patch_size[0]), int(img_size[1] / patch_size[1]))
        self.n_patches = self.grid_size[0] * self.grid_size[1]

        self.bev_patch_embedding = Conv2d(in_channels=bev_in_channels,
                                          out_channels=out_channels,
                                          kernel_size=self.patch_size,
                                          stride=self.patch_size)
        self.prior_patch_embedding = Conv2d(in_channels=prior_in_channels,
                                            out_channels=out_channels,
                                            kernel_size=self.patch_size,
                                            stride=self.patch_size)

        # To do: 仿照Transfusion，使用PositionEmbeddingLearned，替代可学习参数
        self.position_embedding_bev = nn.Parameter(torch.zeros([1, self.n_patches, out_channels]))
        self.position_embedding_prior = nn.Parameter(torch.zeros([1, self.n_patches, out_channels]))

        self.norm_bev = LayerNorm(out_channels)
        self.norm_prior = LayerNorm(out_channels)

        self.dropout = Dropout(dropout_rate)

    # x   [B, C, H, W]
    def forward(self, bev_feature, prior_feature):
        bev_feature = self.bev_patch_embedding(bev_feature)  # [B, C_hidden, gird_size[0], grid_size[1]]
        bev_feature = bev_feature.flatten(2).transpose(-1, -2)  # [B, n_patches, hidden]

        prior_feature = self.prior_patch_embedding(prior_feature)
        prior_feature = prior_feature.flatten(2).transpose(-1, -2)

        bev_embedding = bev_feature + self.position_embedding_bev
        bev_embedding = self.dropout(bev_embedding)
        bev_embedding = self.norm_bev(bev_embedding)

        prior_embedding = prior_feature + self.position_embedding_prior
        prior_embedding = self.dropout(prior_embedding)
        prior_embedding = self.norm_prior(prior_embedding)

        return bev_embedding, prior_embedding


# To do: save decoder_layer, dropout, activation, ffn_channels, cross_only, num_heads as config
class Fusion_Transformer(nn.Module):
    def __init__(self, bev_in_channels,
                 prior_in_channels,
                 hidden_channels,
                 img_size=(200, 400),
                 patch_size=(10, 10),
                 decoder_layer=3,
                 dropout=0.1,
                 activation='relu',
                 ffn_channels=1024,
                 cross_only=False,
                 num_heads=8
                 ):
        super(Fusion_Transformer, self).__init__()
        self.hidden_channels = hidden_channels
        self.bev_in_channels = bev_in_channels
        self.prior_in_channels = prior_in_channels
        self.dropout = dropout
        self.img_size = img_size
        self.patch_size = patch_size[0]

        self.grid_size = (int(img_size[0] / patch_size[0]), int(img_size[1] / patch_size[1]))
        self.n_patches = self.grid_size[0] * self.grid_size[1]

        self.embedding = Embedding(bev_in_channels, prior_in_channels, hidden_channels, img_size, patch_size)
        self.decoder_layers = decoder_layer
        self.decoder = []
        for i in range(decoder_layer):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channels, num_heads, ffn_channels, dropout, activation, cross_only
                )
            )
        self.decoder = nn.ModuleList(self.decoder)
        self.expand = nn.Linear(self.hidden_channels, (self.patch_size**2)*self.bev_in_channels, bias=False)
        self.drop = Dropout(self.dropout)

    def forward(self, bev_feature, prior_feature):
        bev_embedding, prior_embedding = self.embedding(bev_feature, prior_feature)  # [B, n_patches, hidden]
        query_feat = bev_embedding
        for i in range(self.decoder_layers):
            query_feat = self.decoder[i](query_feat, prior_embedding) # [B, n_patches, hidden]

        bsz, n_patch, hidden = query_feat.size()
        query_feat = self.expand(query_feat)  # [B, n_patches, patch_size*patch_size*c_bev]
        query_feat = self.drop(query_feat)

        x = query_feat.permute(0, 2, 1)
        x = x.contiguous().view(bsz, self.bev_in_channels, self.img_size[0], self.img_size[1])  # [B, C_hidden, gird_size[0], grid_size[1]]

        return torch.cat([x, prior_feature], dim=1)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", cross_only=False):
        super(TransformerDecoderLayer, self).__init__()
        self.cross_only = cross_only
        # if not self.cross_only:
        #     self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

    def forward(self, query, key, attn_mask=None):
        """
        :param query: [B, n_patches, hidden]
        :param key: [B, n_patches, hidden]
        :param value: [B, n_patches, hidden]
        :return:
        """
        # query = query.permute(1, 0, 2)  # [n_patches, B, hidden]
        # key = key.permute(1, 0, 2)
        value = key

        # if not self.cross_only:
        #     q = k = v = query
        #     query2 = self.self_attn(q, k, v, attn_mask)[0]
        #     query = query + self.dropout1(query2)
        #     query = self.norm1(query)
        query2 = self.cross_attn(query, key, value, attn_mask=attn_mask)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # query = query.permute(1, 0, 2)  # [B, n_patches, hidden]
        return query
