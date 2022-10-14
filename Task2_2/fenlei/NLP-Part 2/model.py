import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from transformers import BertModel, RobertaModel


class Bert_Model(nn.Module):
    def __init__(self, config, args):
        super(Bert_Model, self).__init__()
        self.args = args
        self.hidden_size = config.hidden_size
        self.num_hiddens = config.num_hiddens
        self.norm_shape = config.norm_shape
        self.ffn_num_input = config.ffn_num_input
        self.ffn_num_hiddens = config.ffn_num_hiddens
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers


        """
        
        定义bert神经网络
        
        """
        self.cls_dropout = nn.Dropout(args.dropout_rate)  ## 自行查询资料，学习这个层的作用以及用法。
        self.encoder = BERTEncoder(self.hidden_size, self.num_hiddens, self.norm_shape,
                    self.ffn_num_input, self.ffn_num_hiddens, self.num_heads, self.num_layers,
                    self.cls_dropout, max_len=1000, key_size=6251,
                    query_size=6251, value_size=6251)
        self.hidden = nn.Sequential(nn.Linear(6251, self.num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(self.hidden_size, self.num_hiddens, 6251)
        self.nsp = NextSentencePred(6251)

    def forward(self, input_ids, input_mask, segment_ids, concat_prog_ids, multi_prog_ids):


        """

        实现神经网络各层之间的联系，也就是前向传播。

        注：
        1. 该神经网络需要先经过bert层的嵌入，然后取last_hidden_state。
        2. bert的嵌入是为了让机器读懂自然语言同时学习文字之间的关系，嵌入之后之后就是分类，那分类应该如何实现？
        3. 定义的dropout层有什么作用？
        4. 前向传播输入的量可以自行修改，具体研究trainer.py文件中的训练相关代码

        """
        encoded_X = self.encoder(input_ids, input_mask, segment_ids)
        mlm_Y_hat = self.mlm(encoded_X)
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


class BERTEncoder(nn.Module):
    def __init__(self, hidden_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=6251, query_size=6251, value_size=6251,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(self.hidden_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))
    def forward(self, tokens, segments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

#@save
class MaskLM(nn.Module):
    def __init__(self, hidden_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, self.hidden_size))
    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

class NextSentencePred(nn.Module):
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)
    def forward(self, X):
        return self.output(X)