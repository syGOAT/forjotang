import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.utils import rnn
import re
import numpy as np
from collections import Iterable
import shutil
import os
import logging


def get_onehot(indices, depth):
    max_len = indices.size(1)
    one_hot = torch.zeros_like(indices).unsqueeze(-1).repeat(1,1,max_len)
    one_hot.scatter_(-1, indices.unsqueeze(-1),1)
    return one_hot


class gcn_layer_v2(nn.Module):
    def __init__(self,input_size, dep_count, gcn_size=300, n_iterator=1, shared_variable=True):
        super().__init__()
        self.input_size = input_size
        self.gcn_size = gcn_size
        self.n_iterator = n_iterator
        self.dep_count = dep_count
        self.shared_variable = shared_variable

        self.loop_w = nn.Linear(input_size, gcn_size)
        self.child_w = nn.Linear(input_size, gcn_size)
        self.parent_w = nn.Linear(input_size, gcn_size)
        self.loop_g = nn.Linear(input_size, 1)
        self.child_g = nn.Linear(input_size ,1)
        self.parent_g = nn.Linear(input_size, 1)

        self.p_embedings_w = nn.Embedding(dep_count,gcn_size)
        self.c_embedings_w = nn.Embedding(dep_count,gcn_size)
        self.p_embedings_g = nn.Embedding(dep_count,1)
        self.c_embedings_g = nn.Embedding(dep_count,1)

    def forward(self, inputs, parent_positions, sentence_deps, attention_mask):
        """
        Imitate ``https://github.com/giorgio-mariani/Semantic-Role-Labeling``
        :param input_gcn: type:tensor, shape: batch_size * max_len * dim, sentence embeddings
        :param parent_positions: type: tensor, shape: batch_size * max_len, head position for current word in full sentence.
        :param sentence_deps: type: tensor, shape: batch_size * max_len
        :param dep_count: type:int: total num of dependency types
        :return:
        output gcn representation. type:tensor, shape: batch_size * max_len * dim

        Cautions: parent_position: padded by actual length for 0 represents a parent.
        edge representation are decided by edge type(sentence deps)
        """
        # attention 这里只需要mask掉 child就行。 mask child, mask邻接矩阵。mask掉parent_position.
        gcn_outputs = inputs
        for num in range(self.n_iterator):
            # print('parent_position',parent_positions)
            # print('gcn_ouputs',gcn_outputs.size())
            par_gcn = torch.gather(gcn_outputs,1, parent_positions.unsqueeze(-1).repeat(1,1,gcn_outputs.size(2)))
            # print('par_gcn',par_gcn)
            loop_gate = torch.sigmoid(self.loop_g(gcn_outputs)).repeat(1,1,self.gcn_size)
            par_gate = torch.sigmoid(self.parent_g(gcn_outputs) \
                                     + self.p_embedings_g(sentence_deps)).repeat(1,1,self.gcn_size)
            child_gate = torch.sigmoid(self.child_g(gcn_outputs)\
                                       + self.c_embedings_g(sentence_deps)).repeat(1,1,self.gcn_size)

            loop_conv = loop_gate * self.loop_w(gcn_outputs)
            parent_conv = par_gate * (self.parent_w(par_gcn) + self.p_embedings_w(sentence_deps))
            child_mat = child_gate * (self.child_w(gcn_outputs) + self.c_embedings_w(sentence_deps))

            masked_parent_positions = get_mask(parent_positions, attention_mask)
            adj_dep = get_onehot(masked_parent_positions, gcn_outputs.size(1))

            child_conv = torch.matmul(adj_dep.permute(0,2,1).float(), child_mat)
            gcn_outputs = loop_conv + parent_conv + child_conv
        gcn_outputs = gcn_outputs * attention_mask.unsqueeze(-1).float()
        return gcn_outputs


class LSTMWrapper(nn.Module):
    """
    ```
     python example
    input = torch.randn(*(3, 5, 2))
    length_list = torch.tensor([4,3,5])
    lstm = LSTMWrapper(2, 2)
    print(lstm(input, length_list))
    ```
    """
    def __init__(self, input_dim, hidden_dim, n_layer=1, concat=False, bidir=True, dropout=0.3, return_last=True):
        super(LSTMWrapper, self).__init__()
        self.rnns = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                input_dim_ = input_dim
                output_dim_ = hidden_dim
            else:
                input_dim_ = hidden_dim if not bidir else hidden_dim * 2
                output_dim_ = hidden_dim
            self.rnns.append(nn.LSTM(input_dim_, output_dim_, 1, bidirectional=bidir, batch_first=True))
        self.dropout = dropout
        self.concat = concat
        self.n_layer = n_layer
        self.return_last = return_last

    def forward(self, input, input_lengths=None):
        # input_length must be in decreasing order
        # sort input accroding to input lengths
        _, idx_sort = torch.sort(input_lengths, dim=-1, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=-1)
        input = torch.index_select(input, 0, idx_sort)
        input_lengths = torch.index_select(input_lengths, 0, idx_sort)

        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []

        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        for i in range(self.n_layer):
            output = F.dropout(output, p=self.dropout, training=self.training)

            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, _ = self.rnns[i](output)

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = output.data.new(1, 1, 1).fill_(0)
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            output = torch.index_select(output, 0, idx_unsort)

            outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]



def get_mask(parent_positions, attention_mask):
    """

    :param parent_positions: batch_size * max_len
    :param attention_mask: batch_size * max_len
    :return:
    """
    zeros_vec = torch.zeros_like(parent_positions)
    masked_parent_positions = torch.where(attention_mask != 0, parent_positions, zeros_vec)
    return masked_parent_positions

def get_sequence_labels(type='bridge'):
    """
     return sequence label list
    """
    # if type == 'bridge':
    #     return ['[PAD]', '[CLS]', 'O', 'S', 'K-S', 'X']
    # elif type == 'intersec':
    #     return ['[PAD]', '[CLS]','s1', 's2','k','c', 'd', 'X']
    # elif type == 'bridge_intersec':
    #     # return ['[PAD]', '[CLS]', 'O', 'S', 'K-S', 'X', 's1', 's2','k','c', 'd']
    #     return {'[PAD]':0, '[CLS]':1, 'O':2, 'S':3, 'K-S':4, 'X':5,'s1':3, 's2':2,
    #             'k':4, 'c':6, 'd':7,'bridge':8, 'intersec':9}
    # elif type  == 'comparison':
    #     question_dict = {'bridge': 0, 'intersec': 1, 'comparison': 2}
    #     direction_dict = {'L':0,'S':1, 'YW':2}
    #     wh_dict = {'NONE':0, 'HOW':1, 'WHEN':2, 'HOW MANY':3, 'HOW LONG':4, 'HOW OLD':5, 'HOW OFTEN':6, 'HOW LARGE':7,
    #                'HOW WIDE':8, 'HOW HIGH':9, 'HOW BIG':10, 'HOW FAR':11,'HOW TALL':12, 'HOW STRONG':13}
    #     seq_label = {'[PAD]':0,'[CLS]':1,'S-COM':2,'K-COM':3, 'W-COM':4, 'C-COM':5, 'O-COM':6,'X':7}
    #     return question_dict, direction_dict, wh_dict, seq_label

    direction_dict = {'L':0, 'S':1, 'YW':2, 'NONE':3, 'Y':4, 'YS':5,'YC':6,'YL':7,'YSA':8, 'C':9,'YD':10}
    wh_dict = {'NONE': 0, 'HOW': 1, 'WHEN': 2, 'HOW MANY': 3, 'HOW LONG': 4, 'HOW OLD': 5, 'HOW OFTEN': 6,
               'HOW LARGE': 7, 'HOW WIDE': 8, 'HOW HIGH': 9, 'HOW BIG': 10, 'HOW FAR': 11,
               'HOW TALL':12, 'HOW STRONG':13, 'WHAT':14, 'WHERE':15}

    if type == 'combined':
        # seq_label = {'[PAD]': 3, '[CLS]': 4,'O':5, 'S':6,'K-S':7, 'X':8, 'S1':6, 'S2':13, 'K':12,
        #              'S-COM': 6, 'K-COM': 11, 'W-COM': 10, 'C-COM': 9, 'O-COM': 5,'bridge':0,'intersec':1, 'comparison':2}
        #
        # seq_label = {'[PAD]':6,'[CLS]':7, 'K-S':8 ,'S':9 ,'O':10 ,"K_ATT":11,'S_ATT':9,'O_ATT':10, 'S_ADV':12, 'O_ADV':10, 'K_I':13, 'S1_I':9,
        # 'S2_I':14, 'K_N':13, 'O_N':8, 'X':15,'bridge':0,'intersec':1, 'comparison':2, 'no_decom':3, 'adv':4, 'att':5} # 0.7157
        # seq_label = {'[PAD]': 6, '[CLS]': 7, 'K-S': 8, 'S': 9, 'O': 10, "K_ATT": 11, 'S_ATT': 9, 'O_ATT': 10,
        #              'S_ADV': 12, 'O_ADV': 10, 'K_I': 13, 'S1_I': 9,
        #              'S2_I': 14, 'K_N': 16, 'O_N': 9, 'X': 15, 'bridge': 0, 'intersec': 1, 'comparison': 2,
        #              'no_decom': 3, 'adv': 4, 'att': 5}

        seq_label = {'[PAD]': 5, '[CLS]': 6, 'K_B': 7, 'S_B': 8, 'O_B': 9,
                     'S_ADV': 10, 'O_ADV': 11, 'K_I': 12, 'S1_I':8 , 'K_ATT':16, 'S_ATT':7, 'O_ATT': 9,
                     'S2_I': 13, 'K_N': 12, 'O_N':8 , 'X': 15,'S_COM':8,'K_COM':16,"W_COM":12,'C_COM':14,'O_COM':9,'bridge': 0, 'intersec': 1, 'comparison': 2,
                     'no_decom': 3, 'adv': 4,}

        # seq_label = {'[PAD]': 5, '[CLS]': 6, 'K_B': 7, 'S_B': 8, 'O_B': 9,
        #              'S_ADV': 11, 'O_ADV': 7, 'K_I': 11, 'S1_I': 8,
        #              'S2_I': 12, 'K_N': 11, 'O_N': 8, 'X': 13, 'S_COM': 8, 'K_COM': 12, "W_COM": 11, 'C_COM': 10,
        #              'O_COM': 9, 'bridge': 0, 'intersec': 1, 'comparison': 2,
        #              'no_decom': 3, 'adv': 4 }

        # seq_label = {'[PAD]': 5, '[CLS]': 6, 'K_B': 7, 'S_B': 8, 'O_B': 16, 'S_ADV': 14, 'O_ADV':15 , 'K_I': 18, 'S1_I': 17, 'S2_I': 12, 'K_N': 11, 'O_N': 17, 'X': 13, 'S_COM': 17, 'K_COM': 19, 'W_COM': 11, 'C_COM': 10, 'O_COM': 9, 'bridge': 0, 'intersec':1, 'comparison':2,'no_decom': 3, 'adv': 4}
        # seq_label = {'[PAD]': 5, '[CLS]': 6, 'K_B': 7, 'S_B': 8, 'O_B': 16, 'S_ADV': 14, 'O_ADV': 15, 'K_I': 20, 'S1_I': 17, 'S2_I': 12, 'K_N': 11, 'O_N': 21, 'X': 13, 'S_COM': 18, 'K_COM': 19, 'W_COM': 11, 'C_COM': 10, 'O_COM': 9, 'bridge': 0, 'intersec': 1, 'comparison': 2, 'no_decom': 3, 'adv': 4}
        # seq_label = {'[PAD]': 5, '[CLS]': 6, 'K_B': 7, 'S_B': 8, 'O_B': 16, 'S_ADV': 14, 'O_ADV': 15, 'K_I': 11, 'S1_I': 8, 'S2_I': 12, 'K_N': 11, 'O_N': 8, 'X': 13, 'S_COM': 8, 'K_COM': 17, 'W_COM': 11, 'C_COM': 10, 'O_COM': 9, 'bridge': 0, 'intersec': 1, 'comparison': 2, 'no_decom': 3, 'adv': 4}
        # seq_label = {'[PAD]': 5, '[CLS]': 6, 'K_B': 7, 'S_B': 8, 'O_B': 9, 'S_ADV': 10, 'O_ADV': 11, 'K_I': 12,
        #              'S1_I': 13, 'S2_I': 14, 'K_N': 15, 'O_N': 16, 'X': 17, 'S_COM': 18, 'K_COM': 19, 'W_COM': 20,
        #              'C_COM': 21, 'O_COM': 22, 'bridge': 0, 'intersec': 1, 'comparison': 2, 'no_decom': 3, 'adv': 4}

        # seq_label = {'[PAD]': 1, '[CLS]': 2, "K_ATT": 3, 'S_ATT': 4, 'O_ATT': 5, 'X':6,'adv':0}
        # seq_label = {'[PAD]': 2, '[CLS]': 3, "K_ATT": 4, 'S_ATT': 5, 'O_ATT': 6, 'X': 7, 'S_ADV': 8, 'O_ADV': 9,'adv': 0, 'att':1}


        """metric {'keywords_em': 0.0, 'keywords_f1': 0.0, 'constraints_em': 0.0, 'constraints_f1': 0.0, 'acc': 0.7347368421052631, 'wh_em': 0, 'wh_f1': 0, 'comparison_em': 0, 'comparison_f1': 0, 's_wh_acc': 1.0, 'direction_acc': 1.0, 'ques_type_acc': 0.8442105263157895, 'bridge_acc': 0.7697368421052632, 'intersec_acc': 0.8247422680412371, 'comparison_acc': 0.0, 'no_decom_acc': 0.6310679611650486, 'adv_acc': 0.794392523364486, 'att_acc': 0.125, 'bridge_type_acc': 0.9276315789473685, 'intersec_type_acc': 0.9278350515463918, 'comparison_type_acc': 0.0, 'no_decom_type_acc': 0.6407766990291263, 'adv_type_acc': 0.9345794392523364, 'att_type_acc': 0.25}"""
    elif type == 'intersec':
        seq_label = {'[PAD]': 0, '[CLS]': 6,'S1_I':2,'S2_I':3,'K_I':4,'X':5, 'intersec':1}
    elif type == 'bridge':
        seq_label = {'[PAD]': 6, '[CLS]': 1, 'O_B': 2, 'S_B': 3, 'K_B': 4, 'X': 5,'bridge':0}
    elif type == 'comparison':
        seq_label = {'[PAD]':0,'[CLS]':1,'S-COM':8,'K-COM':3, 'W-COM':4, 'C-COM':5, 'O-COM':6,'X':7,'comparison':2}
        seq_label = {'[PAD]': 0, '[CLS]': 1, 'S_COM': 8, 'K_COM': 3, 'W_COM': 4, 'C_COM': 5, 'O_COM': 6, 'X': 7,'comparison':2}
    elif type == 'adv':
        seq_label = {'[PAD]': 0, '[CLS]': 1, 'S_ADV': 2, 'O_ADV': 3, 'X': 5, 'adv': 4}
    elif type == 'no_decom':
        seq_label = {'[PAD]': 0, '[CLS]': 1, 'K_N': 2, 'O_N': 4, 'X': 5, 'no_decom': 3}

    question_dict ={'bridge':0,'intersec':1, 'comparison':2,'no_decom':3, 'adv':4, 'att':5, 'multi_decom':6}
    return  question_dict, direction_dict, wh_dict, seq_label

        # no_reach =[[7,12], [7, 11], [7, 10], [7, 9], [12,7], [12,11], [12,10], [12,9], [11, 7], [11,12]]
# def get_sequence_acc(predict, gold):
#     """
#     find start, end ,keyword position.
#     example:
#     String satification condition: between S and S, there are only X and K-S.
#     # [PAD], CLS, O, S, K-S, X
#     # CLS O X S S k-S X X S S O O
#
#     start_position: 3
#     end_postion: 10
#     keyword_position 7
#     :param predict:
#     :param gold:
#     :return:
#     """
#
#     label_map = {'[PAD]': 0, '[CLS]': 1, 'O': 2, 'S': 3, 'K-S': 4, 'X': 5}
#     predict_start = -1
#     predict_end = -1
#     predict_keyword = -1
#     start_first = True
#     end_first = True
#     keyword_first = True
#     end_S = True
#
#     for idx, label in enumerate(predict):
#
#         if label == label_map['S'] and start_first:
#             predict_start = idx
#             start_first = False
#
#         elif label == label_map['O'] and not start_first and end_first and predict[idx - 1] == label_map['S']:
#             end_first = False
#             predict_end = idx - 1
#
#         elif label == label_map['S'] and idx == len(predict) - 1 and not start_first:
#             if predict_end != -1:
#                 return False
#             predict_end = idx
#
#         elif label == label_map['K-S']:
#             predict_keyword = idx
#             keyword_first = False
#
#         if not keyword_first:
#             if label == label_map['S']:
#                 end_S = False
#     if end_S:
#         if  predict_end != -1:
#             return False
#         predict_end = predict_keyword - 1
#
#     # Remove unsatisfied conditions!
#     for i in range(predict_start,predict_end+1):
#         if predict[i] == label_map['[PAD]'] or predict[i] == label_map['[CLS]'] or predict[i] == label_map['O']:
#             return False
#     for i in range(predict_start):
#         if predict[i] == label_map['K-S']:
#             return False
#
#     for i in range(predict_end+1,len(predict)):
#         if predict[i] == label_map['S'] or label_map == label_map['K-S'] or label_map == label_map['[CLS]']:
#             return False
#
#     # print('start',predict_start)
#     # print('end', predict_end)
#     # print('keyword', predict_keyword)
#
#     if predict_start == -1 or predict_start == -1 or predict_keyword == -1:
#         return False
#
#
#     continue_flag = True
#     predict_keyword_pos = predict_keyword
#     for idx in range(predict_keyword + 1, len(predict)):
#         if predict[idx] == label_map['X'] and continue_flag:
#             predict_keyword_pos = idx
#         else:
#             continue_flag = False
#
#     return (predict_start, predict_end, predict_keyword_pos) == tuple(gold)


def find_ids_from_list(sequence_labels, key_id, pad_id):
    """
    Find all key_id and pad_id for it in list
    :param sequence_labels:
    :param key_id:
    :param pad_id:
    :return:
    """
    cond_list = [idx for idx in range(len(sequence_labels)) if sequence_labels[idx] == key_id]
    addition =[]
    for con_pos in  cond_list:
        next_pos = con_pos +1
        if next_pos > len(sequence_labels) -1:
            break
        while sequence_labels[next_pos] == pad_id:
            addition.append(next_pos)
            next_pos +=1
            if next_pos > len(sequence_labels) - 1:
                break
    cond_list.extend(addition)
    cond_list = sorted(cond_list)
    return cond_list

def get_sequence_acc(predictions, ground_truth, metrics, type='bridge', ques_type='bridge'):
    """
    Get F1 and EM for keywords and constraints on sequence label
    :param predictions:
    :param ground_truth:
    :param type:
    :return:
    """

    # if type == 'comparison':
    #     question_map, direction_map, wh_map, label_map = get_sequence_labels(type)
    #
    # elif type == 'bridge' or type =='intersec':
    #     label_list = get_sequence_labels(type)
    #     label_map = {}
    #     for i in range(len(label_list)):
    #         label_map[label_list[i]] = i
    # if type == 'combined' or type =='intersec' :
    question_map, direction_map, wh_map, label_map = get_sequence_labels(type)

    if type == 'combined':
        b_k_id = label_map['K-S']
        i_k_id = label_map['K']
        c_k_id = label_map['K-COM']
        w_id = label_map['W-COM']
        c_id = label_map['C-COM']

        if ques_type == 'bridge':
            if w_id in predictions or c_id in predictions or i_k_id in predictions or c_k_id in predictions:
                return metrics
        elif ques_type == 'intersec':
            if b_k_id in predictions or c_id in predictions or w_id in predictions or c_k_id in predictions:
                return metrics
        elif ques_type == 'comparison':
            if b_k_id in predictions or i_k_id in predictions:
                return metrics

    prediction_dict = {}
    ground_truth_dict = {}
    keyword_id = None
    constraints_id = None

    if ques_type == 'bridge':
        keyword_id = label_map['K-S']
        constraints_id = label_map['S']
    elif ques_type == 'intersec':
        keyword_id = label_map['K']
        constraints_id = label_map['S1']
    elif ques_type == 'comparison':
        constraints_id = label_map['S-COM']
        keyword_id = label_map['K-COM']
        w_id = label_map['W-COM']
        c_id = label_map['C-COM']


    assert keyword_id is not None and constraints_id is not None
    if ques_type == 'bridge' or ques_type =='intersec':
        gold_keyword_ids = find_ids_from_list(ground_truth, keyword_id, pad_id=label_map['X'])
        predict_keyword_ids = find_ids_from_list(predictions,keyword_id, pad_id = label_map['X'])
        gold_constraints_ids = find_ids_from_list(ground_truth, constraints_id, pad_id=label_map['X'])
        predict_constraints_ids = find_ids_from_list(predictions, constraints_id, pad_id=label_map['X'])

        prediction_dict['keywords'] = predict_keyword_ids
        prediction_dict['constraints'] = predict_constraints_ids
        ground_truth_dict['keywords'] = gold_keyword_ids
        ground_truth_dict['constraints'] = gold_constraints_ids

    elif ques_type == 'comparison':
        gold_keyword_ids = find_ids_from_list(ground_truth, keyword_id, pad_id=label_map['X'])
        predict_keyword_ids = find_ids_from_list(predictions, keyword_id, pad_id=label_map['X'])
        gold_constraints_ids = find_ids_from_list(ground_truth, constraints_id, pad_id=label_map['X'])
        predict_constraints_ids = find_ids_from_list(predictions, constraints_id, pad_id=label_map['X'])

        gold_w_ids = find_ids_from_list(ground_truth, w_id, pad_id=label_map['X'])
        predict_w_ids = find_ids_from_list(predictions, w_id, pad_id=label_map['X'])
        gold_c_ids = find_ids_from_list(ground_truth, c_id, pad_id=label_map['X'])
        predict_c_ids = find_ids_from_list(predictions, c_id, pad_id=label_map['X'])

        gold_keyword_ids = sorted(gold_keyword_ids + gold_w_ids)
        predict_keyword_ids = sorted(predict_keyword_ids + predict_w_ids)
        gold_constraints_ids = sorted(gold_constraints_ids + gold_c_ids)
        predict_constraints_ids = sorted(predict_constraints_ids + predict_c_ids)
        prediction_dict['keywords'] = predict_keyword_ids
        prediction_dict['constraints'] = predict_constraints_ids
        ground_truth_dict['keywords'] = gold_keyword_ids
        ground_truth_dict['constraints'] = gold_constraints_ids
        prediction_dict['wh'] = predict_w_ids
        prediction_dict['comp'] = predict_c_ids
        ground_truth_dict['wh'] = gold_w_ids
        ground_truth_dict['comp'] = gold_c_ids

    metrics = get_keyword_and_constraint_f1_em(prediction_dict, ground_truth_dict, metrics)
    return metrics


def sequence_loss(logits,seq_label, mask=None):
    """
    Compute average cross entropy on every sentence position.
    :param logits: batch_size * seq_len * num_tags
    :param seq_label: batch_size * seq_len
    :param mask: batch_size * seq_len
    :return:
    """
    batch_size,seq_len,num_tags = logits.size()
    if mask is None:
        mask = torch.ones_like(seq_label)
    logits = logits.view(batch_size * seq_len, num_tags)
    seq_label = seq_label.view(-1)
    mask = mask.view(-1)
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = torch.sum(loss_fct(logits, seq_label)* mask.float())/(torch.sum(mask).float()+ 1e-12)
    # predict = torch.argmax(logits.view(batch_size, seq_len, -1), dim =-1)
    return loss


def get_keyword_and_constraint_f1_em(predictions, ground_truths, metrics):
    def update_em_f1(metrics, prediction, ground_truth, type='keywords'):
        tp, fp, fn = 0, 0, 0
        for e in prediction:
            if e in ground_truth:
                tp += 1
            else:
                fp += 1
        for e in ground_truth:
            if e not in prediction:
                fn += 1

        prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        em = 1.0 if fp + fn == 0 else 0.0
        if type == 'keywords':
            metrics['keywords_em'] += em
            metrics['keywords_f1'] += f1
        elif type == 'constraints':
            metrics['constraints_em'] += em
            metrics['constraints_f1'] += f1
        elif type== 'comparison':
            metrics['comparison_em'] += em
            metrics['comparison_f1'] += em
        elif type == 'wh_words':
            metrics['wh_em'] += em
            metrics['wh_f1'] += em
    p_keywords = predictions['keywords']
    p_constraints = predictions['constraints']
    g_keywords = ground_truths['keywords']
    g_constraints = ground_truths['constraints']
    if 'wh' in predictions.keys():
        p_wh = predictions['wh']
        p_comp = predictions['comp']
        g_wh = ground_truths['wh']
        g_comp = ground_truths['comp']
        update_em_f1(metrics, p_comp, g_comp, type='comparison')
        update_em_f1(metrics, p_wh, g_wh,type='wh_words')
    update_em_f1(metrics, p_keywords, g_keywords,type='keywords')
    update_em_f1(metrics,  p_constraints, g_constraints,type='constraints')
    return metrics

def compute_span_distance(span1,span2):
    """
    Computer two interval distance. if span1 and span2 have common elements, distance is 0. if span1 is to the left of span2,
    the distance is the left of interval 2 minus the right of interval 1. Otherwise, similar to second condition.
    :param span1:
    :param span2:
    :return:
    """
    span1_start = span1[0]
    span1_end = span1[1]

    span2_start = span2[0]
    span2_end = span2[1]

    len_1 = span1_end - span1_start
    len_2 = span2_end - span2_start

    if len_1 >= len_2:
        max_len_start = span1_start
        max_len_end = span1_end
        min_len_start = span2_start
        min_len_end = span2_end

    else:
        max_len_start = span2_start
        max_len_end = span2_end
        min_len_start = span1_start
        min_len_end = span1_end

    if min_len_end < max_len_start:
        return max_len_start - min_len_end
    elif max_len_end < min_len_start:
        return min_len_start - max_len_end
    else:
        return 0

def recognize_datetime_re(match_strings):
    """Recognize datetime in match strings. Match accoriding to mdy, dmy, y, epoch,  """
    from datetime import datetime
    month_string = 'january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september\
                   |sept|october|oct|november|nov december|dec'

    month_dict = {'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
     'may': 5, 'june': 6, 'jun': 6,
     'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sept': 9, 'october': 10, 'oct': 10,
     'november': 11,
     'nov': 11, 'december': 12, 'dec': 12}
    num_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
                'ten': 10, 'eleven': 11,
                'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
                'eighteen': 18, 'nineteen': 19,
                'twenty': 20}
    flag = False
    num_list = []
    for key in num_dict:
        num_list.append(key)
    num_string = '|'.join(num_list)

    match_strings = match_strings.lower()

    # m,d,y
    mdy_patern = re.compile(r'(%s)\s+(\d{1,2})\S*\s+(\d{4})'% month_string)

    # d,m,y
    dmy_patern = re.compile(r'(\d{1,2}|\d{0})\S*\s+(%s)\S*\s+(\d{4})|(\d{1,2})(?:-|/)(\d{1,2})(?:-|/)(\d{4})'%month_string)

    # epoch
    epoch_pattern = re.compile(r'(?<=\s)([0-9]{2})(?:s)')

    # centuries
    centuries_pattern = re.compile(r'(?<=\s)([0-9]+)(?:th)\s(?:centuries|century)')

    # year
    year_pattern = re.compile(r'(?<=\s)[(]?([0-9]{4})')

    # year span
    span_year = re.compile(r'(\d{4})-(\d{4})')

    # num span
    span_num = re.compile(r'(?<!\d)(\d{1,3})-(\d{1,3})(?!\d)')

    # thousand,
    thousand_num = re.compile(r'(\d{1,3})\sthousand')
    # million
    million_num = re.compile(r'(\d{1,3})\smillion')
    # billion
    # billion_num = re.compile(r'(\d{1,3})\sbillion')

    # 4-6 natural number adding Thousandth symbol
    four_six_num = re.compile(r'(\d{1,3},\d{3})')

    # 6-9 natural number adding Thousandth symbol
    six_nine_num = re.compile(r'(\d{1,3},\d{3},\d{3})')

    # num
    num_pattern = re.compile(r'(?<=\s)([0-9]+|%s)(?=\s)' % num_string)

    mdy_results = list(re.finditer(mdy_patern, match_strings))

    dmy_results = list(re.finditer(dmy_patern, match_strings))

    epoch_results = list(re.finditer(epoch_pattern, match_strings))

    centuries_results = list(re.finditer(centuries_pattern, match_strings))
    # print(centuries_results)
    year_results = list(re.finditer(year_pattern, match_strings))

    span_year_results = list(re.finditer(span_year, match_strings))
    # print('span_year', span_year_results)
    span_num_results = list(re.finditer(span_num, match_strings))

    thousand_num_results = list(re.finditer(thousand_num, match_strings))
    # print('thousand',thousand_num_results)
    million_num_results = list(re.finditer(million_num, match_strings))

    # billion_num_results = list(re.finditer(billion_num, match_strings))

    four_six_num_results = list(re.finditer(four_six_num, match_strings))

    six_nine_num_results = list(re.finditer(six_nine_num, match_strings))

    num_results = list(re.finditer(num_pattern, match_strings))



    data_list = {'date_time': [], 'epoch': [], 'num':[]}

    year_spans = []
    if len(mdy_results) > 0:
        for result in mdy_results:
            month = int(month_dict[result[1]]) if len(result[1])>0 else 1
            day = int(result[2]) if len(result[2]) > 0 else 1
            year = int(result[3]) if len(result[3]) > 0 else 1
            span = result.span()
            year_spans.append(span)
            flag = True
            try:
                data_list['date_time'].append([span,datetime(year, month, day), result.group()])
            except Exception:
                print(match_strings)

    if len(dmy_results) > 0:
        for result in dmy_results:
            if result.groups()[0] is not None:
                day = int(result[1]) if len(result[1]) > 0 else 1
                month = int(month_dict[result[2]]) if len(result[2]) > 0 else 1
                year = int(result[3]) if len(result[3]) > 0 else 1
            elif result.groups()[3] is not None:
                day = int(result[4]) if len(result[4]) > 0 else 1
                month = int(result[5]) if len(result[5]) > 0 else 1
                year = int(result[6]) if len(result[6]) > 0 else 1

                if month > 12:
                    temp = day
                    day = month
                    month = temp
            else:
                NotImplementedError
            span = result.span()
            year_spans.append(span)
            flag = True
            try:
                data_list['date_time'].append([span, datetime(year, month, day), result.group()])
            except Exception:
                print(match_strings)

    if len(epoch_results) > 0:
        for result in epoch_results:
            year = '19'+result[1]
            span = result.span()
            year_spans.append(span)
            flag = True
            try:
                data_list['date_time'].append([span, datetime(int(year), 1, 1),result.group()])
            except Exception:
                print('epoch results')
                print(match_strings)

    if len(centuries_results) > 0:
        for result in centuries_results:
            year = result[1] + '00'
            span = result.span()
            year_spans.append(span)
            try:
                data_list['date_time'].append([span, datetime(int(year), 1, 1), result.group()])
            except Exception:
                print('centuries_results')
                print(match_strings)



    if len(span_year_results) > 0:
        for result in span_year_results:
            y_span = result.span()
            flag_list = []
            for span in year_spans:
                if y_span[1] < span[0] or span[1] < y_span[0]:
                    flag_list.append(True)
                else:
                    flag_list.append(False)

            if all(flag_list):
                if result.groups()[0] is not None:
                    year1 = int(result[1])
                    year2 = int(result[2])
                    y_span = result.span()
                    y1_span = result.span(1)
                    y2_span = result.span(2)
                    year_spans.append(y_span)
                    data_list['date_time'].append([y1_span, datetime(year1, 1, 1),result.group(1)])
                    data_list['date_time'].append([y2_span, datetime(year2, 1, 1), result.group(2)])

    if len(year_results) > 0:
        for result in year_results:
            year = int(result[1])
            y_span = result.span()
            flag_list = []

            for span in year_spans:
                if y_span[1] < span[0] or span[1] < y_span[0]:
                    flag_list.append(True)
                else:
                    flag_list.append(False)
            if all(flag_list):
                year_spans.append(y_span)
                flag = True
                data_list['date_time'].append([y_span, datetime(year, 1, 1), result.group(1)])

    if len(span_num_results) > 0:
        for result in span_num_results:
            num_span_span = result.span()
            flag_list = []
            for span in year_spans:
                if num_span_span[1] < span[0] or span[1] < num_span_span[0]:
                    flag_list.append(True)
                else:
                    flag_list.append(False)
            if all(flag_list):
                if result.groups()[0] is not None:
                    num1 = int(result[1])
                    num2 = int(result[2])
                    num_span = result.span()
                    num1_span = result.span(1)
                    num2_span = result.span(2)
                    year_spans.append(num_span)
                    data_list['num'].append([num1_span, int(num1), result.group(1)])
                    data_list['num'].append([num2_span, int(num2), result.group(2)])

    if len(thousand_num_results) > 0:
        for result in thousand_num_results:
            num_span = result.span()
            flag_list = []
            for span in year_spans:
                if num_span[1] < span[0] or span[1] < num_span[0]:
                    flag_list.append(True)
                else:
                    flag_list.append(False)
            if all(flag_list):
                if result.groups()[0] is not None:
                    num1 = int(result[1])
                    num_span = result.span()
                    year_spans.append(num_span)
                    data_list['num'].append([num_span, int(num1)*1000, result.group()])

    if len(million_num_results) > 0:
        for result in million_num_results:
            num_span = result.span()
            flag_list = []
            for span in year_spans:
                if num_span[1] < span[0] or span[1] < num_span[0]:
                    flag_list.append(True)
                else:
                    flag_list.append(False)
            if all(flag_list):
                if result.groups()[0] is not None:
                    num1 = int(result[1])
                    num_span = result.span()
                    year_spans.append(num_span)
                    data_list['num'].append([num_span, int(num1)*1000000, result.group()])

    if len(six_nine_num_results) > 0:
        for result in six_nine_num_results:
            num_span = result.span()
            num = int(result[1].replace(',', ''))
            year_spans.append(num_span)
            data_list['num'].append([num_span, int(num), result.group()])


    if len(four_six_num_results) > 0:
        for result in four_six_num_results:
            num_span = result.span()
            num = int(result[1].replace(',',''))
            year_spans.append(num_span)
            data_list['num'].append([num_span, int(num), result.group()])



    if len(num_results) > 0:
        for result in num_results:
            num_span = result.span()
            flag_list = []
            for span in year_spans:
                if num_span[1] < span[0] or span[1] < num_span[0]:
                    flag_list.append(True)
                else:
                    flag_list.append(False)

            if all(flag_list):
                if result[1] in num_dict.keys():
                    data_list['num'].append([num_span, num_dict[result[1]], result.group()])
                else:
                    data_list['num'].append([num_span, int(result[1]), result.group()])

    return data_list

def recognize_datetime_re2(match_strings):
    """Recognize datetime in match strings. Match accoriding to mdy, dmy, y, epoch,  """
    from datetime import datetime
    month_string = 'january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september\
                   |sept|october|oct|november|nov december|dec'

    month_dict = {'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
     'may': 5, 'june': 6, 'jun': 6,
     'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sept': 9, 'october': 10, 'oct': 10,
     'november': 11,
     'nov': 11, 'december': 12, 'dec': 12}
    num_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
                'ten': 10, 'eleven': 11,
                'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
                'eighteen': 18, 'nineteen': 19,
                'twenty': 20}
    flag = False
    num_list = []
    for key in num_dict:
        num_list.append(key)
    num_string = '|'.join(num_list)

    match_strings = match_strings.lower()

    # m,d,y
    mdy_patern = re.compile(r'(%s)\s+(\d{2})\S*\s+(\d{4})' % month_string)

    # d,m,y
    dmy_patern = re.compile(r'(\d{2}|\d{0})\S*\s+(%s)\S*\s+(\d{4})|(\d)(?:-|/)(\d+)(?:-|/)(\d+)'%month_string)

    # epoch
    epoch_pattern = re.compile(r'(?<=\s)([0-9]+)(?:s|th)(?=\s)')

    # year
    year_pattern = re.compile(r'(?<=\s)([0-9]{4})')

    # num
    num_pattern = re.compile(r'(?<=\s)([0-9]+|%s)(?=\s)' % num_string)

    # year span
    span_year = re.compile(r'(\d{4})-(\d{4})')


    mdy_results = list(re.finditer(mdy_patern, match_strings))

    dmy_results = list(re.finditer(dmy_patern, match_strings))

    epoch_results = list(re.finditer(epoch_pattern, match_strings))

    year_results = list(re.finditer(year_pattern, match_strings))

    span_results = list(re.finditer(span_year, match_strings))

    num_results = list(re.finditer(num_pattern, match_strings))

    data_list = {'date_time': [], 'epoch': [], 'num':[]}

    year_spans = []
    if len(mdy_results) > 0:
        for result in mdy_results:
            month = int(month_dict[result[1]]) if len(result[1])>0 else 1
            day = int(result[2]) if len(result[2]) > 0 else 1
            year = int(result[3]) if len(result[3]) > 0 else 1
            span = result.span()
            year_spans.append(span)
            flag = True
            try:
                data_list['date_time'].append([span, datetime(year, month, day)])
            except Exception:
                print(match_strings)

    if len(dmy_results) > 0:
        for result in dmy_results:
            if result.groups()[0] is not None:
                day = int(result[1]) if len(result[1]) > 0 else 1
                month = int(month_dict[result[2]]) if len(result[2]) > 0 else 1
                year = int(result[3]) if len(result[3]) > 0 else 1
            elif result.groups()[3] is not None:
                day = int(result[4]) if len(result[4]) > 0 else 1
                month = int(result[5]) if len(result[5]) > 0 else 1
                year = int(result[6]) if len(result[6]) > 0 else 1

                if month > 12:
                    temp = day
                    day = month
                    month = temp
            else:
                NotImplementedError
            span = result.span()
            year_spans.append(span)
            flag = True
            try:
                data_list['date_time'].append([span, datetime(year, month, day)])
            except Exception:
                print(match_strings)

    if len(epoch_results) > 0:
        for result in epoch_results:
            year = result[1]
            span = result.span()
            year_spans.append(span)
            flag = True
            data_list['date_time'].append([span, year])

    if len(year_results) > 0:
        for result in year_results:
            year = int(result[1])
            y_span = result.span()
            flag_list = []

            for span in year_spans:
                if y_span[1] < span[0] or span[1] < y_span[0]:
                    flag_list.append(True)
                else:
                    flag_list.append(False)
            if all(flag_list):
                year_spans.append(y_span)
                flag = True
                data_list['date_time'].append([y_span, datetime(year, 1, 1)])

    if len(span_results) > 0:
        for result in span_results:
            year1 = int(result[1])
            year2 = int(result[2])
            y_span = result.span()
            year_spans.append(y_span)
            data_list['date_time'].append([y_span, datetime(year1, 1, 1)])
            data_list['date_time'].append([y_span, datetime(year2, 1, 1)])

    if len(num_results) > 0:
        for result in num_results:
            num_span = result.span()
            flag_list = []
            for span in year_spans:
                if num_span[1] < span[0] or span[1] < num_span[0]:
                    flag_list.append(True)
                else:
                    flag_list.append(False)

            if all(flag_list):
                if result.groups()[0] is not None:
                    flag = True
                    if result[1] not in num_dict:
                        changed = result[1]
                        if ',' in result[1]:
                            changed = result[1].replace(',', '')

                        data_list['num'].append([num_span, int(changed)])
                    else:
                        data_list['num'].append([num_span, num_dict[result[1]]])

    return data_list

def recognize_datetime(tokens, month_dict):
    """recognize datetime(month,day,year) type if exists otherwise return none"""
    # tokens = strings.split()
    import collections
    month_idx = -1
    month = -1
    day_idx = -1
    year_idx = -1
    date_time = collections.namedtuple('date_time', ['month', 'day', 'year'])
    for idx, token in enumerate(tokens):
        if token.lower().replace('.', '') in month_dict.keys():
            month_idx = idx
            month = month_dict[token.lower()]
            break

    if month_idx != -1:
        if month_idx +1 < len(tokens):
            if any([True if char in [str(i) for i in list(range(10))] else False for char in tokens[month_idx + 1]]):
                if len(tokens[month_idx + 1]) < 3:
                    day_idx = month_idx + 1
                    if month_idx +2 < len(tokens):
                        if any([True if char in [str(i) for i in list(range(10))] else False for char in
                                tokens[month_idx + 2]]):
                            year_idx = month_idx + 2
                    else:
                        return None
                elif len(tokens[month_idx + 1]) > 2:
                    year_idx = month_idx + 1
                    if any([True if char in [str(i) for i in list(range(10))] else False for char in
                            tokens[month_idx - 1]]):
                        day_idx = month_idx - 1

                try:
                    date = date_time(month=month,
                                 day=int(tokens[day_idx]) if day_idx != -1  else -1,
                                 year=int(tokens[year_idx]) if year_idx != -1 else -1)
                except ValueError:
                    return None

                return  date

    return None
#
def recognize_number_from_string(input_string):
    pass
    """return three data types: datetime, number, number span. if input_string is not these types, return none
    span: ('span', [['num', ] ,['num] ])
    number: ('number', )
    date_time ('date_time',  date_time(month,day,year))"""
    month_dict = {'january': 1, 'jan': 1, 'february ': 2, 'feb': 2, 'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
                  'may': 5, 'june': 6, 'jun': 6,
                  'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sept': 9, 'october': 10, 'oct': 10,
                  'november': 11,
                  'nov': 11, 'december': 12, 'dec': 12}
    week_dict = {'monday': 1, 'mon': 1, 'tuesday': 2, 'tue': 2, 'wednesday': 3, 'wed': 3, 'thursday': 4, 'thu': 4,
                 'friday': 5, 'fri': 5,
                 'saturday': 6, 'sat': 6, 'sunday': 7, 'sun': 7}
    num_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
                'ten': 10, 'eleven': 11,
                'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
                'eighteen': 18, 'nineteen': 19,
                'twenty': 20}
    data_type = None
    number_set= [str(i) for i in list(range(10000))]
    input_string = " ".join(input_string.split())
    tokens = re.split(r"[\s]", " ".join(re.split(r'(–|\-|,)', input_string.strip().replace('.','').replace(',','').replace("(","").replace(")",""))))
    span_split_index = -1
    num_flag = False
    datetime_flag = False
    num_list = []

    for idx, token in enumerate(tokens):
        if token in ['to', '-','–']:
            span_split_index = idx

        if token in month_dict.keys():
            datetime_flag = True

        if token in number_set or token in num_dict:
            num_flag = True

    if span_split_index != -1 and span_split_index+1 < len(tokens):
        if tokens[span_split_index-1] in number_set and tokens[span_split_index+1] in number_set:
            first_part = tokens[:span_split_index]
            first_part_list = []
            first_datetime = None
            second_part_list = []
            second_datetime = None

            second_part = tokens[span_split_index:]
            for token in first_part:
                if token in month_dict.keys():
                    first_datetime = recognize_datetime(first_part,month_dict)
                    break
                elif token in number_set:
                    first_part_list.append(int(token))
                elif token in num_dict.keys():
                    first_part_list.append(num_dict[token])

            for token in second_part:
                if token in month_dict.keys():
                    second_datetime = recognize_datetime(second_part, month_dict)
                    break
                elif token in number_set:
                    second_part_list.append(int(token))
                elif token in num_dict.keys():
                    second_part_list.append(num_dict[token])

            if first_datetime is not None and second_datetime is not None:
                return 'span', [('datetime',first_datetime), ('datetime', second_datetime)]
            elif len(first_part_list) > 0 and second_datetime is not None:
                return 'span', [('number', first_part_list), ('datetime', second_datetime)]
            elif first_datetime is not None and len(second_part_list)>0:
                return 'span', [('datetime',first_datetime), ['number', second_datetime]]
            elif len(first_part_list) > 0  and len(second_part_list) > 0:
                return 'span',[('number', first_part_list),('number', second_part_list)]
            else:
                return None,None
        else:
            return None, None


    elif datetime_flag:
        datetime = recognize_datetime(tokens, month_dict)
        if datetime is not None:
            return 'datetime', datetime
        else:
            return None,None

    elif num_flag:
        for idx, token in enumerate(tokens):
            if token in number_set:
                num_list.append(int(token))
            elif token in num_dict.keys():
                num_list.append(num_dict[token])
        if len(num_list) > 0:
            return 'number',num_list
        else:
            return None,None
    else:
        return None,None


def random_exchange_specified_two_label(label_list, id1, id2, random_exchange=False):
    id1_list = []
    id2_list = []
    for idx, label in enumerate(label_list):
        if id1==label:
            id1_list.append(idx)
        if id2 == label:
            id2_list.append(idx)

    import random
    rand_number = random.uniform(0,1)
    if random_exchange and rand_number > 0.5:
        for index in id1_list:
            label_list[index] = id2
        for index in id2_list:
            label_list[index] = id1
    return label_list


def flatten_net_list(net_list):
    flatten_list = []
    all_True = True
    for element in net_list:
        if isinstance(element, list):
            all_True = False
            flatten_list.extend(flatten_net_list(element))
    if all_True:
        return [net_list]
    return flatten_list


def recover_tokenize_results(lower_tokenize_words, orig_tokenize_words, seq_label):
    """recover tokenize tokens according orig words, commas,period follow next word label, """
    assert len(lower_tokenize_words) == len(seq_label)
    if ''.join(lower_tokenize_words) == ' '.join(orig_tokenize_words).lower():
        return lower_tokenize_words, seq_label

    else:
        lower_start = 0
        orig_start = 0
        new_seq_label = []
        new_lower_tokenize_words = []
        previous_equal = True
        append_flag = False

        current_word = ''
        while new_lower_tokenize_words != orig_tokenize_words and lower_start < len(lower_tokenize_words) \
                and orig_start < len(orig_tokenize_words):

            if lower_tokenize_words[lower_start] == orig_tokenize_words[orig_start].lower() and previous_equal:
                new_lower_tokenize_words.append(lower_tokenize_words[lower_start])
                new_seq_label.append(seq_label[lower_start])
                current_word = ''
                orig_start += 1
                lower_start += 1

            elif lower_tokenize_words[lower_start] != orig_tokenize_words[orig_start].lower() and previous_equal:
                current_word = lower_tokenize_words[lower_start]
                lower_start += 1
                append_flag = True
                previous_equal = False


            elif not previous_equal and append_flag:
                current_word += lower_tokenize_words[lower_start]

                if current_word == orig_tokenize_words[orig_start].lower():
                    new_lower_tokenize_words.append(current_word)

                    new_seq_label.append(seq_label[len(new_lower_tokenize_words)])
                    append_flag = False
                    previous_equal = True
                    current_word = ''
                    orig_start += 1
                    lower_start += 1
                else:
                    append_flag = True
                    previous_equal = False
                    lower_start += 1

        if ' '.join(new_lower_tokenize_words) == ' '.join(orig_tokenize_words).lower():
            assert len(orig_tokenize_words) == len(new_seq_label)
            return True, orig_tokenize_words, new_seq_label
        else:
            return False, lower_tokenize_words, seq_label


class weighted_biattention(nn.Module):
    def __init__(self,input_dim, hidden_dim, dropout=0.9):
        """Biattention from Seo"""
        super(weighted_biattention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_linear_1 = nn.Linear(input_dim, 1, bias=False)
        self.memory_linear_1 = nn.Linear(input_dim, 1, bias=False)
        self.input_linear_2 = nn.Linear(input_dim,hidden_dim)
        self.memory_linear_2 = nn.Linear(input_dim, hidden_dim)
        self.memory_linear_output = nn.Linear(hidden_dim*2, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim*4, hidden_dim)
        self.dot_scale = np.sqrt(input_dim)


    def forward(self, h, u, h_mask=None, u_mask=None, u_weights=None):
        """

        :param h: batch_size * h_len * emb_dim
        :param u: batch_size * u_len * emb_dim
        :param h_mask: batch_size * h_len
        :param u_mask: batch_size * u_len
        :param u_weights: batch_size * sent_len
        :return:
        """

        # h = h.to(dtype=next(self.parameters()).dtype)
        # u = u.to(dtype=next(self.parameters()).dtype)
        h_len = h.size(1)
        batch_size = h.size(0)
        u_len = u.size(1)
        u_mask = torch.unsqueeze(u_mask, 1)
        u_mask_aug = u_mask.repeat([1, h_len, 1])
        h_mask = torch.unsqueeze(h_mask, 2)
        h_mask_aug = h_mask.repeat([1, 1, u_len])
        hu_mask = u_mask_aug & h_mask_aug
        hu_mask = hu_mask.to(dtype=next(self.parameters()).dtype)
        h = self.dropout(h)
        u = self.dropout(u)
        h_dot = self.input_linear_1(h)
        u_dot = self.memory_linear_1(u).view(batch_size, 1, u_len)
        cross_dot = torch.matmul(h, u.permute(0, 2, 1).contiguous()) / self.dot_scale
        att = h_dot + u_dot + cross_dot
        att = att - 10000.0 * (1-hu_mask) # batch_size * h_len * u_len
        h = self.input_linear_2(h)
        u = self.memory_linear_2(u)

        weight_one = nn.Softmax(dim=-1)(att) # batch_size * h_len * u_len
        output_one = torch.matmul(weight_one, u) # batch_size * h_len * dim
        weight_two = nn.Softmax(dim=-1)(torch.max(att, dim=-1)[0]).view(batch_size, 1, h_len) # batch_size * 1 * h_len
        output_two = torch.matmul(weight_two, h) # batch_size * 1*dim

        memory = self.memory_linear_output(torch.cat([h, h*output_two], dim=-1))
        output = self.output_linear(torch.cat([h, output_one, h * output_one, output_two * output_one],dim=-1))
        return output, output_two.squeeze(dim=1)

def exp_mask(logits, mask=None ,dtype=None):
    if mask is not None:
        assert logits.size() == mask.size()
        if dtype is not None:
            mask = mask.to(dtype)
        logits = logits -10000.0 * (1- mask)
        return logits
    return logits


def combine_tensors_with_mask_and_max_length(s1, s2, s1_lens, s2_lens, max_len):
    """

    :param s1: batch * s2_len * dim
    :param s2: batch * s1_len * dim
    :param s1_mask: batch * s1_len
    :param s2_mask: batch * s2_len
    :param max_s1_lens: int, default:240
    :param max_s2_lens: int, default:240
    :return:
    """
    N = s1.size(0)
    shape = (max_len, s1.size(2))
    batch_list = []
    batch_length = []
    for i in range(N):
        new_tensor = s1.data.new(*shape).fill_(0)
        one_tensor = s1[i][:s1_lens[i]]
        two_tensor = s2[i][:s2_lens[i]]
        new_tensor[:(s1_lens[i]+s2_lens[i])] = torch.cat([one_tensor, two_tensor], dim=0)
        batch_list.append(new_tensor)
        batch_length.append((s1_lens[i]+s2_lens[i]))

    return torch.stack(batch_list, 0), batch_length


def copy_file(source, target, type='file'):
    """
    copy source file /dirs to target file/dirs
    :param source:
    :param target:
    :param type: file or dir
    :return:
    """
    assert type == 'file' or type == 'dir'

    if os.path.exists(target):
        print('{} exists, remove it!'.format(target))
        shutil.rmtree(target)

    if type == 'file':
        shutil.copyfile(source, target)
        print('copy file {} to {}'.format(source, target))
    elif type == 'dir':
        shutil.copytree(source, target)
        print('copy dir {} to {}'.format(source, target))


def sequence_mask(lens, max_len):
    """Generate a mask tensor len(max_len) * max_len where  """
    if isinstance(lens, torch.Tensor):
        lens = lens.detach().cpu().tolist()
    mask_tensor = torch.arange(max_len)[None, :] < torch.tensor(lens)[:, None]
    return mask_tensor


def generate_position_metrics(subq_f1_list, ques_f1_list):
    """
    Generating a metric according to subques f1 and question f1, that decides whether the two elements in 1 dim of combined subq_f1_logits and ques_f1_logits  need to exchange

    :param subq_f1_list: batch_size
    :param ques_f1_list: batch_size
    :return:
    """

    subq_f1 = np.array(subq_f1_list)[:, np.newaxis]
    ques_f1 = np.array(ques_f1_list)[:, np.newaxis]
    combined_f1_l = np.concatenate((subq_f1, ques_f1), axis=1)
    combined_f1_r = np.concatenate((ques_f1, subq_f1), axis=1)
    position = combined_f1_l > combined_f1_r
    return torch.from_numpy(position).long()


def exchange_position_by_position_metrics(subq_f1_list, ques_f1_list, subq_logits, ques_logits):
    """
    ```
    subq_f1_list = [0.5, 0.9, 0.8]
    ques_f1_list = [0.8, 0.6, 0.7]

    ques_logits = torch.randn(3,2)
    subq_logits = torch.randn(3,2)


    exchange_position_by_position_metrics(subq_f1_list , ques_f1_list,subq_logits, ques_logits)

    ```
    According to position_metrics, exchaging first dimension  of concat subq_logits and ques_logits.
    combined_tensors : batch_size * 2 * lens

    :param subq_f1_list: batch_size
    :param ques_f1_list: batch_size
    :param ques_logits: batch_size * lens
    :param subq_logits: batch_size * lens
    :return:
    """
    assert len(subq_f1_list) == len(ques_f1_list), 'subq_f1 len {}, \
    ques_f1_list len {} are not equal.'.format(len(subq_f1_list), len(ques_f1_list))
    position_metrics = generate_position_metrics(subq_f1_list, ques_f1_list).cuda()
    max_len = subq_logits.size(1)
    position_metrics = position_metrics.unsqueeze(-1).repeat(1, 1, max_len)
    ques_subq_logits = torch.cat([ques_logits.unsqueeze(1), subq_logits.unsqueeze(1)], 1)
    new_ques_subq_logits = torch.gather(ques_subq_logits, 1, position_metrics)
    # N = ques_subq_logits.size(0)
    # batch_list = []
    # for i in range(N):
    #     indx = position_metrics[i]
    #
    #     ques_sub = ques_subq_logits[i]
    #     batch_list.append(torch.index_select(ques_sub, 0, indx).unsqueeze(0))
    return new_ques_subq_logits

def weigth_decay(step,t_total, warmup_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return max(0.0, float(t_total -step)/ float(max(1.0, t_total -warmup_steps)))


def set_logger(save_to_file=None, name=None,global_rank=-1):
    if name is not None:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if global_rank in [-1, 0] else logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    # using StreamHandler to help output info to screen
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if global_rank in [-1, 0] else logging.WARN)
    ch.setFormatter(formatter)
    logger.propagate = False
    # add Streamhandler to logger
    logger.addHandler(ch)
    if save_to_file is not None:
        # add filehander to logger
        fh = logging.FileHandler(os.path.join(save_to_file, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def process_file_dirs(args, file_marks=None, debug=False):
    """Automatic generate checkpoints dirs, predictions dirs, tensorboard dirs and logger dirs,
    given output dirs and write dirs! file_marks is smilar list consisting of tuple like [('ht',args.hopot), ('bri, args.weakly_supervise)]
    """
    def cat_args_to_file_mark(args):

        file_mark = ''
        for idx, (name, value) in enumerate(args):
            if idx != 0:
                file_mark += '_' + name + '_' + str(value)
            else:
                file_mark += name + '_' + str(value)
        return file_mark

    def create_dirs(dir_list):
        """Create dir while dir does not exists!"""
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)

    if file_marks is not None:
        args.file_marks = cat_args_to_file_mark(file_marks)
    else:
        args.file_marks = ''

    if args.debug:
        args.file_marks = 'debug_' + args.file_marks

    if args.write_dir is None:
        args.write_dir = args.output_dir

    args.tb_source_dir = os.path.join(args.output_dir, 'tb', "tb_{}".format(args.model_descri.split('|')[0]))
    args.logger_dir = os.path.join(args.output_dir, 'logger', 'logger_{}'.format(args.model_descri.split('|')[0]))
    args.write_dir = os.path.join(args.write_dir, 'predictions',
                                  'predictions_{}'.format(args.model_descri.split('|')[0]))
    args.output_dir = os.path.join(args.output_dir,'checkpoints',
                                   'checkpoint_{}'.format(args.model_descri.split('|')[0]))
    create_dirs([args.tb_source_dir, args.logger_dir, args.output_dir, args.write_dir])


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x)) # same code
    return e_x / e_x.sum(axis=axis, keepdims=True)

if __name__ == '__main__':
    # inputs = torch.randn(2,3,2)
    # parent_positions = torch.tensor([[0,1,1], [0,2,1]])
    # sentence_deps = torch.tensor([[0, 2, 3], [0, 1, 2]])
    # attention_mask = torch.tensor([[1,1,0],[1,0,0]])
    # gcn = gcn_layer_v2(input_size=2, dep_count= 10,gcn_size=5)
    # print(gcn(inputs, parent_positions,sentence_deps,attention_mask))

    """Evaluate keyword and constraints """
    predictions = {}
    ground_truth = {}
    predictions['keywords'] =[1,2,3,4]
    predictions['constraints'] = [7,8,9,10]
    ground_truth['keywords'] = [1,2,3,5]
    ground_truth['constraints'] = [7,8,9,11]
    metrics = {'keywords_em': 0, 'keywords_f1': 0, 'constraints_em': 0, 'constraints_f1': 0, 'acc': 0}

    # print(get_keyword_and_constraint_f1_em(predictions, ground_truth, metrics))
    "['[PAD]', '[CLS]', 'O', 'S', 'K-S', 'X']"
    predictions = [1,3,3,3,4,5,5,3,3,4,2,2,2]
    golds = [1,3,3,3,4,5,5,3,3,3,2,3,3]
    print(get_sequence_acc(predictions, golds, metrics))


