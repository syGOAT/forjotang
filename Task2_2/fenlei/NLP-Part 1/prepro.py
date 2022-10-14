import torch
import json
import os
import sys

sys.path.append('.')
from util import set_logger
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from finqa_utils import *
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaConfig, RobertaForQuestionAnswering
from config import set_args, set_environ
from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer,
                          XLMConfig, XLMForQuestionAnswering,
                          XLMTokenizer, XLNetConfig,
                          XLNetForQuestionAnswering,
                          XLNetTokenizer,
                          RobertaConfig,
                          RobertaForQuestionAnswering,
                          RobertaTokenizer)

logger = set_logger()

concat_prog_list = []  #存放你之前统计得到的所有concat_op类别

all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max",
           "table_min", "table_sum", "table_average"]


class InputExample(object):
    def __init__(self, unique_id, q_id, text_a, text_b=None, label=None):
        self.q_id = q_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, qid, example_index, tokens, input_ids, input_mask, segment_ids, concat_prog_ids, multi_prog_ids,
                 concat_prog, multi_prog):
        self.qid = qid
        self.example_index = example_index
        self.tokens = tokens,
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.concat_prog_ids = concat_prog_ids
        self.multi_prog_ids = multi_prog_ids
        self.concat_prog = concat_prog
        self.multi_prog = multi_prog


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class MathQAExample:
    def __init__(self, qid, original_question, question_tokens, concat_prog, multi_prog):
        self.qid = qid
        self.original_question = original_question
        self.question_tokens = question_tokens
        self.concat_prog = concat_prog
        self.multi_prog = multi_prog


class CustomDataLoader:
    def __init__(self, args, tokenizer, is_training=True):
        self.args = args
        self.tokenizer = tokenizer
        self.is_training = is_training

        if is_training:
            self.input_file = args.train_file
        else:
            self.input_file = args.predict_file

    def read_examples(self):
        """ 得到模型的输入和输出 的自然语言表示"""
        with open(self.input_file, 'r') as f:
            input_data = json.load(f)

        examples = []
        for entry in tqdm(input_data):
            # 处理输入
            question = entry["qa"]["question"]
            this_id = entry["id"]
            ## 得到回答问题的上下文
            context = ""
            for ind, each_sent in entry["qa"]["model_input"]:
                context += each_sent
                context += " "
            context = context.strip()
            context = context.replace(". . . . . .", "")
            context = context.replace("* * * * * *", "")
            ## 拼接 问题和 上下文
            original_question = question + " " + self.tokenizer.sep_token + " " + context.strip()
            original_question_tokens = original_question.split(' ')
            question_tokens = []

            # tokenizer
            for i, tok in enumerate(original_question_tokens):
                tok_proc = tokenize(self.tokenizer, tok)
                question_tokens.extend(tok_proc)


            r"""
            
            处理输出, 自行完成concat_prog的处理和生成。
            
            example：  
                如果program为 'divide(100, 100), divide(3.8, #0)'，则它对应的concat_prog为'divide0-divide1'，该部分需
                要各位同学自行完成从program到concat_prog的变换算法。
            
            注：同学可以使用python自带的相关字符串、数组操作方法自行设计算法，也可以尝试参考finqa_utils文件，尝试使用其中的一些论文源码函数，
                可以极大简化你的代码哦。
            
            
            """
            original_program = entry['qa']['program']
            import re
            ops = re.findall(', (.*?)[(].*?[)]', ', '+original_program)
            concat_op = ''
            i = 0
            for op in ops:
                concat_op += op + str(i) + '_'
                i += 1
            concat_prog = concat_op.rstrip('_')



            """
            *
            *
            *
            concat_prog处理部分
            *
            *
            *
            """




            concat_prog = ''

            multi_prog = []  # 多分类标签，本项目暂时不需要用，置空即可

            #             print(concat_prog)
            examples.append(MathQAExample(
                qid=this_id,
                original_question=original_question,
                question_tokens=question_tokens,
                concat_prog=concat_prog,
                multi_prog=multi_prog))
        return examples

    def convert_examples_to_features(self):
        features = []
        examples = self.read_examples()
        #         print(len(examples))
        max_seq_length = self.args.max_seq_length
        for example_index, example in enumerate(examples):
            question_tokens = example.question_tokens
            if len(question_tokens) > max_seq_length:
                question_tokens = question_tokens[:max_seq_length - 2]
            tokens = [self.tokenizer.cls_token] + question_tokens + [self.tokenizer.sep_token]
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids.extend(padding)
            input_mask.extend(padding)
            segment_ids.extend(padding)

            # 将输出变为ids
            assert len(input_ids) == max_seq_length
            concat_prog = example.concat_prog
            multi_prog = example.multi_prog
            self.args.multi_labels = len(concat_prog_list)

            multi_prog_ids = [0] * len(all_ops)
            if concat_prog in concat_prog_list:
                #                 print("pass")
                concat_prog_ids = concat_prog_list.index(concat_prog)
                for prog in multi_prog:
                    multi_prog_ids[all_ops.index(prog)] = 1
                features.append(InputFeatures(
                    qid=example.qid,
                    example_index=example_index,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    concat_prog_ids=concat_prog_ids,
                    multi_prog_ids=multi_prog_ids,
                    concat_prog=concat_prog,
                    multi_prog=multi_prog

                ))
        return features

    def data_loader(self):
        args = self.args
        global_rank = -1 if args.local_rank == -1 else torch.distributed.get_rank()
        features = self.convert_examples_to_features()
        # 合并所有的每个feature所有的ids.
        #         print(len(features))
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_concat_prog_ids = torch.tensor([f.concat_prog_ids for f in features], dtype=torch.long)
        all_multi_prog_ids = torch.tensor([f.multi_prog_ids for f in features], dtype=torch.float)

        if self.is_training:
            # 生成batch.
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_concat_prog_ids,
                                    all_multi_prog_ids)
            args.n_gpu = 1
            args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            all_example_index = torch.tensor([f.example_index for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_concat_prog_ids,
                                    all_multi_prog_ids, all_example_index)
            batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
        if self.is_training:
            return dataloader
        else:
            return dataloader



##  数据清洗测试部分。

if __name__ == '__main__':
    MODEL_CLASSES = {
        'bert': (BertConfig, BertTokenizer),
    }
    args = set_args()
    args.n_gpu = 1
    args.per_gpu_eval_batch_size = 8
    model_name_or_path = "bert-base-uncased"
    config_class, tokenizer_class = MODEL_CLASSES['bert']

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
    train_data = CustomDataLoader(args, tokenizer, is_training=False)

    for batch in train_data.data_loader():
        # print(batch[3])
        pass
