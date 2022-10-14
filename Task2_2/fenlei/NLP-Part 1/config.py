import argparse
import torch
import random
import numpy as np
import shutil
from tensorboardX import SummaryWriter
import os
from util import copy_file
import logging


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def set_logger(save_to_file=None, name=None, global_rank=-1):
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


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="./dataset/train_debug.json", type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default="./dataset/test_debug.json", type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or evaluate.sh-v1.1.json")
    parser.add_argument("--model_type", default='roberta', type=str,
                        help="Model type selected in the list:")
    parser.add_argument("--model_name", default="bert-base-uncased",
                        type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default='checkpoints', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--write_dir", default='results', type=str,
                        help="The write directory where evaluation/inference results will be saved.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=None, type=int,
                        help="Batch size per GPU/CPU for evaluation. Defaults to 2x --per_gpu_train_batch_size")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta_1", default=0.9, type=float,
                        help="Beta 1 for Adam optimizer.")
    parser.add_argument("--adam_beta_2", default=0.999, type=float,
                        help="Beta 2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")

    parser.add_argument("--num_train_epochs",
                        default=50.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--preprocess_only", action='store_true',
                        help="Whether to only run preprocessing.")
    parser.add_argument('--dropout_rate', default=0.3, help='dropout rate')

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_freq', type=int, default=None,
                        help="Save/evaluate X times per epoch. Overrides save_steps.")
    parser.add_argument("--train_last_checkpoint", type=str,
                        default="./checkpoints_prog/checkpoints/checkpoint_multi_prog/model.pt",
                        help="旧模型位置")
    parser.add_argument("--eval_checkpoint", type=str,
                        default=None,
                        help="Evaluate only this checkpoint directory within args.output_dir.")

    parser.add_argument('--max_steps', type=int, default=-1, help='max steps to train')
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--master_port", type=int, default=15349,
                        help="master_port for distributed training on gpus. Must be unused and in [12000, 20000]")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--debug', action='store_true', help="use little examples to debug")
    parser.add_argument('--tb_target_dir', type=str, help='the dir the tb record to move!')
    parser.add_argument('--model_descri', type=str, help='describle the usage of model!')
    parser.add_argument('--num_labels', type=int, default=2, help='the classification number!')
    parser.add_argument('--metric', type=str, default='acc', help='the evaluation indicator!')
    parser.add_argument('--cache_postfix', type=str, help='the postfix for cached_file')
    parser.add_argument('--task', type=str, help='the target for model')
    parser.add_argument('--overwrite_cache', action='store_true', help='Whether to overwrite cached features file!')
    parser.add_argument('--tb_source_dir', type=str, help='the directory of summary')
    parser.add_argument('--multi_prog', action='store_true', help='Whether to use multi label loss to classify program')
    parser.add_argument('--concat_progs_number', type=int, default=129, help='The number of concat program ')
    parser.add_argument('--multi_labels', type=int, default=10, help='The number of multi labels')
    args = parser.parse_args()
    return args


def process_file_dirs(args, file_marks=None):
    """
    Automatic generate checkpoints dirs, predictions dirs, tensorboard dirs and logger dirs.
    """

    # def cat_args_to_file_mark(args):
    #
    #     file_mark = ''
    #     for idx, (name, value) in enumerate(args):
    #         if idx != 0:
    #             file_mark += '_' + name + '_' + str(value)
    #         else:
    #             file_mark += name + '_' + str(value)
    #     return file_mark

    def create_dirs(dir_list):
        """Create dir while dir does not exists!"""
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)

    # if file_marks is not None:
    #     args.file_marks = cat_args_to_file_mark(file_marks)
    # else:
    #     args.file_marks = ''
    #
    # if args.debug:
    #     args.file_marks = 'debug_' + args.file_marks

    if args.write_dir is None:
        args.write_dir = args.output_dir

    args.tb_source_dir = os.path.join(args.output_dir, 'tb', "tb_{}".format(args.model_descri.split('|')[0]))
    args.logger_dir = os.path.join(args.output_dir, 'logger', 'logger_{}'.format(args.model_descri.split('|')[0]))
    args.write_dir = os.path.join(args.write_dir, 'predictions',
                                  'predictions_{}'.format(args.model_descri.split('|')[0]))
    args.output_dir = os.path.join(args.output_dir, 'checkpoints',
                                   'checkpoint_{}'.format(args.model_descri.split('|')[0]))
    create_dirs([args.tb_source_dir, args.logger_dir, args.output_dir, args.write_dir])


def set_environ(args):
    args.model = args.model_type
    # 创建 log, tensorboard, output,checkpoints输出目录
    process_file_dirs(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    print(device)
    args.n_gpu = torch.cuda.device_count()
    tb_writer = SummaryWriter(args.tb_source_dir)
    logger = set_logger(args.logger_dir, name='__main__')
    # set seed
    set_seed(args)

    return args, logger, tb_writer, device
