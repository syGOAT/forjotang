from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import set_args, set_environ
from tensorboardX import SummaryWriter
import sys
sys.path.append('.')
import torch
from training.trainer import Trainer
from prepro import CustomDataLoader
from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer,
                          XLMConfig, XLMForQuestionAnswering,
                          XLMTokenizer, XLNetConfig,
                          XLNetForQuestionAnswering,
                          XLNetTokenizer,
                          RobertaConfig,
                          RobertaForQuestionAnswering,
                          RobertaTokenizer)
from model import Bert_Model
import os


MODEL_CLASSES = {
    'bert': (BertConfig, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetTokenizer),
    'xlm': (XLMConfig, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaTokenizer),
}
if __name__ == '__main__':
    args = set_args()
    args, logger, tb_writer, device = set_environ(args)
    # Loading pretrained model and tokenizer

    # set tokenizer
    config_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    if args.do_train:
        model = Bert_Model(config, args)
        if os.path.exists(args.train_last_checkpoint):
            stat_dict = torch.load(args.train_last_checkpoint, map_location=device)
            model.load_state_dict(stat_dict)
    else:
        logger.info('loading eval checkpoints {}'.format(args.eval_checkpoint))
        model = Bert_Model(config, args)
        stat_dict = torch.load(args.eval_checkpoint, map_location='cpu')
        model.load_state_dict(stat_dict)

    model.to(device)
    logger.info('Training/evaluation parameters {}'.format(args))

    if args.do_train:
        logger.info('Starting to train .......')
        train_model = Trainer(args=args, model=model, tokenizer=tokenizer,
                              load_and_cache_examples=CustomDataLoader, logger=logger,
                              tb_writer=tb_writer, is_training=True)
        global_step, tr_loss = train_model.train()
        logger.info('global_steps={} , average loss = {}'.format(global_step, tr_loss))

    if args.do_eval:
        model = torch.nn.DataParallel(model)
        logger.info('Staring to evaluating .......')

        eval_model = Trainer(args, model, tokenizer, CustomDataLoader, logger, tb_writer, False)
        results = eval_model.evaluate(args, model, tokenizer, CustomDataLoader, write_file=True)
        print(results)
