import torch
import sys

sys.path.append('..')
from abc import abstractmethod
from util import softmax
from pytorch_transformers import AdamW, WarmupLinearSchedule
import sys
import json
import tqdm
from collections import Counter
import os
from apex import amp
from tqdm import tqdm, trange
import numpy as np


def accuracy(args, logits, prog_ids):
    results = 0
    """
    
    准确率函数的具体实现
    
    注：
    1. logits为模型的输出值，这个输出是个什么玩意儿？它代表了什么？我们该如何处理这个输出以获取我们想要的值？
    2. prog_ids为我们预处理时针对数据的真实结果处理得到的目标值，logits应如何与它进行对比？
    3. 结合trainer中训练和测试部分的代码，我们应当如何去评估他的accuracy？
    
    """

    return results


class BaseTrainer:
    """Base class for all trainers"""

    def __init__(self, args, model, tokenizer, load_and_cache_examples, logger, tb_writer, is_training):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.load_and_cache_examples = load_and_cache_examples
        self.is_training = is_training
        self.logger = logger
        self.tb_writer = tb_writer
        self.best_metric = -1
        self.metric = args.metric
        self.global_step = 0
        self.tr_loss, self.logging_loss, self.tr_batch_loss = 0, 0, 0
        self.global_rank = -1 if self.args.local_rank == -1 else torch.distributed.get_rank()
        self.step = None
        self.scheduler = None
        self.optimizer = None
        self.epoch_iterator = None

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        args = self.args
        model = self.model
        logger = self.logger
        tb_writer = self.tb_writer
        self.global_rank = -1 if self.args.local_rank == -1 else torch.distributed.get_rank()

        train_dataloader = \
            self.load_and_cache_examples(args, self.tokenizer, self.is_training).data_loader()

        if self.args.preprocess_only:
            sys.exit(0)

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // len(train_dataloader) // args.gradient_accumulation_steps + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        if args.save_freq is not None:
            args.save_steps = len(train_dataloader) // args.gradient_accumulation_steps // args.save_freq

        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                               betas=(args.adam_beta_1, args.adam_beta_2),
                               eps=args.adam_epsilon)

        self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=args.warmup_proportion * t_total,
                                              t_total=t_total)

        if args.fp16:
            self.model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=args.fp16_opt_level)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)

        # Train
        args.effective_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * \
                                          (torch.distributed.get_world_size() if self.global_rank != -1 else 1)

        logger.info("***** Runing training *****")
        logger.info(" Num examples {}".format(len(train_dataloader)))
        logger.info(" Num Epochs {}".format(int(args.num_train_epochs)))
        logger.info(" Instantaneous batch size of per GPU = {}".format(args.per_gpu_train_batch_size))
        logger.info(
            " Total train batch size (w. parallel, distributed & accumulation) = {}".format(
                args.effective_train_batch_size))
        logger.info(" Gradient Accmulation steps = {}".format(args.gradient_accumulation_steps))
        logger.info(" Total optimization steps = {}".format(t_total))
        logger.info(' Save step every = {}'.format(args.save_steps))

        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=self.global_rank not in [-1, 0])

        tr_loss = 0.0
        for _ in train_iterator:
            self.epoch_iterator = tqdm(train_dataloader, desc='Iteration', disable=self.global_rank not in [-1, 0])
            for step, batch in enumerate(self.epoch_iterator):
                self.step = step
                batch = tuple(t.to(args.device) for t in batch)
                loss = self._train_epoch(batch)
                tr_loss += loss.item()
                if args.max_steps > 0 and self.global_step > args.max_steps:
                    self.epoch_iterator.close()
                    break
            if args.max_steps > 0 and self.global_step > args.max_steps:
                train_iterator.close()
                break
        if self.global_rank in [-1, 0]:
            tb_writer.close()

        logger.info('max metric {} {}'.format(self.metric, self.best_metric))
        return self.global_step, tr_loss / self.global_step


class Trainer(BaseTrainer):
    def __init__(self, args, model, tokenizer, load_and_cache_examples, logger, tb_writer, is_training, ):
        super(Trainer, self).__init__(args, model, tokenizer, load_and_cache_examples, logger, tb_writer, is_training, )

    def _train_epoch(self, batch):
        self.model.train()
        args = self.args
        optimizer = self.optimizer
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'input_mask': batch[1],
                  'segment_ids': batch[2],
                  'concat_prog_ids': batch[3],
                  'multi_prog_ids': batch[4]
                  }

        loss = self.model(**inputs)

        if args.n_gpu > 1:
            loss = loss.mean()

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.tr_batch_loss += loss.item()
        self.tr_loss += loss.item()
        global_examples_seen = 0
        if (self.step + 1) % args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            self.global_step += 1
            global_examples_seen = self.global_step * args.effective_train_batch_size
            self.epoch_iterator.desc = 'loss: {:.2e} lr: {:.2e}'.format(self.tr_batch_loss, self.scheduler.get_lr()[0])
            self.tr_batch_loss = 0.0

        if self.global_rank in [-1, 0] and args.logging_steps > 0 and self.global_step % args.logging_steps == 0:
            self.tb_writer.add_scalar('lr', self.scheduler.get_lr()[0], global_examples_seen)
            self.tb_writer.add_scalar('loss', (self.tr_loss - self.logging_loss) / args.logging_steps,
                                      global_examples_seen)
            self.logging_loss = self.tr_loss

        if self.global_rank in [-1, 0] and args.save_steps > 0 and self.global_step % args.save_steps == 0:
            results_file = os.path.join(args.output_dir, 'results.json')
            self.logger.info('evaluate global rank {}'.format(self.global_rank))

            results = self.evaluate(args, self.model, self.tokenizer, self.load_and_cache_examples,
                                    metric=self.metric, max_metric=self.best_metric)

            if self.best_metric < results[self.metric]:
                self.best_metric = results[self.metric]
                self.logger.info('max_metric {}'.format(self.best_metric))
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_dict = model_to_save.state_dict()
            torch.save(model_dict, os.path.join(args.output_dir, 'model.pt'))
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
            results['checkpoint_path'] = os.path.join(args.output_dir, 'pytoch_model.bin')
            results['model_describle'] = args.model_descri
            results['f1'] = self.best_metric
            with open(results_file, 'w') as f:
                json.dump(results, f)
            self.logger.info('Saving model checkpoint to {}'.format(args.output_dir))
        return loss

    def evaluate(self, args, model, tokenizer, load_and_cache_examples,
                 prefix='dev', write_file=True, max_metric=None, metric=None):

        eval_dataloader = load_and_cache_examples(args, tokenizer, is_training=False).data_loader()
        if args.preprocess_only:
            return
        if args.eval_batch_size is None:
            args.eval_batch_size = args.train_batch_size * 2

        self.logger.info("***** Running evaluation {} *****".format(prefix))
        self.logger.info("  Num examples = %d", len(eval_dataloader))
        self.logger.info("  Batch size = %d", args.eval_batch_size)
        eval_accuracy = 0.0
        nb_eval_examples = 0
        model.eval()
        for batch in tqdm(eval_dataloader, desc='Evaluating'):

            batch = tuple(t.to(args.device) for t in batch[:-1])
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'input_mask': batch[1],
                          'segment_ids': batch[2],
                          'concat_prog_ids': batch[3],
                          'multi_prog_ids': batch[4]
                          }
                logits = model(**inputs)
                if args.multi_prog:
                    tmp_eval_accuracy = accuracy(args, logits, batch[4])
                else:
                    tmp_eval_accuracy = accuracy(args, logits, batch[3])
                eval_accuracy += tmp_eval_accuracy
                nb_eval_examples += batch[0].size(0)

        if nb_eval_examples > 0:
            eval_accuracy = eval_accuracy / nb_eval_examples

        results = {"acc": eval_accuracy}
        self.logger.info('eval_accuracy: {}'.format(eval_accuracy))
        return results
