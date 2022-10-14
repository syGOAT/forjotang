import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import random
import enum
import six
import copy
from six.moves import map
from six.moves import range
from six.moves import zip
from config import set_args

all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max",
           "table_min", "table_sum", "table_average"]
args = set_args()

def tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.

    Args:
      tokenizer: a tokenizer from bert.tokenization.FullTokenizer
      text: text to tokenize
      apply_basic_tokenization: If True, apply the basic tokenization. If False,
        apply the full tokenization (basic + wordpiece).

    Returns:
      tokenized text.

    A special token is any text with no spaces enclosed in square brackets with no
    space, so we separate those out and look them up in the dictionary before
    doing actual tokenization.
    """

    if  args.model_type in ["bert", "finbert"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
    elif args.model_type in ["roberta", "longformer"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize

    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.get_vocab():
                tokens.append(token)
            else:
                tokens.append(tokenizer.unk_token)
        else:

            tokens.extend(tokenize_fn(token))

    return tokens


def program_tokenization(original_program):
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)

    program.append('EOF')
    return program


class MathQAExample:
    def __init__(self, qid, original_question, question_tokens, concat_prog, multi_prog):
        self.qid = qid
        self.original_question = original_question
        self.question_tokens = question_tokens
        self.concat_prog = concat_prog
        self.multi_prog = multi_prog


def get_program_op_args(program):
    program = program[:-1]  # remove EOF
    # check structure
    for ind, token in enumerate(program):
        if ind % 4 == 0:
            if token.strip("(") not in all_ops:
                return False, 'None'
        if (ind + 1) % 4 == 0:
            if token != ")":
                return False, "None"

    program = "|".join(program)
    steps = program.split(")")[:-1]
    program_ops = collections.defaultdict(list)

    for ind, step in enumerate(steps):
        step = step.strip()
        if len(step.split("(")) > 2:
            return False, 'None'
        op = step.split("(")[0].strip("|").strip()
        op = op + str(ind)
        args = step.split("(")[1].strip("|").strip()
        arg1 = args.split("|")[0].strip()
        arg2 = args.split("|")[1].strip()
        program_ops[op].append(arg1)
        program_ops[op].append(arg2)
    return True, program_ops


def read_mathqa_entry(entry, tokenizer):
    question = entry["qa"]["question"]
    this_id = entry["id"]
    context = ""
    for ind, each_sent in entry["qa"]["model_input"]:
        context += each_sent
        context += " "
    context = context.strip()
    # process "." and "*" in text
    context = context.replace(". . . . . .", "")
    context = context.replace("* * * * * *", "")
    original_question = question + " " + tokenizer.sep_token + " " + context.strip()
    original_question_tokens = original_question.split(' ')
    question_tokens = []

    for i, tok in enumerate(original_question_tokens):
        tok_proc = tokenize(tokenizer, tok)
        question_tokens.extend(tok_proc)

    original_program = entry['qa']['program']
    prog_tokens = program_tokenization(original_program)
    flag, prog_args = get_program_op_args(prog_tokens)
    concat_prog = '_'.join(list(prog_args.keys()))
    multi_prog = list(prog_args.keys())

    return MathQAExample(
        qid=this_id,
        original_question=original_question,
        question_tokens=question_tokens,
        concat_prog=concat_prog,
        multi_prog=multi_prog)
