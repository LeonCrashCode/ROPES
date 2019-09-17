# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

class Annotation:
    def __init__(self):

        pass
    def from_string(self, string):

        parts = string.split("|||")

        self.start = int(parts[0].strip().split()[1])
        self.end = int(parts[0].strip().split()[2])
        self.type = parts[1].strip().split(":")
        self.tokens = parts[2].strip().split()

        self.required = parts[3].strip()
        self.others = parts[4].strip()
        self.human_id = int(parts[5].strip())

        assert self.required == "REQUIRED"
        assert self.others == "-NONE-"
        #assert self.human_id == 0

    def serialization(self):
        return "A "+str(self.start) + " " + str(self.end) + "|||"+":".join(self.type)+"|||"+" ".join(self.tokens)+"|||"+self.required+"|||"+self.others+"|||"+str(self.human_id)
    def dump(self, writer):
        writer.write(self.serialization()+"\n")

class Example:
    def __init__(self):
        pass
    def from_raw(self, raw):
        self.tokens = raw[0].split()[1:]
        self.annotations = []
        for r in raw[1:]:
            ann = Annotation()
            ann.from_string(r)
            self.annotations.append(ann)
    def from_example(self, tokens, annotations):
        self.tokens = tokens
        self.annotations = annotations
    def dump(self, writer):
        writer.write("S "+" ".join(self.tokens)+"\n")
        for annotation in self.annotations:
            annotation.dump(writer)
        if len(self.annotations) == 0:
            writer.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
        writer.write("\n")

class Dataset:

    def __init__(self):

        self.data = []
        pass
    def append(self, item):
        self.data.append(item)

    def readfile(self, filename):
        temps = []
        with open(filename, "r") as reader:
            for line in reader:
                line = line.strip()
                if line == "":
                    example = Example()
                    example.from_raw(temps)
                    self.data.append(example)
                    temps = []
                else:
                    temps.append(line)

                if len(self.data) % 100000 == 0:
                    logger.info("...%d" % (len(self.data)))
            logger.info("The number of examples is %d" % (len(self.data)))
    def dump(self, filename):
        with open(filename, "w") as writer:
            for example in self.data:
                example.dump(writer)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, tokens, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens = tokens
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_part1_ids, label_part1_mask, label_part2_ids, label_part2_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_part1_ids = label_part1_ids
        self.label_part1_mask = label_part1_mask
        self.label_part2_ids = label_part2_ids
        self.label_part2_mask = label_part2_mask

def read_train_examples(input_file):
    """Read a SQuAD json file into a list of SquadExample."""

    dataset = Dataset()
    dataset.readfile(input_file)

    examples = []

    guid = 10000000
    for example in dataset.data:
        labels = ["O" for i in range(len(example.tokens)*2+1)]
        for ann in example.annotations:
            if ann.type[0] == "noop":
                pass
            elif ann.type[0] == "M":
                labels[ann.start*2] = "Y" # yes, want to insert
            elif ann.type[0] == "U":
                start = ann.start * 2
                end = ann.end * 2
                i = start + 1
                while i < end:
                    labels[i] = "U"
                    i += 2
            elif ann.type[0] == "R":
                start = ann.start * 2
                end = ann.end * 2
                i = start + 1
                while i < end:
                    labels[i] = "R"
                    i += 2
            else:
                assert "Unrecognized type"
        guid += 1
        inputexample = InputExample(
                    guid=guid,
                    tokens=example.tokens,
                    labels=labels)
        examples.append(inputexample)
    return examples

def read_dev_examples(input_file):
    pass

def read_label_list(input_file):
    labels = []
    with open(input_file,"r") as reader:
        for line in reader:
            line = line.strip()
            labels.append(line)
    return labels

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tok_to_orig_index = []
        orig_to_tok_index = []
        tokens = []
        labels = []
        for (i, token) in enumerate(example.tokens):
            orig_to_tok_index.append(len(tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i) # mapping to the original index (without tokenization) from tok index (with tokenization)
                tokens.append(sub_token)

            labels.append(example.labels[i*2])
            l = example.labels[i*2+1]
            labels += [l] + [l, l] * (len(sub_tokens) - 1)
        labels.append(example.labels[-1])
        # there are two parts in labels, first part is missing, and the other part is unnecessarity and replacement

        labels_part1 = []
        labels_part2 = []
        assert len(labels) % 2 == 1
        for i in range(len(labels)):
            if i % 2 == 0:
                labels_part1.append(labels[i])
            else:
                labels_part2.append(labels[i])

        # tokens_b = None
        # if example.text_b:
        #     tokens_b = tokenizer.tokenize(example.text_b)
        #     # Modifies `tokens_a` and `tokens_b` in place so that the total
        #     # length is less than the specified length.
        #     # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
        #     special_tokens_count = 4 if sep_token_extra else 3
        #     _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        # else:
        #     # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        #     special_tokens_count = 3 if sep_token_extra else 2
        #     if len(tokens_a) > max_seq_length - special_tokens_count:
        #         tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = [cls_token] + tokens + [sep_token]

        labels_part2 = ["O"] + labels_part2 + ["O"]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        segment_ids[0] = cls_token_segment_id

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        padding_length = max_seq_length - len(labels_part1) - 1 
        label_part1_mask = [1 if mask_padding_with_zero else 0] * len(labels_part1)
        label_part1_mask = label_part1_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        label_part1_ids = [label_map[l] for l in labels_part1]
        label_part1_ids = label_part1_ids + ([pad_token] * padding_length)

        padding_length = max_seq_length - len(labels_part2)
        label_part2_mask = [1 if mask_padding_with_zero else 0] * len(labels_part2)
        label_part2_mask = label_part2_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        label_part2_ids = [label_map[l] for l in labels_part2]
        label_part2_ids = label_part2_ids + ([pad_token] * padding_length)


        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_part1: %s" % " ".join([str(x) for x in labels_part1]))
            logger.info("label_part1_ids: %s" % " ".join([str(x) for x in label_part1_ids]))
            logger.info("label_part1_mask: %s" % " ".join([str(x) for x in label_part1_mask]))

            logger.info("label_part2: %s" % " ".join([str(x) for x in labels_part2]))
            logger.info("label_part2_ids: %s" % " ".join([str(x) for x in label_part2_ids]))
            logger.info("label_part2_mask: %s" % " ".join([str(x) for x in label_part2_mask]))
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_part1_ids=label_part1_ids,
                              label_part1_mask=label_part1_mask,
                              label_part2_ids=label_part2_ids,
                              label_part2_mask=label_part2_mask))
    return features


# def _truncate_seq_pair(tokens_a, tokens_b, max_length):
#     """Truncates a sequence pair in place to the maximum length."""

#     # This is a simple heuristic which will always truncate the longer sequence
#     # one token at a time. This makes more sense than truncating an equal percent
#     # of tokens from each, since if one sequence is very short then each token
#     # that's truncated likely contains more information than a longer sequence.
#     while True:
#         total_length = len(tokens_a) + len(tokens_b)
#         if total_length <= max_length:
#             break
#         if len(tokens_a) > len(tokens_b):
#             tokens_a.pop()
#         else:
#             tokens_b.pop()


# def simple_accuracy(preds, labels):
#     return (preds == labels).mean()


# def acc_and_f1(preds, labels):
#     acc = simple_accuracy(preds, labels)
#     f1 = f1_score(y_true=labels, y_pred=preds)
#     return {
#         "acc": acc,
#         "f1": f1,
#         "acc_and_f1": (acc + f1) / 2,
#     }


# def pearson_and_spearman(preds, labels):
#     pearson_corr = pearsonr(preds, labels)[0]
#     spearman_corr = spearmanr(preds, labels)[0]
#     return {
#         "pearson": pearson_corr,
#         "spearmanr": spearman_corr,
#         "corr": (pearson_corr + spearman_corr) / 2,
#     }


# def compute_metrics(task_name, preds, labels):
#     assert len(preds) == len(labels)
#     if task_name == "cola":
#         return {"mcc": matthews_corrcoef(labels, preds)}
#     elif task_name == "sst-2":
#         return {"acc": simple_accuracy(preds, labels)}
#     elif task_name == "mrpc":
#         return acc_and_f1(preds, labels)
#     elif task_name == "sts-b":
#         return pearson_and_spearman(preds, labels)
#     elif task_name == "qqp":
#         return acc_and_f1(preds, labels)
#     elif task_name == "mnli":
#         return {"acc": simple_accuracy(preds, labels)}
#     elif task_name == "mnli-mm":
#         return {"acc": simple_accuracy(preds, labels)}
#     elif task_name == "qnli":
#         return {"acc": simple_accuracy(preds, labels)}
#     elif task_name == "rte":
#         return {"acc": simple_accuracy(preds, labels)}
#     elif task_name == "wnli":
#         return {"acc": simple_accuracy(preds, labels)}
#     else:
#         raise KeyError(task_name)

# processors = {
#     "cola": ColaProcessor,
#     "mnli": MnliProcessor,
#     "mnli-mm": MnliMismatchedProcessor,
#     "mrpc": MrpcProcessor,
#     "sst-2": Sst2Processor,
#     "sts-b": StsbProcessor,
#     "qqp": QqpProcessor,
#     "qnli": QnliProcessor,
#     "rte": RteProcessor,
#     "wnli": WnliProcessor,
# }

# output_modes = {
#     "cola": "classification",
#     "mnli": "classification",
#     "mnli-mm": "classification",
#     "mrpc": "classification",
#     "sst-2": "classification",
#     "sts-b": "regression",
#     "qqp": "classification",
#     "qnli": "classification",
#     "rte": "classification",
#     "wnli": "classification",
# }

# GLUE_TASKS_NUM_LABELS = {
#     "cola": 2,
#     "mnli": 3,
#     "mrpc": 2,
#     "sst-2": 2,
#     "sts-b": 1,
#     "qqp": 2,
#     "qnli": 2,
#     "rte": 2,
#     "wnli": 2,
# }
