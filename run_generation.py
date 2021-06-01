#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""

from collections import defaultdict
import argparse
import logging
import pickle
import json
from tqdm import tqdm, trange
import re

import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from utils.my_transformer import GPT2DoubleHeadsModel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import os

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2DoubleHeadsModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prompt_text = (args.padding_text if args.padding_text else PADDING_TEXT) + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


class TextDataset(Dataset):
    def __init__(self, file_path: str, args, block_size=512):
        self.p = re.compile(r'([\d ,' ']+ - \d{1,2} - \d{1,2})')
        self.is_dev = 'dev' in file_path
        logger.info("Prepare Data...")
        cached_features_file = os.path.join(file_path, "data.pkl")
        if os.path.exists(cached_features_file):
            if self.is_dev:
                self.histories, self.responses, self.segments, self.knowledges, self.kn_vocs, self.chooses = pickle.load(open(cached_features_file, "rb"))
            else:
                self.histories, self.segments, self.knowledges, self.kn_vocs = pickle.load(open(cached_features_file, "rb"))
        else:
            self.histories = []
            self.responses = []
            self.knowledges = []
            self.kn_vocs = []
            self.segments = []

            with open(os.path.join(file_path, "samples.txt"), encoding='utf-8') as f:
                x = f.readlines()

            with open("vocab.txt", encoding="utf-8") as f:
                self.voc = [word.strip().split('\t')[0] for word in f.readlines()]

            for _, sample in enumerate(x):
                sample = json.loads(sample, encoding='utf-8')
                situation = self.sentence2id(sample["situation"])
                sample["Bot"] = u"User 主动" in sample["goal"].split("-->")[0]
                segment = []
                history = []
                for _ in range(len(sample["history"])):
                    sen_ids = self.sentence2id(sample["history"][_])
                    history.append(sen_ids)
                    if _ % 2 == 1 - sample["Bot"]:
                        segment.append([2 - sample["Bot"]] * len(sen_ids))
                    else:
                        segment.append([1 + sample["Bot"]] * len(sen_ids))
                if len(history) != 0:
                    if self.is_dev:
                        tmp = np.concatenate(history[::-1][1:args.windows][::-1], 0).tolist()
                        tmp_s = np.concatenate(segment[::-1][1:args.windows][::-1], 0).tolist()
                    else:
                        tmp = np.concatenate(history[::-1][:args.windows-1][::-1], 0).tolist()
                        tmp_s = np.concatenate(segment[::-1][:args.windows-1][::-1], 0).tolist()
                else:
                    tmp = []
                    tmp_s = []
                self.histories.append(situation + tmp + [3])
                self.segments.append([1] * len(situation) + tmp_s + [1])
                if self.is_dev:
                    self.responses.append(history[-1] + [3])
                self.knowledges.append([self.sentence2id(k) for k in sample["knowledge"]])
                self.kn_vocs.append(list(set(np.concatenate(self.knowledges[-1], 0))))

            if self.is_dev:
                pickle.dump((self.histories, self.responses, self.segments, self.knowledges, self.kn_vocs, self.chooses), open(cached_features_file, "wb"))
            else:
                pickle.dump((self.histories, self.segments, self.knowledges, self.kn_vocs), open(cached_features_file, "wb"))

        logger.info(str(len(self.histories)))

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, item):
        if self.is_dev:
            return self.histories[item], self.responses[item], self.knowledges[item], self.kn_vocs[item], self.segments[item]
        return self.histories[item], self.knowledges[item], self.kn_vocs[item], self.segments[item]

    def sentence2id(self, sentence):
        sentence = self.filiter(sentence)
        sen_ids = []
        for word in sentence.split(' '):
            if word == '': continue
            if word in self.voc:
                sen_ids.append(self.voc.index(word))
            else:
                sen_ids.append(len(self.voc) - 1)
        return sen_ids

    def filiter(self, sentence):
        if sentence.startswith('['):
            sentence = sentence[3:]
        date_list = re.findall(self.p, sentence)
        if date_list:
            for date in date_list:
                sentence = sentence.replace(date, date.replace('-', '年', 1).replace('-', '月', 1) + ' 日')
        return sentence


def pad_data(insts, pad_len, pad_num=-1, pad_id=0):
    """ padding ids """
    insts_pad = []
    if isinstance(insts[0], list):
        for inst in insts:
            inst_pad = inst + [pad_id] * (pad_len - len(inst))
            insts_pad.append(inst_pad)
        if len(insts_pad) < pad_num:
            insts_pad += [[pad_id] * pad_len] * (pad_num - len(insts_pad))
    else:
        insts_pad = insts + [pad_id] * (pad_len - len(insts))
    return insts_pad


def cal_max_len(ids):
    """ calculate max sequence length """
    if isinstance(ids[0], list):
        pad_len = max([cal_max_len(k) for k in ids])
    else:
        pad_len = len(ids)
    return pad_len


def collate(example):
    example = np.array(example)
    pad_histories = max([cal_max_len(s_inst) for s_inst in example[:, 0]])
    pad_kns = max([cal_max_len(k_inst) for k_inst in example[:, 1]])
    pad_kn_num = max([len(k_inst) for k_inst in example[:, 1]])
    pad_kn_vocs = max([cal_max_len(v_inst) for v_inst in example[:, 2]])
    pad_segments = max([cal_max_len(t_inst) for t_inst in example[:, 3]])

    histories = [pad_data(h_inst, pad_histories) for h_inst in example[:, 0]]
    kns = [pad_data(k_inst, pad_kns, pad_kn_num) for k_inst in example[:, 1]]
    kn_vocs = [pad_data(v_inst, pad_kn_vocs) for v_inst in example[:, 2]]
    segments = [pad_data(t_inst, pad_segments) for t_inst in example[:, 3]]

    return histories, kns, kn_vocs, segments


def collate_dev(example):
    example = np.array(example)
    pad_histories = max([cal_max_len(s_inst) for s_inst in example[:, 0]])
    pad_responses = max([cal_max_len(s_inst) for s_inst in example[:, 1]])
    pad_kns = max([cal_max_len(k_inst) for k_inst in example[:, 2]])
    pad_kn_num = max([len(k_inst) for k_inst in example[:, 2]])
    pad_kn_vocs = max([cal_max_len(v_inst) for v_inst in example[:, 3]])
    pad_segments = max([cal_max_len(t_inst) for t_inst in example[:, 4]])

    histories = [pad_data(h_inst, pad_histories) for h_inst in example[:, 0]]
    responses = [pad_data(s_inst, pad_responses) for s_inst in example[:, 1]]
    kns = [pad_data(k_inst, pad_kns, pad_kn_num) for k_inst in example[:, 2]]
    kn_vocs = [pad_data(v_inst, pad_kn_vocs) for v_inst in example[:, 3]]
    segments = [pad_data(t_inst, pad_segments) for t_inst in example[:, 4]]

    return histories, responses, kns, kn_vocs, segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--output_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--windows", type=int, default=500, required=True)

    parser.add_argument("--padding_text", type=str, default="", help="Padding text for Transfo-XL and XLNet.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)


    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    eval_dataset = TextDataset(file_path=args.input_file, args=args)
    eval_sampler = SequentialSampler(eval_dataset)
    if 'dev' in args.input_file:
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=20, collate_fn=collate_dev)
    else:
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=20, collate_fn=collate)
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", 20)

    output_sequences = []
    true_responses = []
    x_atts = []
    his = []
    for batch in tqdm(eval_dataloader, desc="Testing"):
        if 'dev' in args.input_file:
            histories, responses, knowledges, kn_vocs, segments = batch
            true_responses.extend(responses)
        else:
            histories, knowledges, kn_vocs, segments = batch

        histories = torch.LongTensor(histories).to(args.device)
        knowledges = torch.LongTensor(knowledges).to(args.device)
        kn_vocs = torch.LongTensor(kn_vocs).to(args.device)
        segments = torch.LongTensor(segments).to(args.device)
        his.extend(histories.tolist())

        o_s, x_att = model.generate(
            input_ids=histories,
            max_length=args.length + len(histories[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=False,
            num_return_sequences=args.num_return_sequences,
            pad_token_id=0,
            eos_token_ids=[3],
            knowledges=knowledges,
            kn_vocs=kn_vocs,
            segments=segments
        )
        output_sequences.extend(o_s)
        x_atts.extend(torch.squeeze(x_att).tolist())
        ma = 10000000 if 'dev'in args.input_file else 10000000
        if len(output_sequences) > ma:
            break

    with open("record.txt", "w", encoding='utf-8') as f:
        for att in x_atts:
            s = ''
            for num in att:
                if num != 0.0:
                    s += '%.2f '% num
            f.write(s + '\t' + str(np.argmax(att, -1)) + '\n')

    with open("vocab.txt", encoding="utf-8") as f:
        voc = [word.strip().split('\t')[0] for word in f.readlines()]

    generated_sequences = []
    ori_sequences = []
    ave_lenth = 0
    test_seqs = []
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()

        # Decode text
        message = ' '.join([voc[word] for word in his[generated_sequence_idx]])
        response = ' '.join([voc[word] for word in generated_sequence])

        if 'dev' in args.input_file:
            true_response = ' '.join([voc[word] for word in true_responses[generated_sequence_idx]])
            true_response = true_response[: true_response.find(args.stop_token) if args.stop_token else None]

        # Remove all text after the stop token
        message = message[: message.find(args.stop_token) if args.stop_token else None]
        response = response[: response.find(args.stop_token) if args.stop_token else None]

        new_response = ""
        # distinct
        for i in range(len(response)):
            if response[i] != '\n': new_response += response[i]
            else: new_response += ' '

        new_response = new_response.strip()

        ave_lenth += len(new_response.split(' '))
        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
                message + '\n' + new_response + '\n\n'
        )

        generated_sequences.append(f"{new_response}\n")
        ori_sequences.append(total_sequence)
        if 'dev' in args.input_file:
            test_seqs.append(new_response + '\t' + true_response + '\n')

    ave_lenth /= len(output_sequences)
    logger.info(f"Average length: {ave_lenth}")

    with open(args.output_file, "w", encoding='utf-8') as f:
        f.writelines(generated_sequences)

    with open(args.output_file + '.ori', "w", encoding='utf-8') as f:
        f.writelines(ori_sequences)

    if 'dev' in args.input_file:
        with open(args.output_file + '.test', "w", encoding='utf-8') as f:
            f.writelines(test_seqs)

    return generated_sequences


if __name__ == "__main__":
    main()
