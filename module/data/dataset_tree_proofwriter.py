import os
import json
import torch
import random

import numpy as np
from torch.utils.data import Dataset

class TreeDataset(Dataset):
    def __init__(self, file_path, mode, params, do_shuffle=False):
        self.max_input_num = params['max_input_num']
        self.max_output_num = params['max_output_num']
        self.tokenizer = params['tokenizer']
        self.data_ratio = params['data_ratio']
        self.mode = mode
        self.file_path = file_path
        self.do_shuffle = do_shuffle

        file_items = self.file_path.split("/")
        data_dir = "/".join(file_items[:-1])

        file_name = "input{}_".format(self.max_input_num) +"output{}_".format(self.max_output_num) + file_items[-1].split('.')[0] + ".npz"
        self.np_file = os.path.join(data_dir, file_name)

        self.load_dataset()

    def truncate(self, text, max_len):
        text = list(text)
        if len(text) > max_len - 1:
            text = text[:max_len-1]
        return text

    def subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return subsequent_mask == 0

    def remove_sent(self, sent):
        # remove the intermediate sentence
        index_s = sent.find(":")
        rst_list = []
        while index_s != -1:
            index_s = sent.find(":")
            index_e = sent.find(";")
            rst_list.append(sent[:index_s])
            sent = sent[index_e+1:]
        new_sent = "; ".join(rst_list)
        return new_sent

    def load_dataset(self):
        if False and os.path.exists(self.np_file):
            with np.load(self.np_file) as dataset:
                self.inputs = dataset["inputs"]
                self.attention_mask = dataset["attention_mask"]
                self.segment_ids = dataset["segment_ids"]
                self.decoder_outputs = dataset["decoder_outputs"]
                self.decoder_inputs = dataset["decoder_inputs"]
        else:
            self.inputs = []
            self.attention_mask = []
            self.segment_ids = []
            self.decoder_outputs = []
            self.decoder_inputs = []
            self.decoder_attention_mask = []
            self.ids = []

            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        item = json.loads(line)

                        idx = item['id']
                        context = item['context']
                        #question = item['question']
                        #answer = item['answer']
                        hypothesis = item['hypothesis']
                        proof = item['proof']
                        #proof = self.remove_sent(proof)

                        # encoder
                        hypothesis_tokens = self.tokenizer.tokenize(hypothesis)
                        #question_tokens = self.tokenizer.tokenize(question)
                        #answer_tokens = self.tokenizer.tokenize(answer)
                        context_tokens = self.tokenizer.tokenize(context)
                        #input_tokens = hypothesis_tokens + ['</s>'] + question_tokens + ['</s>'] + answer_tokens + ['</s>'] + context_tokens
                        input_tokens = hypothesis_tokens + ['</s>'] + context_tokens
                        #input_tokens = context_tokens

                        input_tokens = self.truncate(input_tokens, self.max_input_num)
                        input_tokens.append('</s>')
                        input_token_id = self.tokenizer.convert_tokens_to_ids(input_tokens)
                        input_id = np.zeros(self.max_input_num, dtype=np.int)
                        input_id[:len(input_token_id)] = input_token_id

                        attention_mask = np.zeros(self.max_input_num, dtype=np.int)
                        attention_mask[:len(input_token_id)] = 1
                        #print(len(input_token_id))

                        # decoder
                        proof_tokens = self.tokenizer.tokenize(proof)

                        decode_output = self.truncate(proof_tokens, self.max_output_num)
                        decode_output.append('</s>')
                        decode_output = self.tokenizer.convert_tokens_to_ids(decode_output)
                        decode_output_id = np.zeros(self.max_output_num, dtype=np.int)
                        decode_output_id[:len(decode_output)] = decode_output
                        decode_output_id[decode_output_id == self.tokenizer.pad_token_id] = -100

                        decoder_attention_mask = np.zeros(self.max_output_num, dtype=np.int)
                        decoder_attention_mask[:len(decode_output)] = 1
                        subsequent_mask = self.subsequent_mask(self.max_output_num).squeeze()
                        decoder_attention_mask = decoder_attention_mask & subsequent_mask


                        self.inputs.append(input_id)
                        self.attention_mask.append(attention_mask)
                        self.decoder_outputs.append(decode_output_id)
                        self.decoder_attention_mask.append(decoder_attention_mask)
                    else:
                        print("wrong line in file" + idx)

            # transfer np file for storage
            self.inputs = np.array(self.inputs)
            self.attention_mask = np.array(self.attention_mask)
            self.decoder_outputs = np.array(self.decoder_outputs)
            self.decoder_attention_mask = np.array(self.decoder_attention_mask)
            #np.savez(self.np_file, inputs=self.inputs, attention_mask=self.attention_mask, segment_ids=self.segment_ids,        decoder_inputs=self.decoder_inputs, decoder_outputs=self.decoder_outputs)

            if self.mode == "train":
                length = self.inputs.shape[0]
                keep_length = int(length * self.data_ratio)
                ids = list(range(length))
                random.shuffle(ids)
                keep_ids = ids[:keep_length]

                self.inputs = self.inputs[keep_ids]
                self.attention_mask = self.attention_mask[keep_ids]
                self.decoder_outputs = self.decoder_outputs[keep_ids]
                self.decoder_attention_mask = self.decoder_attention_mask[keep_ids]

        self.total_size = self.inputs.shape[0]
        self.indexes = list(range(self.total_size))
        if self.do_shuffle:
            random.shuffle(self.indexes)

    def __len__(self):
        return self.total_size

    def __getitem__(self, item):
        index = self.indexes[item]
        return[
            torch.tensor(self.inputs[index]),
            torch.tensor(self.attention_mask[index]),
            torch.tensor(self.decoder_outputs[index]),
            torch.tensor(self.decoder_attention_mask[index])
        ]