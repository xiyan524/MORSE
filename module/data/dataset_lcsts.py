import os
import json
import torch
import random

import numpy as np
from torch.utils.data import Dataset

class LCSTSDataset(Dataset):
    def __init__(self, file_path, params, do_shuffle=False):
        self.max_input_num = params['max_input_num']
        self.max_output_num = params['max_output_num']
        self.tokenizer = params['tokenizer']
        self.file_path = file_path
        self.do_shuffle = do_shuffle

        file_items = self.file_path.split("/")
        data_dir = "/".join(file_items[:-1])

        file_name = "input{}_".format(self.max_input_num) +"output{}_".format(self.max_output_num) + file_items[-1].split('.')[0] + ".npz"
        self.np_file = os.path.join(data_dir, file_name)

        self.load_dataset()

    def truncate(self, text, max_len, mode):
        text = list(text)
        if len(text) > max_len - 1:
            text = text[:max_len-1]
        if "start" == mode:
            text.insert(0, '<S>')
        elif "end" == mode:
            text.append('<T>')
        return text

    def load_dataset(self):
        if os.path.exists(self.np_file):
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
            self.ids = []

            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        item = json.loads(line)

                        content = self.truncate(item['content'], self.max_input_num, "")
                        content_id = self.tokenizer.convert_tokens_to_ids(content)
                        input_id = np.zeros(self.max_input_num, dtype=np.int)
                        input_id[:len(content_id)] = content_id

                        attention_mask = np.zeros(self.max_input_num, dtype=np.int)
                        attention_mask[:len(content_id)] = 1
                        segment_id = np.zeros(self.max_input_num, dtype=np.int)

                        summary_input = self.truncate(item['summary'], self.max_output_num, "start")
                        summary_input_id = self.tokenizer.convert_tokens_to_ids(summary_input)
                        decode_input_id = np.zeros(self.max_output_num, dtype=np.int)
                        decode_input_id[:len(summary_input_id)] = summary_input_id

                        summary_output = self.truncate(item['summary'], self.max_output_num, "end")
                        summary_output_id = self.tokenizer.convert_tokens_to_ids(summary_output)
                        decode_output_id = np.zeros(self.max_output_num, dtype=np.int)
                        decode_output_id[:len(summary_output_id)] = summary_output_id

                        self.inputs.append(input_id)
                        self.attention_mask.append(attention_mask)
                        self.segment_ids.append(segment_id)
                        self.decoder_inputs.append(decode_input_id)
                        self.decoder_outputs.append(decode_output_id)
                    else:
                        print("wrong line in file" + item["id"])

            # transfer np file for storage
            self.inputs = np.array(self.inputs)
            self.attention_mask = np.array(self.attention_mask)
            self.segment_ids = np.array(self.segment_ids)
            self.decoder_inputs = np.array(self.decoder_inputs)
            self.decoder_outputs = np.array(self.decoder_outputs)
            np.savez(self.np_file, inputs=self.inputs, attention_mask=self.attention_mask, segment_ids=self.segment_ids,
                     decoder_inputs=self.decoder_inputs, decoder_outputs=self.decoder_outputs)

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
            torch.tensor(self.segment_ids[index]),
            torch.tensor(self.decoder_inputs[index]),
            torch.tensor(self.decoder_outputs[index])
        ]