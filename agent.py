from openai import OpenAI
import argparse
import numpy as np
from numpy.linalg import norm
import json
from typing import Tuple, Union
from tqdm import tqdm
from copy import deepcopy
import random
import time
import os
from chat_api import chat_doubao

chat_message = chat_doubao

class RetrieveAgent():
    def __init__(self, total: int, prompt_template: str, skip_context: int) -> None:
        self.src_text_list = []
        self.tgt_text_list = []
        self.total = total
        
        self.prompt_template = prompt_template

        self.example_number = None
        self.skip_context = skip_context

    def insert(self, new_src: str, new_tgt: str):
        if self.total == -1 or len(self.src_text_list) < self.total:
            self.src_text_list.append(new_src)
            self.tgt_text_list.append(new_tgt)
        else:
            self.src_text_list = self.src_text_list[1:] + [new_src]
            self.tgt_text_list = self.tgt_text_list[1:] + [new_tgt]
    
    def match(self, query: str, num: int) -> Tuple[list[str]]:
        if len(self.src_text_list) <= num:
            return (self.src_text_list, self.tgt_text_list)

        sent_list = ''
        for idx, src in enumerate(self.src_text_list):
            sent_list += f'<Sentence {idx + 1}> {src}\n'
        sent_list = sent_list.strip()

        if self.example_number is None or len(self.example_number) != num:
            random.seed(0)
            self.example_number = random.sample(list(range(max(10, num))), num)
            self.example_number.sort()
        example_num_prompt = [str(i) for i in self.example_number]
        example_num_prompt = ', '.join(example_num_prompt[:-1]) + ' and ' + example_num_prompt[-1] if num > 1 else example_num_prompt[0]
        example_list_prompt = str(self.example_number)

        prompt = self.prompt_template.format(
            top_num=num,
            sentence_list=sent_list,
            example_number=example_num_prompt,
            example_list=example_list_prompt,
            query=query
        )

        chosen_ids = chat_message(prompt)

        global TRANS_CNT
        if (TRANS_CNT + 1) % 10 == 0:
            print('\n\n##### prompt:')
            print(prompt + '\n\n')
            print('\n\n##### chosen ids:')
            print(chosen_ids + '\n\n')
        if chosen_ids is None:
            return ([], [])
        try:
            chosen_ids = eval(chosen_ids)
        except Exception as e:
            chosen_ids = []
        chosen_ids = [i for i in chosen_ids if type(i) is int and 1 <= i <= len(self.src_text_list)]
        chosen_ids.sort()
        return ([self.src_text_list[i-1] for i in chosen_ids], [self.tgt_text_list[i-1] for i in chosen_ids])