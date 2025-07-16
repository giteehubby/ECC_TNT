# from openai import OpenAI
# import argparse
import numpy as np
from numpy.linalg import norm
import json
from typing import Tuple, Union
from tqdm import tqdm
from copy import deepcopy
import random
# import time
# import os
from chat_api import chat_doubao
from chat_api import chat_qwen
from chat_api import chat_deepseek
from chat_api import get_embedding
from LTCR import extract_last_json

# self.chat_message = chat_doubao
lang_dict = {'zh': 'Chinese', 'ja': 'Japanese', 'en': 'English', 'de': 'German', 'fr': 'French', 'ar': 'Arabic', 'ko': 'Korean'}
SRC_SEP, TGT_SEP = ' ', ' '

def cosine_similarity(a: Union[np.array, list], b: Union[np.array, list]):
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    return np.dot(a, b) / (norm(a) * norm(b))

class EmbeddingDict():
    def __init__(self, total: int) -> None:
        self.embedding_list = []
        self.src_text_list = []
        self.tgt_text_list = []
        self.total = total
    
    def insert(self, new_src: str, new_tgt: str) -> None:
        if self.total == -1 or len(self.src_text_list) < self.total:
            # total==-1:保存所有的句子并从中检索 
            self.src_text_list.append(new_src)
            self.tgt_text_list.append(new_tgt)
        else:
            self.src_text_list = self.src_text_list[1:] + [new_src]
            self.tgt_text_list = self.tgt_text_list[1:] + [new_tgt]
    
    def match(self, query: str, num: int) -> Tuple[list[str]]:
        if len(self.embedding_list) <= num:
            if len(self.embedding_list) <self.total:
                query_embedding = get_embedding(query)
                self.embedding_list.append(query_embedding)
            return (self.src_text_list, self.tgt_text_list)
        query_embedding = get_embedding(query)
        sim_list = [cosine_similarity(query_embedding, embedding) for embedding in self.embedding_list]
        

        idx_list = list(range(len(sim_list)))
        idx_list.sort(key=lambda x: sim_list[x], reverse=True)
        # import pdb
        # pdb.set_trace()
        if self.total == -1 or len(self.embedding_list) < self.total:
            self.embedding_list.append(query_embedding)
        else:
            self.embedding_list = self.embedding_list[1:] + [query_embedding]
        return ([self.src_text_list[i] for i in idx_list[:num]], [self.tgt_text_list[i] for i in idx_list[:num]])

class RetrieveAgent():
    def __init__(self, chat_message, total: int, prompt_template: str) -> None:
        self.src_text_list = []
        self.tgt_text_list = []
        self.total = total
        self.chat_message = chat_message
        self.prompt_template = prompt_template

        self.example_number = None

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


        chosen_ids = self.chat_message(prompt)


        if chosen_ids is None:
            return ([], [])
        try:
            chosen_ids = eval(chosen_ids)
        except Exception as e:
            chosen_ids = []
        chosen_ids = [i for i in chosen_ids if type(i) is int and 1 <= i <= len(self.src_text_list)]
        chosen_ids.sort()

        return ([self.src_text_list[i-1] for i in chosen_ids], [self.tgt_text_list[i-1] for i in chosen_ids])



class Summary():
    def __init__(self, chat_message, src_template: str, tgt_template: str) -> None:
        self.src_summary = None
        self.tgt_summary = None
        self.chat_message = chat_message
        self.src_template = src_template
        self.tgt_template = tgt_template

    def set_summary(self, s_sum: str, t_sum) -> None:
        self.src_summary = s_sum
        self.tgt_summary = t_sum


    def update_summary(self, record_list: list[dict]) -> Tuple[str]:
        # generate summary
        src_list = [i['src'] for i in record_list]
        gen_list = [i['gen'] for i in record_list]
        
        src_para = SRC_SEP.join(src_list)
        gen_para = TGT_SEP.join(gen_list)
        
        prompt = self.src_template.format(summary=self.src_summary,paragraph=src_para)
        self.src_summary = self.chat_message(prompt)

        prompt = self.tgt_template.format(summary=self.tgt_summary,paragraph=gen_para)
        self.tgt_summary = self.chat_message(prompt)

        return (self.src_summary, self.tgt_summary)
    
    def get_summary(self) -> Tuple[str, str]:
        return (self.src_summary, self.tgt_summary)
    
class Noun_Record():
    def __init__(self, chat_message, prompt_template: str, src_lang: str, tgt_lang: str) -> None:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.chat_message = chat_message
        self.entity_dict = dict()
        self.prompt_template = prompt_template

    def extract_entity(self, src: str, tgt: str) -> list[str]:
        if src is None or src == '':
            return []
        prompt = self.prompt_template.format(
            src_lang=lang_dict[self.src_lang],
            tgt_lang=lang_dict[self.tgt_lang],
            src=src,
            tgt=tgt
        )
        new_info = self.chat_message(prompt)
        new_info = extract_last_json(new_info, key_word='proper nouns')

        conflicts = list()
        if new_info is not None:
            new_proper_noun_pairs = new_info['proper nouns']
            for ent_pair in new_proper_noun_pairs:                
                src_ent, tgt_ent = ent_pair['proper noun'], ent_pair['corresponding translation']
                if self.entity_dict.get(src_ent, '') == '':
                    # self.all_records[src_ent] = [tgt_ent,]
                    if tgt_ent != 'N/A' or tgt_ent is not None or tgt_ent != '':
                        self.entity_dict[src_ent] = tgt_ent
                else: 
                    if self.entity_dict[src_ent] != tgt_ent:
                        conflicts.append(f'"{src_ent}" - "{self.entity_dict[src_ent]}"/"{tgt_ent}"')
        return conflicts            

    
    def get_history_dict_string(self, sentence: str, only_relative: bool) -> str:
        if only_relative:
            # 只提取记录中与句子相关的
            entity_list = [ent for ent in self.entity_dict if ent in sentence]
            hist_list = [f'"{ent}" - "{self.entity_dict[ent]}"' for ent in entity_list]
            hist_prompt = ', '.join(hist_list)
            return hist_prompt
        else:
            hist_list = [f'"{ent}" - "{self.entity_dict[ent]}"' for ent in self.entity_dict]
            hist_prompt = ', '.join(hist_list)
            return hist_prompt

    def get_history_dict(self) -> dict:
        return deepcopy(self.entity_dict)

    def set_history_dict(self, h_dict: dict) -> None:
        self.entity_dict = h_dict

class Context():
    def __init__(self, window_size: int) -> None:
        self.windows_size = window_size
        self.src_context = []
        self.tgt_context = []
    
    def update(self, src: str, tgt: str) -> None:
        if src is None or src == '':
            return
        if self.windows_size == -1:
            # 无限长的上下文
            self.src_context.append(src)
            self.tgt_context.append(tgt)
        else:
            self.src_context = self.src_context[-(self.windows_size - 1):] + [src]
            self.tgt_context = self.tgt_context[-(self.windows_size - 1):] + [tgt]
    
    def get_context(self) -> Tuple[list[str]]:
        return (self.src_context, self.tgt_context)


class memo_doct_agent_s():
    def __init__(self,src_lang, tgt_lang, context_window, retriever,chat_message,
                 src_summary_tpl, tgt_summary_tpl, extract_tpl,trans_tpl,
                 long_window,retrive_tpl:str='') -> None:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.chat_message = chat_message
        self.short_memory = Context(context_window)
        if retriever == 'agent':
            self.long_memory = RetrieveAgent(chat_message, long_window,retrive_tpl)
        elif retriever == 'embedding':
            self.long_memory = EmbeddingDict(long_window)
        else:
            print('The type of the retriever must be "embedding" or "agent"!')
        
        self.doc_summary = Summary(chat_message, src_summary_tpl, tgt_summary_tpl)

        self.noun_record = Noun_Record(chat_message, extract_tpl, src_lang, tgt_lang)

        self.translate_template = trans_tpl

    def translate(self,
        src_text: str,
        rel_src_sents: list[str], rel_tgt_sents: list[str],
        src_summary: str, tgt_summary: str,
        noun_record: str,
        src_context: list, tgt_context: list, context_window: int,
        prompt_template: str
    ) -> dict:

        '''
        src_text: 源语言文本
        rel_src_sents: 相关源语言句子列表（长期记忆）
        rel_tgt_sents: 相关目标语言句子列表（长期记忆）
        src_summary: 源语言摘要
        tgt_summary: 目标语言摘要
        historical_prompt: 
        src_context: 源语言上下文（短期记忆）
        tgt_context: 翻译结果上下文（短期记忆）
        '''
        if src_text is None or src_text == '':
            return ('', 'N/A')
        if rel_src_sents is None or len(rel_src_sents) == 0:
            rel_instances = 'N/A'
        else:
            rel_instances = ''
            for rel_src, rel_tgt in zip(rel_src_sents, rel_tgt_sents):
                rel_instances += f'<{lang_dict[self.src_lang]} source> {rel_src}\n<{lang_dict[self.tgt_lang]} translation> {rel_tgt}\n'
            rel_instances = rel_instances.strip()
        if src_summary is None:
            src_summary = 'N/A'
        if tgt_summary is None:
            tgt_summary = 'N/A'
        if noun_record is None or noun_record == '':
            noun_record = 'N/A'
        if src_context is None or len(src_context) == 0:
            src_context_prompt, tgt_context_prompt = 'N/A', 'N/A'
        else:
            global SRC_SEP, TGT_SEP
            src_context_prompt = SRC_SEP.join(src_context)
            tgt_context_prompt = TGT_SEP.join(tgt_context)

        prompt = prompt_template.format(
            src_lang=lang_dict[self.src_lang],
            tgt_lang=lang_dict[self.tgt_lang],
            src_summary=src_summary,
            tgt_summary=tgt_summary,
            rel_inst=rel_instances,
            src=src_text,
            hist_info=noun_record,
            src_context=src_context_prompt,
            tgt_context=tgt_context_prompt,
            context_window=context_window,
        )

        gen = self.chat_message(prompt).replace('\n','')

        return (gen,prompt) if gen else ('',prompt)

    def translate_sentences(self,sentences,retrive_top_k,summary_step,only_relative:bool=True,output_file:str='./temp.json'):
        trans_records = []
        for idx,src_sentence in enumerate(tqdm(sentences)):
            if src_sentence is None or src_sentence == '':
                record = {'idx': idx, 'src': src_sentence, 'gen': '', 'prompt': ''}
                trans_records.append(record)
                json.dump(trans_records, open(output_file, 'w',encoding='utf-8'), ensure_ascii=False, indent=4)
                continue
            record = dict()
            # import pdb
            # pdb.set_trace()
            long_mem_srcs, long_mem_tgts = self.long_memory.match(src_sentence, retrive_top_k)
            long_mem_srcs, long_mem_tgts = deepcopy(long_mem_srcs), deepcopy(long_mem_tgts)

            src_summary, tgt_summary = self.doc_summary.get_summary()
            
            # 历史专有名词翻译信息
            hist_info = self.noun_record.get_history_dict_string(src_sentence, only_relative)

            src_context, tgt_context = self.short_memory.get_context()

            result, prompt = self.translate(src_sentence, long_mem_srcs, long_mem_tgts, src_summary, tgt_summary, hist_info, src_context, tgt_context, self.short_memory.windows_size, self.translate_template)

            record['idx'] = idx
            record['src'] = src_sentence
            record['gen'] = result
            record['prompt'] = prompt

            if (idx + 1) % summary_step == 0:
                record['new_src_summary'], record['new_tgt_summary'] = self.doc_summary.update_summary(trans_records[-summary_step:])

            self.long_memory.insert(src_sentence, result)

            conflict_list = self.noun_record.extract_entity(src_sentence, result)
                # new_ents = result['New proper nouns']
            if only_relative:
                # 提取到的与当前句子相关的实体
                record['hist_info'] = hist_info
            record['entity_dict'] = self.noun_record.get_history_dict()
                # conflict_list = ent_history.update_history(new_ents)
            if len(conflict_list) > 0:
                record['conflict'] = conflict_list

            # 更新短期记忆
            self.short_memory.update(src_sentence, result)

            trans_records.append(record)
            json.dump(trans_records, open(output_file, 'w',encoding='utf-8'), ensure_ascii=False, indent=4)
        return [record['gen'] for record in trans_records]
    
    def translate_stream(self,sentences,retrive_top_k,summary_step,only_relative:bool=True,output_file:str='./temp.json'):
        trans_records = []
        for idx,src_sentence in enumerate(tqdm(sentences)):
            if src_sentence is None or src_sentence == '':
                record = {'idx': idx, 'src': src_sentence, 'gen': '', 'prompt': ''}
                trans_records.append(record)
                json.dump(trans_records, open(output_file, 'w',encoding='utf-8'), ensure_ascii=False, indent=4)
                yield [record['gen'] for record in trans_records]
                continue
            record = dict()
            # import pdb
            # pdb.set_trace()
            long_mem_srcs, long_mem_tgts = self.long_memory.match(src_sentence, retrive_top_k)
            long_mem_srcs, long_mem_tgts = deepcopy(long_mem_srcs), deepcopy(long_mem_tgts)

            src_summary, tgt_summary = self.doc_summary.get_summary()
            
            # 历史专有名词翻译信息
            hist_info = self.noun_record.get_history_dict_string(src_sentence, only_relative)

            src_context, tgt_context = self.short_memory.get_context()

            result, prompt = self.translate(src_sentence, long_mem_srcs, long_mem_tgts, src_summary, tgt_summary, hist_info, src_context, tgt_context, self.short_memory.windows_size, self.translate_template)

            record['idx'] = idx
            record['src'] = src_sentence
            record['gen'] = result
            record['prompt'] = prompt

            if (idx + 1) % summary_step == 0:
                record['new_src_summary'], record['new_tgt_summary'] = self.doc_summary.update_summary(trans_records[-summary_step:])

            self.long_memory.insert(src_sentence, result)

            conflict_list = self.noun_record.extract_entity(src_sentence, result)
                # new_ents = result['New proper nouns']
            if only_relative:
                # 提取到的与当前句子相关的实体
                record['hist_info'] = hist_info
            record['entity_dict'] = self.noun_record.get_history_dict()
                # conflict_list = ent_history.update_history(new_ents)
            if len(conflict_list) > 0:
                record['conflict'] = conflict_list

            # 更新短期记忆
            self.short_memory.update(src_sentence, result)

            trans_records.append(record)
            json.dump(trans_records, open(output_file, 'w',encoding='utf-8'), ensure_ascii=False, indent=4)
            yield [record['gen'] for record in trans_records]


if __name__=='__main__':
    with open('prompts/zh_direct_summary_prompt.txt', 'r', encoding='utf-8') as src_summary_tpl_f:
        src_summary_tpl = src_summary_tpl_f.read()
    with open('prompts/en_direct_summary_prompt.txt', 'r') as tgt_summary_tpl_f:
        tgt_summary_tpl = tgt_summary_tpl_f.read()
    with open('prompts/history_prompt_json.txt', 'r', encoding='utf-8') as extract_tpl_f:
        extract_tpl = extract_tpl_f.read()
    with open('prompts/trans_summary_long_context_history_prompt.txt', 'r') as trans_tpl_f:
        translate_tpl = trans_tpl_f.read()
    # with open('prompts/retrieve_prompt.txt', 'r', encoding='utf-8') as retrieve_tpl_f:
    #     retrieve_tpl = retrieve_tpl_f.read()

    mt_sagent = memo_doct_agent_s(
        'zh', 'en', 3, 'embedding', chat_doubao, 
        src_summary_tpl, tgt_summary_tpl, extract_tpl, translate_tpl, 20)

    with open('data/0.chs_re.txt', 'r', encoding='utf-8') as f:
        src_text_list = f.readlines()
    results = mt_sagent.translate_sentences(src_text_list, 2, 10, True, './doubao_embedding_res.json')
    with open('output_doubao.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(results))

