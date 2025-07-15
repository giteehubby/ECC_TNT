from chat_api import chat_doubao
from tqdm import tqdm

lang_dict = {'zh': 'Chinese', 'ja': 'Japanese', 'en': 'English', 'de': 'German', 'fr': 'French', 'ar': 'Arabic', 'ko': 'Korean'}

import json5
import re
def extract_last_json(s: str, key_word:str='proper nouns') -> dict:
    """
    Extract the last valid JSON object from a string with potential nested/invalid JSON.
    Returns the parsed dictionary or None if no valid JSON is found.
    """
    stack = []
    last_valid_json = None
    s = re.sub(r'#(?![^\n]*")[^"\n]*', '', s) #去除注释
    # # s = re.sub(r'//[^\n]*', '', s) #去除注释


    # Track potential JSON starts and validate backward
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
        elif char == '}':
            if stack:
                start = stack.pop()
                try:
                    candidate = s[start:i+1]
                    last_json = json5.loads(candidate)
                    if key_word in last_json.keys():
                        last_valid_json = last_json
                    # last_valid_json = json.loads(candidate)
                except ValueError as e:
                    pass

    if last_valid_json is None:
        return None


    return last_valid_json


class LTCR:
    def __init__(self,chat_message,extract_tpl,src_lang,tgt_lang):
        self.chat_message = chat_message
        self.prompt_template = extract_tpl
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.all_records = dict()
        self.first_record = dict()

    def __call__(self,src_sentences,tgt_sentences):
        assert len(src_sentences) == len(tgt_sentences),\
        "Source and target sentences must have the same length."
        for src_sentence, tgt_sentence in \
            tqdm(zip(src_sentences, tgt_sentences),desc='extracting translate pairs...',total=len(src_sentences)):
            prompt = self.prompt_template.format(
                src_lang=lang_dict[self.src_lang],
                tgt_lang=lang_dict[self.tgt_lang],
                src=src_sentence,
                tgt=tgt_sentence
            )
            new_info = self.chat_message(prompt)
            new_info = extract_last_json(new_info, key_word='proper nouns')

            # print(f'src:\n\t{src_sentence}\ntgt:\n\t{tgt_sentence}\nnew_info:\n\t{new_info}')
            print(f'prompt:\n\t{prompt}\nnew_info:\n\t{new_info}')
            if new_info is not None:
                new_proper_noun_pairs = new_info['proper nouns']
                for ent_pair in new_proper_noun_pairs:
                    
                    src_ent, tgt_ent = ent_pair['proper noun'], ent_pair['corresponding translation']
                    if self.first_record.get(src_ent, '') == '':
                        self.all_records[src_ent] = [tgt_ent,]
                        if tgt_ent != 'N/A' or tgt_ent is not None or tgt_ent != '':
                            self.first_record[src_ent] = tgt_ent
                    else: # self.first_record[src_ent] != tgt_ent:
                        self.all_records[src_ent].append(tgt_ent)
                    # else:
                    #     print(ent_pair)
        n_count = 0
        c_count = 0
        f_count = 0
        for src_n in self.all_records.keys():
            n_count += (len(self.all_records[src_n]) - 1)
            for n in self.all_records[src_n][1:]:
                c_count += int(n == self.first_record[src_n])
                f_count += int(n in self.first_record[src_n] or self.first_record[src_n] in n)
                
        result = {
            'n_count': n_count,
            'c_count': c_count,
            'f_count': f_count,
            'all_records': self.all_records,
            'first_record': self.first_record
        }
        return result
    

def main():
    with open('./all_lang_data/2017-01-ted-test/zh-en/IWSLT17.TED.tst2017.zh-en.en.0',
              'r',encoding='utf-8') as f:
        src_text_list = f.readlines()
    with open('./all_lang_data/2017-01-ted-test/zh-en/IWSLT17.TED.tst2017.zh-en.zh.0',
              'r',encoding='utf-8') as f:
        tgt_text_list = f.readlines()
    with open('prompts/history_prompt_json.txt', 'r', encoding='utf-8') as extract_tpl_f:
        extract_tpl = extract_tpl_f.read()
    lcdr = LCDR(chat_doubao, extract_tpl,'zh','en')
    result = lcdr(src_text_list, tgt_text_list)
    print(result)

if __name__ == "__main__":
    main()

# python LTCR.py