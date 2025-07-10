from volcenginesdkarkruntime import Ark
import time
from openai import OpenAI

with open('ark_key','r') as f:
    doubao_key = f.read()
doubao_client = Ark(api_key=doubao_key)

def chat_doubao(prompt: str) -> str:
    try_cnt = 0
    while try_cnt < 10:
        try:
            completion = doubao_client.chat.completions.create(
            model="doubao-seed-1-6-250615",
            messages=[
                {"role": "user", "content": prompt}
            ],
            thinking={
                 "type": "disabled", # 不使用深度思考能力
                 # "type": "enabled" # 使用深度思考能力
                 # "type": "auto" # 模型自行判断是否使用深度思考能力
             },
            # max_tokens=2048,    ## 模型最大输出长度,按需调整
        )
            if completion is None:
                raise RuntimeError('Returned None!')
            break
        except Exception as e:
            completion = None
            print(e)
            try_cnt += 1
            print("Retry in 2 seconds")
            time.sleep(2)

    if completion is None:
        print('Error waiting')
        return None
    
    response = completion.choices[0].message.content.strip()
    return response

with open('openai_key','r') as f:
    openai_key = f.read()
embedding_client =  OpenAI(api_key=openai_key)

def get_embedding(text: str) -> str:
    text = text.replace("\n", " ")
    while True:
        completion = embedding_client.embeddings.create(
                input=[text],
                model='text-embedding-3-small'
            )
        if completion is not None:
            return completion.data[0].embedding