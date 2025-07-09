from volcenginesdkarkruntime import Ark
import time


doubao_client = Ark(api_key="6210bb04-f3d1-4313-aa97-27f08aa5acec")

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