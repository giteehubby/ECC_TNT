from volcenginesdkarkruntime import Ark
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI


#
# with open('ark_key.key','r') as f:
#     doubao_key = f.read()
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
                 #"type": "disabled", # 不使用深度思考能力
                 # "type": "enabled" # 使用深度思考能力
                  "type": "auto" # 模型自行判断是否使用深度思考能力
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

# with open('openai_key.key','r') as f:
#     openai_key = f.read()
embedding_client =  Ark(api_key="6210bb04-f3d1-4313-aa97-27f08aa5acec")

def get_embedding(text: str) -> str:
    text = text.replace("\n", " ")
    while True:
        completion = embedding_client.embeddings.create(
                input=[text],
                model='doubao-embedding-large-text-250515'
            )
        if completion is not None:
            return completion.data[0].embedding


# 全局模型和分词器
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path: str ='Qwen2'):
    global model,tokenizer,device
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    # 根据硬件条件选择加载方式
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    model.eval()


def chat_qwen(prompt:str)->str :
    global model,tokenizer,device
    # 确保模型已加载
    if model is None or tokenizer is None:
        load_model()
    try_cnt = 0
    max_new_tokens=2048
    while try_cnt < 10:
        try:
            # 构建聊天格式的消息
            messages = [{"role": "user", "content": prompt}]

            # 应用聊天模板
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # 准备模型输入
            model_inputs = tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(device)

            # 生成响应
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # 提取新生成的token（排除输入部分）
            output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]

            # 解码响应
            response = tokenizer.decode(
                output_ids,
                skip_special_tokens=True
            ).strip()

            return response

        except RuntimeError as e:
            # 处理显存不足的情况
            if "CUDA out of memory" in str(e):
                print("CUDA OOM error, reducing max tokens")
                max_new_tokens = max(512, int(max_new_tokens * 0.8))
                continue
            else:
                print(f"Runtime error: {e}")
                try_cnt += 1
                time.sleep(2)

        except Exception as e:
            print(f"Error: {e}")
            try_cnt += 1
            time.sleep(2)

    print('Error after 10 attempts')
    return None


def load_deepseek_model(model_path: str = "DeepSeek/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B"):
    """加载本地DeepSeek模型"""
    global model, tokenizer
    # 关键：必须设置 trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    # 根据硬件配置加载模型
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    model.eval()



def chat_deepseek(prompt: str, max_new_tokens: int = 2048) -> str:
    """使用DeepSeek模型生成响应"""
    global model, tokenizer
    if model is None or tokenizer is None:
        load_deepseek_model()  # 确保模型已加载
    try:
        # 1. 构建消息格式（DeepSeek专用格式）
        messages = [
            {"role": "user", "content": prompt}
        ]

        # 2. 应用聊天模板
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(device)

        # 3. 生成文本
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # 4. 解码并提取新生成的文本
        response = tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    except RuntimeError as e:
        # 处理显存不足错误
        if "CUDA out of memory" in str(e):
            print(f"显存不足，减少生成长度到{max_new_tokens // 2}")
            return chat_deepseek(prompt, max_new_tokens // 2)

        print(f"运行时错误: {e}")
        return None

    except Exception as e:
        print(f"未知错误: {e}")
        return None


