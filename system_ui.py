import streamlit as st
import requests
import time
from volcenginesdkarkruntime import Ark
from PIL import Image
import json
import re
import logging

from chat_api import chat_doubao, chat_deepseek_api,chat_qwen
from sagent import memo_doct_agent_s


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置页面标题和图标
st.set_page_config(
    page_title="上下文一致性增强的篇章级小说翻译系统",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4a69bd;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #3c5aa6;
    }
    </style>
""", unsafe_allow_html=True)

# 应用标题
st.markdown('<h1 class="title">🌐 上下文一致性增强的篇章级小说翻译系统</h1>', unsafe_allow_html=True)

def clean_response(response):
    """清洗模型响应"""
    cleaned = re.sub(r'^(Output:|Output：|输出:|输出：|翻译:|翻译：|Response:|Response：|回答:|回答：|\s*)+', '', response)
    cleaned = re.sub(r'^(Assistant:|助手：|模型:|模型：|AI:|AI：|\s*)+', '', cleaned)
    cleaned = re.sub(r'[。！？：；,.!?:;]+$', '', cleaned)
    return cleaned.strip()

def translate_with_doubao(text, src_lang, tgt_lang, api_key):
    """使用火山引擎API进行翻译"""
    try:
        client = Ark(api_key=api_key)

        lang_map = {
            "中文": "Chinese",
            "英语": "English",
            "法语": "French",
            "西班牙语": "Spanish",
            "日语": "Japanese"
        }

        completion = client.chat.completions.create(
            model="doubao-seed-1-6-250615",
            messages=[
                {"role": "system",
                 "content": f"你是一位精通{lang_map[src_lang]}和{lang_map[tgt_lang]}的专业翻译，擅长将{lang_map[src_lang]}小说翻译成{lang_map[tgt_lang]}。请保持原文风格。"},
                {"role": "user", "content": text}
            ],
            thinking={"type": "auto"}
        )

        return clean_response(completion.choices[0].message.content)

    except Exception as e:
        logger.error(f"火山引擎翻译失败: {str(e)}")
        raise

# 初始化session state
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""
if 'source_text' not in st.session_state:
    st.session_state.source_text = ""
if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False
if 'api_config' not in st.session_state:
    st.session_state.api_config = {
        "url": "http://localhost:8000/translate",
        "model": "doubao-seed-1-6-250615",
        "api_key": ""
    }

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 系统配置")

    # 模型选择
    model_options = ["doubao-seed-1-6-250615", "deepseek-r1-250528", "Qwen2.5-0.5B-Instruct"]
    model = st.selectbox(
        "翻译模型",
        options=model_options,
        index=model_options.index(st.session_state.api_config["model"]),
        help="选择使用的翻译模型"
    )

    # # API密钥
    # api_key = st.text_input(
    #     "API密钥",
    #     value=st.session_state.api_config["api_key"],
    #     type="password",
    #     help="火山引擎ARK SDK所需的API密钥"
    # )
    #
    # # 保存配置
    # if st.button("保存配置"):
    #     st.session_state.api_config = {
    #         "url": "https://ark.volcengineapi.com",
    #         "model": model,
    #         "api_key": api_key
    #     }
    #     st.success("配置已保存!")
    #
    # st.divider()

    # 文件上传
    st.header("📁 文件翻译")
    uploaded_file = st.file_uploader(
        "上传文件进行翻译",
        type=["txt"],
        help="支持文本文件"
    )

    if uploaded_file is not None:
        st.session_state.source_text = uploaded_file.getvalue().decode("utf-8")
        st.success("文本文件内容已加载!")

# 主界面布局
col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 源文本")
        source_text = st.text_area(
            "请输入要翻译的文本:",
            value=st.session_state.source_text,
            placeholder="在此输入文本...",
            height=300,
            key="source_text_area",
            label_visibility="collapsed"
        )
        st.session_state.source_text = source_text

        # 语言选择
        col_lang1, col_lang2 = st.columns(2)
        with col_lang1:
            src_lang = st.selectbox(
                "源语言",
                options=["中文", "英语", "法语", "德语", "日语"],
                index=0
            )
        with col_lang2:
            tgt_lang = st.selectbox(
                "目标语言",
                options=["英语", "中文", "法语", "德语", "日语"],
                index=0
            )
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🌍 翻译结果")
        translated_text = st.text_area(
            "翻译结果将显示在这里:",
            value=st.session_state.translated_text,
            placeholder="翻译结果...",
            height=300,
            key="translated_text_area",
            disabled=True,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

# 翻译控制区域
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⚡ 翻译控制")
translate_btn = st.button("开始翻译", key="translate_btn", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

lang_dict = {"英语":'en', "中文":'zh', "法语":'fr', "德语":'de', "日语":'ja'}
context_window = 20
src_lang=lang_dict[src_lang]
tgt_lang=lang_dict[tgt_lang]
with open('prompts/all_lan_summary_prompts/'+src_lang+'_directly_summary_prompt.txt', 'r', encoding='utf-8') as src_summary_tpl_f:
    src_summary_tpl = src_summary_tpl_f.read()
with open('prompts/all_lan_summary_prompts/'+src_lang+'_directly_summary_prompt.txt', 'r',encoding='utf-8') as tgt_summary_tpl_f:
    tgt_summary_tpl = tgt_summary_tpl_f.read()
with open('prompts/history_prompt_json.txt', 'r', encoding='utf-8') as extract_tpl_f:
    extract_tpl = extract_tpl_f.read()
with open('prompts/trans_summary_long_context_history_prompt.txt', 'r') as trans_tpl_f:
    translate_tpl = trans_tpl_f.read()
with open('prompts/retrieve_prompt.txt', 'r', encoding='utf-8') as retrieve_tpl_f:
    retrieve_tpl = retrieve_tpl_f.read()

# 翻译按钮逻辑
if translate_btn:
    if not st.session_state.source_text.strip():
        st.warning("请输入要翻译的文本或上传文件")
    else:
        st.session_state.is_loading = True
        with st.spinner("正在翻译，请稍候..."):
            try:
                start_time = time.time()

                if st.session_state.api_config["model"] == "doubao-seed-1-6-250615":
                    chat_message = chat_doubao
                elif st.session_state.api_config["model"] == "deepseek-r1-250528":
                    chat_message = chat_deepseek_api
                else:
                    chat_message = chat_qwen

                mt_agent2 = memo_doct_agent_s(src_lang, tgt_lang, context_window, 'embedding', chat_message,
                                              src_summary_tpl, tgt_summary_tpl, extract_tpl, translate_tpl, 20)

                translated_text = '\n'.join(mt_agent2.translate_sentences(st.session_state.source_text.split('\n'), 2, 10, True,
                                                  model + '_emdedding1' + '.json'))

                processing_time = time.time() - start_time

                st.session_state.translated_text = translated_text

                # 显示翻译信息
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("翻译状态", "成功 ✅")
                with col2:
                    st.metric("处理时间", f"{processing_time:.2f}秒")
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.session_state.translated_text = f"翻译错误: {str(e)}"
                st.error(f"翻译失败: {str(e)}")
            finally:
                st.session_state.is_loading = False
                st.rerun()

# 页脚
st.divider()
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        技术支持: 火山引擎ARK SDK
    </div>
""", unsafe_allow_html=True)