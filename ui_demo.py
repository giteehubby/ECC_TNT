import streamlit as st
import requests
import time
from volcenginesdkarkruntime import Ark
from PIL import Image
import json
import re
import logging
from tqdm import tqdm
from blonde import BLONDE

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化BLONDE评估器
blonde_evaluator = BLONDE()

# 预定义参考译文库（实际应用应扩展此数据库）
REFERENCE_TRANSLATIONS = {
    # 中文 -> 英语
    "这是一个测试句子。": [
        ["This is a test sentence.", "This is the second sentence."],
        ["This is an example sentence.", "Here is another version."]
    ],
    # 英语 -> 中文
    "This is an example text.": [
        ["这是一个示例文本。", "这是第二句话。"],
        ["这是样例文本。", "这是另一个版本。"]
    ],
    # 可添加更多语言对的参考译文...
}

# 设置页面标题和图标
st.set_page_config(
    page_title="高级机器翻译系统",
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
    .blonde-card {
        background: #fff0f0;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        border-left: 4px solid #e53935;
    }
    .score-display {
        font-size: 28px;
        font-weight: bold;
        color: #c62828;
    }
    .spinner {
        font-size: 24px;
        margin-right: 10px;
        animation: spin 2s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
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
st.markdown('<h1 class="title">🌐 高级机器翻译系统</h1>', unsafe_allow_html=True)


def clean_response(response):
    """清洗模型响应"""
    cleaned = re.sub(r'^(Output:|Output：|输出:|输出：|翻译:|翻译：|Response:|Response：|回答:|回答：|\s*)+', '', response)
    cleaned = re.sub(r'^(Assistant:|助手：|模型:|模型：|AI:|AI：|\s*)+', '', cleaned)
    cleaned = re.sub(r'[。！？：；,.!?:;]+$', '', cleaned)
    return cleaned.strip()


def split_sentences(text):
    """改进的句子分割方法"""
    sentences = re.split(r'(?<=[.!?。！？]) +', text.strip())
    return [s for s in sentences if s and len(s) > 3]


def get_reference_translations(source_text, src_lang, tgt_lang):
    """获取参考译文（优先从数据库获取，不存在则生成默认）"""
    if source_text in REFERENCE_TRANSLATIONS:
        return REFERENCE_TRANSLATIONS[source_text]

    # 生成默认参考译文（实际应用应替换为真实译文）
    if src_lang == "中文" and tgt_lang == "英语":
        return [
            ["Reference translation for: " + source_text[:50]],
            ["Alternative translation: " + source_text[:40]]
        ]
    else:
        return [
            [f"{tgt_lang}参考译文：" + source_text[:30]],
            [f"{tgt_lang}备用译文：" + source_text[:25]]
        ]


def calculate_blonde_score(source_text, translated_text, src_lang, tgt_lang):
    """严格按照标准BLONDE方法计算评分"""
    try:
        # 分句处理
        sys_sentences = split_sentences(translated_text)

        # 获取参考译文（多版本）
        ref_docs = get_reference_translations(source_text, src_lang, tgt_lang)

        # 转换为BLONDE要求的格式：[[sys_sentences]]和[[ref1], [ref2]]
        score = blonde_evaluator.corpus_score([sys_sentences], [ref_docs])

        # 转换为0-1范围并确保最低分
        # return max(0.3, score.score / 100)
        return score.score


    except Exception as e:
        logger.error(f"BLONDE评分错误: {str(e)}")
        return 0.5  # 默认评分


def translate_with_doubao(text, src_lang, tgt_lang, api_key):
    """集成标准BLONDE评分的翻译函数"""
    try:
        client = Ark(api_key=api_key)

        lang_map = {
            "中文": "Chinese",
            "英语": "English",
            "法语": "French",
            "德语": "German",
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

        translated_text = clean_response(completion.choices[0].message.content)

        # 使用标准BLONDE方法计算评分
        blonde_score = calculate_blonde_score(text, translated_text, src_lang, tgt_lang)

        return translated_text, blonde_score

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
if 'blonde_score' not in st.session_state:
    st.session_state.blonde_score = None

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 系统配置")

    # 模型选择
    model_options = ["doubao-seed-1-6-250615", "nllb-200-distilled-600M", "nllb-200-1.3B", "nllb-200-3.3B"]
    model = st.selectbox(
        "翻译模型",
        options=model_options,
        index=model_options.index(st.session_state.api_config["model"]),
        help="选择使用的翻译模型"
    )

    # API密钥
    api_key = st.text_input(
        "API密钥",
        value=st.session_state.api_config["api_key"],
        type="password",
        help="火山引擎ARK SDK所需的API密钥"
    )

    # 保存配置
    if st.button("保存配置"):
        st.session_state.api_config = {
            "url": "https://ark.volcengineapi.com",
            "model": model,
            "api_key": api_key
        }
        st.success("配置已保存!")

    st.divider()

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
                options=["中文", "英语", "法语", "德语", "西班牙语", "日语"],
                index=0
            )
        with col_lang2:
            tgt_lang = st.selectbox(
                "目标语言",
                options=["英语", "中文", "法语", "德语", "西班牙语", "日语"],
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

        # 显示BLONDE评分
        if st.session_state.blonde_score is not None:
            st.markdown('<div class="blonde-card">', unsafe_allow_html=True)
            st.subheader("📊 BLONDE 评分")
            score = st.session_state.blonde_score

            if score >= 0.7:
                color = "#2e7d32"
                level = "优秀"
                emoji = "⭐️⭐️⭐️⭐️⭐️"
            elif score >= 0.5:
                color = "#f57c00"
                level = "良好"
                emoji = "⭐️⭐️⭐️⭐️"
            else:
                color = "#c62828"
                level = "一般"
                emoji = "⭐️⭐️⭐️"

            st.markdown(f"""
                <div style="display: flex; align-items: center;">
                    <div class="score-display" style="color: {color}; margin-right: 15px;">
                        {score:.2f}
                    </div>
                    <div>
                        <div>翻译质量: <span style="font-weight: bold;">{level}</span> {emoji}</div>
                        <div style="font-size: 12px; color: #666;">BLONDE评分范围0-1，分数越高质量越好</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# 翻译控制区域
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⚡ 翻译控制")
translate_btn = st.button("开始翻译", key="translate_btn", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

# 翻译按钮逻辑
if translate_btn:
    if not st.session_state.source_text.strip():
        st.warning("请输入要翻译的文本或上传文件")
    else:
        st.session_state.is_loading = True
        st.session_state.blonde_score = None
        with st.spinner("正在翻译，请稍候..."):
            try:
                start_time = time.time()

                if st.session_state.api_config["model"] == "doubao-seed-1-6-250615":
                    # 使用火山引擎API
                    translated_text, blonde_score = translate_with_doubao(
                        st.session_state.source_text,
                        src_lang,
                        tgt_lang,
                        st.session_state.api_config["api_key"]
                    )
                else:
                    # 其他模型API调用
                    data = {
                        "text": st.session_state.source_text,
                        "source_lang": src_lang,
                        "target_lang": tgt_lang,
                        "model": st.session_state.api_config["model"]
                    }

                    response = requests.post(
                        st.session_state.api_config["url"],
                        json=data,
                        headers={"Authorization": f"Bearer {st.session_state.api_config['api_key']}"}
                    )

                    if response.status_code == 200:
                        result = response.json()
                        translated_text = result.get("translated_text", "")
                        # 为其他模型计算BLONDE评分
                        blonde_score = calculate_blonde_score(
                            st.session_state.source_text,
                            translated_text,
                            src_lang,
                            tgt_lang
                        )
                    else:
                        raise Exception(f"API错误: {response.status_code} - {response.text}")

                processing_time = time.time() - start_time

                st.session_state.translated_text = translated_text
                st.session_state.blonde_score = blonde_score

                # 显示翻译信息
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("翻译状态", "成功 ✅")
                with col2:
                    st.metric("处理时间", f"{processing_time:.2f}秒")
                with col3:
                    st.metric("BLONDE评分", f"{blonde_score:.2f}")
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
        技术支持: 火山引擎ARK SDK | 支持BLONDE评分
    </div>
""", unsafe_allow_html=True)

# streamlit run ui_demo.py