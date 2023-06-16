import os

import streamlit_chat
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS
import sentence_transformers

from tool.chatllm import ChatLLM
from tool.chinese_text_splitter import ChineseTextSplitter
from tool.pandora_with_langchain_llm import *
from typing import Callable, Optional

chain: Optional[Callable] = None

import torch
LLM_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
EMBEDDING_DEVICE = "cpu"
MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'model_cache')

st.set_page_config(
    page_title="学习助手 演示",
    page_icon=":robot:"
)

class KnowledgeBasedChatLLM:

    llm: object = None
    embeddings: object = None

    def init_model_config(self,model_type):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="/data/tool/text2vec-base-chinese", )
        self.embeddings.client = sentence_transformers.SentenceTransformer(
            self.embeddings.model_name,
            device=EMBEDDING_DEVICE,
            cache_folder=os.path.join(MODEL_CACHE_PATH,
                                      self.embeddings.model_name))

        if model_type == 'chatglm':
            self.llm = ChatLLM()

            self.llm.model_type = 'chatglm'
            self.llm.model_name_or_path = "./chatglm-6b"
            self.llm.load_llm(llm_device=LLM_DEVICE)
        elif model_type == 'gpt':
            self.llm = CustomChatGPT(base_url="http://192.168.12.24:8008/")


    def init_knowledge_vector_store(self, filepath):

        docs = self.load_file(filepath)
        vector_store = FAISS.from_documents(docs, self.embeddings)
        vector_store.save_local('faiss_index')
        return vector_store

    def get_knowledge_based_answer(self,model_type,
                                   query,
                                   top_k: int = 6,
                                   history_len: int = 3,
                                   temperature: float = 0.01,
                                   top_p: float = 0.1,
                                   history=[]):
        self.history_len = history_len
        self.top_k = top_k

        prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
            如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。

            已知内容:
            {context}

            问题:
            {question}"""

        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])

        vector_store = FAISS.load_local('faiss_index', self.embeddings)

        if model_type == 'chatglm':
            self.llm.temperature = temperature
            self.llm.top_p = top_p
            self.llm.history = history[
                               -self.history_len:] if self.history_len > 0 else []
            knowledge_chain = RetrievalQA.from_llm(
                llm=self.llm,
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": self.top_k}),
                prompt=prompt)
            knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
                input_variables=["page_content"], template="{page_content}")

            knowledge_chain.return_source_documents = True

            result = knowledge_chain({"query": query})
        elif model_type == 'gpt':
            knowledge_chain = RetrievalQA.from_llm(
                llm=self.llm,
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": self.top_k}),
                prompt=prompt)
            knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
                input_variables=["page_content"], template="{page_content}")

            knowledge_chain.return_source_documents = True
            result = knowledge_chain({"query": query})
            print(result)
        return result

    def load_file(self, filepath):
        if filepath.lower().endswith(".md"):
            loader = UnstructuredFileLoader(filepath, mode="elements")
            docs = loader.load()
        elif filepath.lower().endswith(".pdf"):
            loader = UnstructuredFileLoader(filepath)
            textsplitter = ChineseTextSplitter(pdf=True)
            docs = loader.load_and_split(textsplitter)
        else:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False)
            docs = loader.load_and_split(text_splitter=textsplitter)
        return docs

@st.cache_resource
def init_model(model_type):
    print("init_model")
    knowladge_based_chat_llm = KnowledgeBasedChatLLM()
    knowladge_based_chat_llm.init_model_config(model_type= model_type)
    return knowladge_based_chat_llm


MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2

def predict(model_type,knowladge_based_chat_llm,input,
            top_k,
            history_len,
            temperature,
            top_p,
            history=None):

    if history == None:
        history = []

    with container:

        if len(history) > 0:
            if len(history) > MAX_BOXES:
                history = history[-MAX_TURNS:]
            for i, (query, response) in enumerate(history):
                streamlit_chat.message(query, avatar_style="big-smile", key=str(i) + "_user")
                streamlit_chat.message(response, avatar_style="bottts", key=str(i))
        streamlit_chat.message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        resp = knowladge_based_chat_llm.get_knowledge_based_answer(
            model_type=model_type,
            query=input,
            top_k=top_k,
            history_len=history_len,
            temperature=temperature,
            top_p=top_p,
            history=history)
        history.append((input, resp['result']))
        with st.empty():
            for query, response in history:
                st.write(response)

    return history

def init_vector_store(file_path,knowladge_based_chat_llm):
    vector_store = knowladge_based_chat_llm.init_knowledge_vector_store(
        file_path)
    return vector_store

container = st.container()

prompt_text = st.text_area(label="用户命令输入",
            height = 100,
            placeholder="请在这儿输入您的命令")

model_type = st.sidebar.radio(
    "模型类型",
    ('gpt','chatglm'))

# model_type = 'chatglm'
top_k = st.sidebar.slider(
    'top_k', 1, 10, 6, step=1
)

top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.6, step=0.01
)

history_len = st.sidebar.slider(
    'history_len', 0, 5, 3, step=1
)

temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.95, step=0.01
)

data = st.sidebar.file_uploader("上传文本文件", type=["txt", "md", "pdf", "docx"])
init_vs = st.sidebar.button("知识库文件向量化")
knowladge_based_chat_llm = init_model(model_type=model_type)

if init_vs:
    if data is not None:
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        filename = data.name
        with open(os.path.join("uploads", filename), "wb") as f:
            f.write(data.getbuffer())
        init_vector_store(os.path.join("uploads", filename),knowladge_based_chat_llm)
        st.sidebar.write("知识库向量化完成")
    else:
        st.sidebar.write("请上传文本文件")

st.markdown("""提醒：使用时请先上传自己的知识文件，并且文件中不含某些特殊字符，否则将返回error.""")
if 'state' not in st.session_state:
    st.session_state['state'] = []
if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        st.session_state["state"] = predict(model_type,knowladge_based_chat_llm,prompt_text, top_k,history_len, temperature, top_p, st.session_state["state"])
