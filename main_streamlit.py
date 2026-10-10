# 배포시 ---- (맨위 추가)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import init_chat_model

from langsmith import Client
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.callbacks.base import BaseCallbackHandler

import tempfile
import os

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory() # 임시폴더 생성
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)#임시폴더에서 업로드된 pdf로딩
    pages = loader.load_and_split()
    return pages

api_key = st.text_input('OpenAI_API_Key:',type='password')

st.title("ChatPDF🔎")
st.write("---") 

uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type=['pdf'])
if uploaded_file is not None:
  pages = pdf_to_document(uploaded_file)
  ## 분할
  text_splitter = RecursiveCharacterTextSplitter(
  # Set a really small chunk size, just to show.
  chunk_size=100, #각 청크의 최대 길이
  chunk_overlap=20, #인접한 청크사이의 중복영역, 문장이 끊기는 문제 해결 20글자 겹침
  length_function=len,#청크길이 측정하는 함수
  is_separator_regex=False,#단순한 문자열로 해석
)
  texts = text_splitter.split_documents(pages)

  ## 임베딩
  embeddings = OpenAIEmbeddings(
     model="text-embedding-3-large", 
     api_key= api_key
     )

  ## 벡터DB 설정 및 저장
  db = Chroma.from_documents(texts, embeddings) ##, persist_directory="./chroma_db"

  # 캐싱 문제 발생시 아래 코드를 2줄을 주석해제 하여서 사용(배포시 해당구간 사용)
  import chromadb
  chromadb.api.client.SharedSystemClient.clear_system_cache()

  ## Stream사용을 위한 클래스 선언
  class StreamHandler(BaseCallbackHandler):
      def __init__(self, container, initial_text=""):
          self.container = container
          self.text=initial_text
      def on_llm_new_token(self, token: str, **kwargs) -> None:
          self.text+=token
          self.container.markdown(self.text)

  ## user input 검색기능 구현
  st.header('PDF에게 질문하기')
  question = st.text_input('질문을 입력하세요')
  if st.button('답변 받기'):
     with st.spinner('Wait for it...'):
        st.write('답변을 생각중입니다.'
                  )
        ## 검색 기능 구현
        llm = init_chat_model("gpt-5.6-luna", api_key=api_key)

        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(), llm=llm
        )
        ## 프롬프트, llm 체인 실행
        client = Client()
        prompt = client.pull_prompt('rlm/rag-prompt',dangerously_pull_public_prompt=True)

        #검색결과 format
        def format_docs(docs):
            return '\n\n'.join(doc.page_content for doc in docs)

        # 출력공간 확보 stream 부분
        chat_box = st.empty()
        stream_handler = StreamHandler(chat_box)
            
        generate_llm = init_chat_model(model="gpt-5.5", openai_api_key=api_key, streaming=True, callbacks=[stream_handler])

        # 체인
        rag_chain = (
          {'context':retriever_from_llm | format_docs, "question":RunnablePassthrough()}  #입력값 그대로 사용
          | prompt
          | llm
          | StrOutputParser()
        )

        #실행
        result = rag_chain.invoke(question)
        # st.write(result)