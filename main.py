__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from dotenv import load_dotenv
# load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
# from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import tempfile
import os

## 메모리에 temp directory를 생성하여 파일을 업로드 후 읽어들이는 함수
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

## 제목
st.title("ChatPDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요.", type=['pdf'])
st.write("---")

# 업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    

    # 1. Loader : 페이지 로더 후 페이지별 나누기 
    # loader = PyPDFLoader("test.pdf")
    # pages = loader.load_and_split() 
    # print(pages[0])


    # 2. Split : 문자열별 나누기
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=100, # 몇 글자 단위로 쪼갤 것인지
        chunk_overlap=20,   # 정확하게 chunk size로 자르면 문맥이 끊길 수 있으니 앞뒤로 문맥이 끊어지기 않게 버퍼 사이즈를 줌
        length_function=len,    # 길이를 결정하는 함수
        is_separator_regex=False,   # 정규 표현식으로 자를 것인지
    )

    texts = text_splitter.split_documents(pages)

    # print(texts[0])

    # 3-1. Embedding 처리
    embeddings_model = OpenAIEmbeddings()

    # embeddings = embeddings_model.embed_documents(
    #     texts
    # )
    # len(embeddings), len(embeddings[0])

    # 3-2. load it into Chroma
    ## 해당 설정의 경우 벡터 DB가 in-memory 기반으로 저장된다.
    db = Chroma.from_documents(texts, embeddings_model)
    ## 해당 옵션을 설정할 경우 벡터 DB를 해당 폴더에 영구적으로 저장하여 사용할 수 있다.
    # db = Chroma.from_documents(texts, embeddings_model, persist_directory="./chroma_db")

    # 4. Question : 질문을 Vector화 했을 때 Vector DB에서 연관된 데이터와 가까운(관련 있는) 데이터를 가져옴
    # question = "LangChain 이 뭐야?"
    # llm = ChatOpenAI(temperature=0) # temperature : 언어 생성 모델에서 생성된 텍스트의 다양성(degree of diversity)을 조절하는 하이퍼파라미터
    # retriever_from_llm = MultiQueryRetriever.from_llm(
    #     retriever=db.as_retriever(), llm=llm    # Vector DB와 LLM 설정
    # )

    # docs = retriever_from_llm.get_relevant_documents(query=question)
    # print(len(docs))
    # print(docs)
    
    #Question
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')
    
    if st.button("질문하기"):
        with st.spinner('답변 중...'):   

            template = """Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Use three sentences maximum and keep the answer as concise as possible.
            Always say "thanks for asking!" at the end of the answer.

            {context}

            Question: {question}

            Helpful Answer:"""
            custom_rag_prompt = PromptTemplate.from_template(template)

            llm = ChatOpenAI(temperature=0) # temperature : 언어 생성 모델에서 생성된 텍스트의 다양성(degree of diversity)을 조절하는 하이퍼파라미터

            rag_chain = (
                {"context": db.as_retriever(), "question": RunnablePassthrough()}
                | custom_rag_prompt
                | llm
                | StrOutputParser()
            )

            result = rag_chain.invoke(question)

            st.write(result)

