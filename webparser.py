from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager		#실시간 답변 스트림에서 사용
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler		  #실시간 답변 스트림에서 사용
from langchain.text_splitter import RecursiveCharacterTextSplitter			#청크사이즈 셋팅에서 사용
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import Docx2txtLoader
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import WebBaseLoader
import bs4


# 실시간 답변 스트림 -> 질의응답 단의 출력문을 print(llm(question))로 설정해야함
'''
llm = Ollama(
    model="llama3.1:8b", callback_manager=CallbackManager([OutCallbackHandler()])
)
'''

# 답변을 한번에 출력 -> 질의응답 출력문을 print(llm.invoke(question)) 로 설정해야함
llm = Ollama(model="llama3.1:8b")

#웹 임베딩
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/article/437/0000378416",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
    header_template={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
    },
)
#loader = PyPDFLoader('airportshop3.pdf')
data = loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
oembed = OllamaEmbeddings(model="bge-m3")
#oembed = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)


template = """
다음과 같은 맥락을 사용하여 마지막 질문에 대답하십시오. 
당신은 웹에서 수집한 정보를 기반으로 사람들의 질문에 답변하는 챗봇입니다.
질문이 한국어로 되어 있으면, 한국어로 대답하세요. 질문이 다른 언어로 되어 있으면, 그 언어로 대답하세요.
묻는 질문에만 대답하고, 묻지 않은 정보에 대한 문장을 생성하지 마십시오.
모르는 정보에 대한 질문이 있다면 '모르겠습니다' 라고 답하시오.
{context}
질문: {question}
답변:"""



prompt = PromptTemplate(template=template, input_variables=["question"])

qachain=RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt})

#질의응답
ask = ''
while 1:
  ask = input('(끝내려면 \'종료\' 를 입력하세요)질문을 입력하세요: ')
  if ask=='종료':
    break

  
  answer=qachain.invoke(ask)
  #print(qachain.invoke(ask))
  print(answer['result'])

