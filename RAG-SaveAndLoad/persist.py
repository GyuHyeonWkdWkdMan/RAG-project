from langchain.text_splitter import RecursiveCharacterTextSplitter        #청크사이즈 셋팅에서 사용
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
#from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, UnstructuredPDFLoader

#문서 임베딩
persist_directory = './chroma'

#loader = PyPDFLoader('your_file')
loader = PyPDFDirectoryLoader("name_of_pdf_directory")
data = loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(data)

#임베딩 모델 변경
oembed = OllamaEmbeddings(model="bge-m3")
#oembed = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed, persist_directory=persist_directory)

#최신 버전의 크로마 라이브러리는 아래 함수를 사용하지 않아도 알아서 저장됨
#vectorstore.persist()


print(f"Document embeded and saved into {persist_directory}!")
