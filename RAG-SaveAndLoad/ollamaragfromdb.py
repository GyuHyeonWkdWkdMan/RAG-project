from langchain.callbacks.manager import CallbackManager    #실시간 답변 스트림에서 사용
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler      #실시간 답변 스트림에서 사용
from langchain_community.embeddings import OllamaEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate



# 실시간 답변 스트림 -> 질의응답 단의 출력문을 print(llm(question))로 설정해야함
'''
llm = Ollama(
    model="llama3.1:8b", callback_manager=CallbackManager([OutCallbackHandler()])
)
'''

# 답변을 한번에 출력 -> 질의응답 출력문을 print(llm.invoke(question)) 로 설정해야함
llm = Ollama(model="llama3.1:8b")


#문서 임베딩
persist_directory = './chroma'

#임베딩 모델 변경
oembed = OllamaEmbeddings(model="bge-m3")
#oembed = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(embedding_function=oembed, persist_directory=persist_directory)

template = """
다음과 같은 맥락을 사용하여 마지막 질문에 대답하십시오.
당신은 농식품 문서에서 정보를 수집해 제공하는 챗봇입니다.
모르는 정보에 대한 질문이 있다면 '모르겠습니다' 라고 답하시오.
{context}
질문: {question}
답변:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
qachain=RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever(),chain_type_kwargs={"prompt": prompt})

#질의응답
ask = ''
while 1:
  ask = input('(끝내려면 \'종료\' 를 입력하세요)질문을 입력하세요: ')
  if ask=='종료':
        break

 
  answer=qachain.invoke(ask)
  #print(qachain.invoke(ask))
  print(answer['result'])


