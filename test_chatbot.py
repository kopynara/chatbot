# ch07_pdf_chatbot.py

import gradio as gr
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 📖 PDF 로딩 및 분할
loader = PyPDFLoader("FinanceTerms_2023_700.pdf")
texts = loader.load_and_split()
print(f"총 문서 청크 개수: {len(texts)}")

# 🔎 로컬 임베딩 클래스 정의
class LocalEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"로컬 임베딩 모델 로딩 중: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# ✅ 로컬 임베딩 사용
embedding = LocalEmbeddings()
vectordb = Chroma.from_documents(texts, embedding=embedding)
print(f"벡터DB 저장 완료. 청크 개수: {vectordb._collection.count()}")

# 📝 프롬프트 템플릿
template = """
당신은 한국은행에서 만든 금융 용어를 설명해주는 금융챗봇입니다.
안상준 개발자가 제작했으며, 주어진 검색 결과(context)를 바탕으로만 답변하세요.

Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# 🤖 로컬 LLM (DistilGPT2)
local_llm = pipeline("text-generation", model="distilgpt2")

# 💬 챗봇 함수 (로컬 RAG)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

def get_chatbot_response(input_text):
    context_docs = retriever.get_relevant_documents(input_text)
    context = "\n".join([doc.page_content for doc in context_docs])
    prompt_text = prompt.format(context=context, question=input_text)
    response = local_llm(prompt_text, max_length=300, do_sample=True)[0]["generated_text"]
    return response

# 🖥️ Gradio UI 구성
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="경제금융용어 챗봇 (로컬)", type="messages")
    msg = gr.Textbox(label="질문을 입력하세요!")
    clear = gr.Button("대화 초기화")

    def respond(message, chat_history):
        bot_message = get_chatbot_response(message)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

# 🚀 실행 (외부 접속 가능)
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
