# 多模态RAG智能问答助手 

#  pip install langchain langchain-openai chromadb sentence-transformers torch pillow clip fastapi uvicorn python-dotenv


import os
from typing import List, Tuple
import torch
import clip
from PIL import Image
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# ========== 1. CLIP 多模态嵌入 ==========
class CLIPEmbeddings(Embeddings):
    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        text_tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_text(text_tokens)
        return embeddings.cpu().numpy().tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
    
    def embed_image(self, image_path: str) -> List[float]:
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image)
        return embedding.cpu().numpy().tolist()[0]

# ========== 2. HyDE 检索器 ==========
class HyDERetriever:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        
    def _generate_hypothetical_doc(self, query: str) -> str:
        prompt = f"请针对以下问题，生成一段可能包含答案的文档片段。\n问题：{query}\n文档："
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    def retrieve(self, query: str, k: int = 4, use_hyde: bool = True):
        if use_hyde:
            hypothetical_doc = self._generate_hypothetical_doc(query)
            return self.vector_store.similarity_search(hypothetical_doc, k=k)
        return self.vector_store.similarity_search(query, k=k)

# ========== 3. 重排序器 ==========
class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
    def rerank(self, query: str, documents: List[str], top_k: int = 2):
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return scored[:top_k]

# ========== 4. 完整RAG流水线 ==========
class RAGPipeline:
    def __init__(self, persist_dir="./chroma_db"):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.embeddings = CLIPEmbeddings()
        self.vector_store = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
        self.retriever = HyDERetriever(self.vector_store, self.llm)
        self.reranker = Reranker()
        self.prompt = ChatPromptTemplate.from_template("""
基于以下信息回答问题。如果信息不足，请如实说不知道。

信息：{context}
问题：{question}
回答：""")
        
    def add_documents(self, texts: List[str]):
        docs = [Document(page_content=t) for t in texts]
        self.vector_store.add_documents(docs)
        
    def query(self, question: str):
        # 检索
        docs = self.retriever.retrieve(question, k=4)
        # 重排序
        contents = [d.page_content for d in docs]
        reranked = self.reranker.rerank(question, contents, top_k=2)
        context = "\n\n".join([contents[idx] for idx, _ in reranked])
        # 生成
        prompt_str = self.prompt.format(context=context, question=question)
        answer = self.llm.invoke(prompt_str)
        return answer.content

