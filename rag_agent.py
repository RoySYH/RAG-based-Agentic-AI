from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import re
import os
from dotenv import load_dotenv, find_dotenv

# 載入 .env 檔案中的環境變數
load_dotenv(find_dotenv())

# 確認 Gemini API 密鑰已載入
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# 載入知識庫
loader = TextLoader("policy.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# 建立向量檢索
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

# 初始化LLM（使用 Gemini API）
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # 可改為 gemini-1.5-pro 若有權限
    temperature=0.3,
    max_output_tokens=150
)

# 時間衝突檢查
def check_time_conflict(time_slot):
    booked_slots = ["10:00 AM", "2:00 PM", "上午10點", "下午2點"]
    if time_slot in booked_slots:
        return False, "Suggested alternative times: 11:00 AM or 3:00 PM"
    return True, "is available"

# Agentic RAG查詢
def query_rag_agent(question):
    # 提取時間（支援中文和英文格式）
    time_pattern = r"(上午|下午)?\d{1,2}點|\d{1,2}:\d{2}\s*(AM|PM)"
    time_match = re.search(time_pattern, question)
    
    if time_match:
        time_slot = time_match.group(0)
        is_available, suggestion = check_time_conflict(time_slot)
        if not is_available:
            return f"Time slot {time_slot} is booked, {suggestion}"
        return f"Time slot {time_slot} {suggestion}"
    
    # 檢索並生成回答
    docs = db.similarity_search(question, k=2)
    context = " ".join([doc.page_content for doc in docs])
    prompt = f"Based on the following information, provide a concise answer in English:\n{context}\nQuestion: {question}\nAnswer:"
    response = llm.invoke(prompt).content
    # 後處理：移除提示部分
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    # 過濾無效內容（保留英文、數字、空格和基本標點）
    response = "".join(c for c in response if c.isalnum() or c.isspace() or c in ',.!?')
    # 若回答過短或無效，返回備用回答
    if len(response) < 5 or question.lower() in response.lower():
        return "Book through the system by selecting a date and time."
    return response

# 測試並儲存輸出
if __name__ == "__main__":
    questions = [
        "How to book a meeting room?",
        "Can I book meeting room A at 10:00 AM tomorrow?",
        "Is 11:00 AM available?"
    ]
    with open("output.txt", "w", encoding="utf-8") as f:
        for q in questions:
            answer = query_rag_agent(q)
            f.write(f"Q: {q}\nA: {answer}\n\n")
            print(f"Q: {q}\nA: {answer}\n")