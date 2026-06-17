from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from src.chain import get_chain
from src.vectorstore import load_vectorstore

chain = get_chain()
vectorstore = load_vectorstore()

# Sample Q&A pairs about your CV
eval_questions = [
    "What programming languages do you know?",
    "What is your most recent work experience?",
    "What projects have you built?",
    "What is your educational background?",
    "Do you have experience with machine learning?",
    "What certifications do you hold?",
    "What MLOps tools do you use?",
    "What is your availability for new roles?"
]

# Reference answers (ground truth from the CV)
ground_truths = [
    "Python (primary), JavaScript, SQL, PHP",
    "Freelance Data Scientist & ML Engineer from 2020 to present — over 5 years of independent client work across finance, computer vision, data analytics, and institutional software.",
    "Projects include: Sovereign Ledger (financial management system for a health-worker cooperative), Indonesian Stock MLOps Platform (BUY/SELL signals for 45 blue-chip stocks), Multi-Stage Computer Vision Pipeline (YOLO/SSD + fine-grained classifier), CV RAG Chatbot (this project), and Makmur Grosir E-Commerce Image Scraper (Playwright-based scraper across 4 Indonesian platforms).",
    "Bachelor of Computer Science from Universitas Terbuka with a GPA of 3.36/4.0. Previously studied Computer Information Systems at UIN Syarif Hidayatullah Jakarta.",
    "Yes — over 6 years of experience including computer vision (YOLO, SSD, OpenCV, PyTorch, TensorFlow), MLOps (MLflow, Docker, Grafana), LLM/RAG applications (LangChain, FAISS, Groq), and full-stack financial software.",
    "CISDM from Cybertrend (2019), CIPMA from Cybertrend (2019), MTA from Microsoft (2018), Python Data Science from SanberCode (2020), CISCA from PASAS Institute (2017).",
    "MLflow for experiment tracking, Docker for reproducible environments, Grafana for monitoring, and FastAPI for model serving.",
    "Open to remote full-time, contract, or freelance engagements in ML Engineering, Data Science, MLOps, or Full-Stack Development. Available immediately."
]

answers, contexts = [], []
for q in eval_questions:
    result = chain.invoke({"input": q})
    # Extract answer from the result (RunnableWithMessageHistory returns dict with 'answer' key)
    if isinstance(result, dict) and "answer" in result:
        answers.append(result["answer"])
    elif isinstance(result, dict) and "output" in result:
        answers.append(result["output"])
    else:
        answers.append(str(result))
    
    # Extract context if present
    if isinstance(result, dict) and "context" in result:
        contexts.append([d.page_content for d in result["context"]])
    else:
        contexts.append([""])

dataset = Dataset.from_dict({
    "question": eval_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(results)
# Save for README badge
results.to_pandas().to_csv("evaluation_results.csv", index=False)
print("\n✅ Evaluation complete. Results saved to evaluation_results.csv")