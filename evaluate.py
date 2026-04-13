from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from rag_pipeline import get_chain, load_vectorstore

chain = get_chain()
vectorstore = load_vectorstore()

# Sample Q&A pairs about your CV — write 8-10 of these
eval_questions = [
    "What programming languages do you know?",
    "What is your most recent work experience?",
    "What projects have you built?",
    "What is your educational background?",
    "Do you have experience with machine learning?"
]

# Reference answers (ground truth — write what's actually in your CV)
ground_truths = [
    "Python, JavaScript, SQL...",   # fill from your actual CV
    "Software Engineer at ...",
    "Built a recommendation system...",
    "Bachelor's degree in...",
    "Yes, including NLP and computer vision..."
]

answers, contexts = [], []
for q in eval_questions:
    result = chain.invoke({"question": q})
    answers.append(result["answer"])
    contexts.append([d.page_content for d in result["source_documents"]])

dataset = Dataset.from_dict({
    "question": eval_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
print(results)
# Save for README badge
results.to_pandas().to_csv("evaluation_results.csv", index=False)