import json
import pandas as pd
from langchain.llm import OpenAIWrapper
from langchain.chains import SingleTurnChain
from tqdm import tqdm

def load_dataset(data_path):
    with open(data_path, "r") as f:
        content = json.load(f)
    return content

def create_prompt(question, columns_used, sql, results, result_description):
    return f"""
    Task Type: Tabular Question Answering (Tabular QA)

    I am a tabular QA expert, and here is a task that involves extracting and reasoning over tabular data to answer a specific question: {question}

    Columns Used for answering the question: {columns_used}
    Executed SQL Statement: {sql}
    Query Results obtained after extracting the database by sql: {results}
    Description of Results: {result_description}

    Based on the information provided, please generate an answer to the question. Give the answer directly without analyze.
    The answer should be simple and clear, not textual information about the reasoning process.
    """

if __name__ == '__main__':
    api_key = '666'
    data_path = "/root/nlp_project/RAG/retrievaled_data(latin-1).json"
    res_path = "/root/nlp_project/RAG/baseline_rag_result.csv"

    # Initialize LangChain LLM Wrapper
    llm_wrapper = OpenAIWrapper(api_key=api_key)
    reasoning_chain = SingleTurnChain(llm_wrapper)

    # Load dataset
    val_data = load_dataset(data_path)

    # Process data
    responses = []
    ground_truth = []
    for val in val_data:
        question = val['question']
        columns_used = ", ".join(val['columns_used'])
        sql = val['sql']
        results = str(val['results'])
        result_description = val['result_description']
        answer = val['answer']
        prompt = create_prompt(question, columns_used, sql, results, result_description)

        # Generate response using LangChain
        response = reasoning_chain.run(prompt)
        responses.append(response)
        ground_truth.append(answer)

    # Save results
    df = pd.DataFrame({'response': responses, 'ground_truth': ground_truth})
    df.to_csv(res_path, index=False)
