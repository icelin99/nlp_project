"""
This file is for processing data which is obtained after retrival.
Making data into a format for LLM inference.
"""

import json

retrieval_data = "/root/nlp_project/RAG/retrievaled_data(latin-1).json"

prompt_format = """
Task Type: Tabular Question Answering (Tabular QA)

I am a tabular QA expert, and here is a task that involves extracting and reasoning over tabular data to answer a specific question.

Question: {question}

Columns Used: {columns_used}
Executed SQL Statement: {sql}
Query Results: {results}
Description of Results: {result_description}

Based on the information provided, please generate an answer to the question. Give the answer directly without analyze.
The answer should be simple and clear, not textual information about the reasoning process.
"""
