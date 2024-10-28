import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 函数：加载模型和分词器
def get_tokenizer_model(model_directory):
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )
    return tokenizer, model

# 函数：生成SQL语句
def generate_sql(question, column_used, model, tokenizer):
    prompt = f"question: {question} column_used: {column_used}"
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
    attention_mask = inputs['attention_mask']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # 这里我们使用 greedy decoding 而不是 beam search
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=attention_mask,
        max_new_tokens=300,
        num_beams=1,
        early_stopping=False,
    )
    
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query

# 主函数：执行测试
def main():
    # 设置模型和分词器路径
    model_directory = '/root/nlp_project/sqlcoder-7b-2'  # 更新为你的本地模型路径
    tokenizer, model = get_tokenizer_model(model_directory)
    
    # 测试数据
    test_questions = [
        ("How many products are sold?", "products"),
        ("What is the total revenue?", "revenue"),
        ("Who are the top customers?", "customers")
    ]
    
    # 运行测试并打印结果
    for question, column_used in test_questions:
        sql_query = generate_sql(question, column_used, model, tokenizer)
        print(f"Question: {question}")
        print(f"Generated SQL: {sql_query}\n")

if __name__ == "__main__":
    main()
