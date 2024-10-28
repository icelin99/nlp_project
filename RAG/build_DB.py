import os
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Elasticsearch connection
es = Elasticsearch(hosts=["http://localhost:9200"], request_timeout=30)

def create_index(index_name, dataframe):
    # 创建索引并定义映射
    if not es.indices.exists(index=index_name):
        properties = {col: {'type': 'text'} for col in dataframe.columns} 
        es.indices.create(index=index_name, body={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": properties
            }
        })

def process_parquet_file(file_path, dataset_name):

    df = pd.read_parquet(file_path)

    # 创建对应的索引
    create_index(dataset_name, df)

    # 转换DataFrame为字典格式以便上传
    records = df.to_dict(orient='records')
    return records

def bulk_upload_to_es(index_name, records):
    actions = [
        {"_index": index_name, "_source": record}
        for record in records
    ]
    # 执行批量上传
    bulk(es, actions)

# 遍历指定目录下的所有Parquet文件
root_directory = "/root/nlp_project/databench/dataset"
for dataset_name in os.listdir(root_directory):
    dataset_path = os.path.join(root_directory, dataset_name, "all.parquet")
    if os.path.exists(dataset_path):
        records = process_parquet_file(dataset_path, dataset_name)
        bulk_upload_to_es(dataset_name, records)  # 确保索引名是已存在或已配置的

print("数据上传完成。")