import sqlite3
import json

# Connect to the SQLite database
conn = sqlite3.connect('/root/nlp_project/databench/dataset.sqlite3')

# Create a cursor object using the cursor() method
cursor = conn.cursor()

sql = """
  SELECT MAX("retweets<gx:number>") AS "max_retweets", COUNT(*) AS "num_tweets" FROM "054_Joe" GROUP BY "max_retweets" ORDER BY "max_retweets" DESC LIMIT 2;
"""
cursor.execute(sql)
   
results = cursor.fetchall()
  

cursor.close()
print(f"result: {results}")