# flake8: noqa
import json
import os
import sys
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import asyncio
from functools import partial

import chainlit as cl
import chainlit.data as cl_data
from google.cloud import bigquery as bq
from loguru import logger

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(PARENT_DIR)

from fact_structured.dbconnector.bqconnector import BigQueryConnector
from fact_structured.handler.eda import EDAHandler
from fact_structured.handler.multi_table import MultiTableHandler
from fact_structured.tools.multi_table import MultiTableTool
# from multi_table import MultiTableHandler
from fact_structured.llm.fordllm import FordLLM
from fact_structured.utils.helpers import encode_image, load_json

from ui.cl_datalayer import BQFeedback, BQLogger
from ui.config import ChatConfig
from ui.utils import (
    fetch_allowed_models,
    get_chart_filename,
    raise_exception,
    store_logs,
)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
import pandas as pd
from google.cloud import secretmanager
from fordllm.utils import TokenFetcher
import json, re

#  "output": "WITH \nPrice_Ranged_Parsed AS (\n  SELECT\n    model,\n    volume,\n    price_range_en,\n    CASE\n      WHEN price_range_en LIKE '%-%' THEN\n        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\\d+\\.?\\d*)(?:k|M)') AS FLOAT64) *\n        CASE\n          WHEN price_range_en LIKE '%k-%k%' THEN 1000\n          WHEN price_range_en LIKE '%M-%M%' THEN 1000000\n          ELSE NULL\n        END\n      WHEN price_range_en LIKE '>%' THEN\n        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^>(\\d+\\.?\\d*)(?:k|M)') AS FLOAT64) *\n        CASE\n          WHEN price_range_en LIKE '%k' THEN 1000\n          WHEN price_range_en LIKE '%M' THEN 1000000\n          ELSE NULL\n        END\n      WHEN price_range_en LIKE '%k' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\\d+)k') AS INT64) * 1000\n      WHEN price_range_en LIKE '%M' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\\d+\\.?\\d*)M') AS FLOAT64) * 1000000\n      ELSE NULL\n    END AS parsed_start_price,\n    CASE\n      WHEN price_range_en LIKE '%-%' THEN\n        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'-(\\d+\\.?\\d*)(?:k|M)$') AS FLOAT64) *\n        CASE\n          WHEN price_range_en LIKE '%k-%k%' THEN 1000\n          WHEN price_range_en LIKE '%M-%M%' THEN 1000000\n          ELSE NULL\n        END\n      WHEN price_range_en LIKE '>%' THEN 999999999999.0\n      WHEN price_range_en LIKE '%+%' THEN 999999999999.0\n      WHEN price_range_en LIKE '%k' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\\d+)k') AS INT64) * 1000\n      WHEN price_range_en LIKE '%M' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\\d+\\.?\\d*)M') AS FLOAT64) * 1000000\n      ELSE NULL\n    END AS parsed_end_price\n  FROM\n    `ford-f3ff981824ba0ca90a815557.china_sales.china_vehicle_sales`\n  WHERE\n    year = 2024\n    \n),\nEscape_Sales AS (\n  SELECT\n    SUM(volume) AS escape_volume\n  FROM\n    Price_Ranged_Parsed\n  WHERE\n    parsed_start_price IS NOT NULL\n    AND parsed_end_price IS NOT NULL\n    AND parsed_start_price < 200000\n    AND parsed_end_price > 150000\n    AND LOWER(model) = LOWER('ford escape')\n),\nTotal_Sales AS (\n  SELECT\n    SUM(volume) AS total_volume\n  FROM\n    Price_Ranged_Parsed\n  WHERE\n    parsed_start_price IS NOT NULL\n    AND parsed_end_price IS NOT NULL\n    AND parsed_start_price < 200000\n    AND parsed_end_price > 150000\n)\nSELECT\n  escape_volume AS Escape_Volume,\n  total_volume AS Total_Volume,\n  ROUND(SAFE_DIVIDE(escape_volume, total_volume) * 100, 2) AS Escape_Share_Percentage\nFROM\n  Escape_Sales\nCROSS JOIN\n  Total_Sales\n"



# Patch for LLM Clean Output
def patched_clean(self, raw_output: str) -> str:
    env   = json.loads(raw_output)
    text  = env["choices"][0]["message"]["content"].strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, 1, flags=re.I)
        text = re.sub(r"\s*```$", "", text, 1)
    logger.info(f"Inside New Implementation: {text}")
    try:                    # try to parse once
        parsed = json.loads(text)
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        return json.dumps(text,   ensure_ascii=False)

MultiTableTool.clean_llm_output = patched_clean
# _original_qv = MultiTableTool.query_validation
decorated_qv = MultiTableTool.query_validation     # wrapper (kept)
orig_inner    = decorated_qv.__wrapped__

# Patch for query_validation
def fixed_inner(self, *args, **kwargs):
    """
    Runs the original query validation logic and make sure to return tuple for which is result,payload for wrapper




    Returns: tuple

    """

    result = orig_inner(self, *args, **kwargs)


    if result is None:
        return {}, {}
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


decorated_qv.__wrapped__ = fixed_inner



chat_config = ChatConfig()
feedback_table = f"china_sales_feedback"
logger_table = f"china_sales_logger"
cl_data._data_layer = BQFeedback(
    project_id=chat_config.PROJECT_ID,
    logger_db=chat_config.CHAINLIT_LOGGER_BQ_DB,
    feedback_table=feedback_table,
)
os.environ['http_proxy'] = 'http://internet.ford.com:83'
os.environ['https_proxy'] = 'http://internet.ford.com:83'
os.environ['HTTP_PROXY'] = 'http://internet.ford.com:83'
os.environ['HTTPS_PROXY'] = 'http://internet.ford.com:83'
os.environ['no_proxy'] = 'localhost,127.0.0.1,.ford.com,.local,.testing,.internal,192.168.0.0/16'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,.ford.com,.local,.testing,.internal,192.168.0.0/16'

model_list = [
    "claude-3.5-sonnet",
    "gpt-4.1",
    "gemini-2.0-flash-001",
    "meta-llama-3.1-405b",
    "o1-preview",
    "o1-mini",
    "meta-llama-3.3-70b",
    "gemini-1.5-pro",
    "qwen-2.5-coder-32b",
    "deepseek-r1-distill-llama-70b",
    "gemini-1.5-flash",
    "meta-llama-3.2-11b",
    "o3-mini",
    "gpt-4.1-mini",
]

fetch_allowed_models_list = list(fetch_allowed_models(chat_config.METADATA_ENDPOINT))

connector = BigQueryConnector(project_id=chat_config.PROJECT_ID)

bq_logger = BQLogger(
    project_id=chat_config.PROJECT_ID,
    logger_db=chat_config.CHAINLIT_LOGGER_BQ_DB,
    logger_table=logger_table,
)
unique_folder_name = str(uuid.uuid4())
curr_dir = os.getcwd()
local_dir = os.path.join(curr_dir, "data", unique_folder_name)
os.makedirs(local_dir, exist_ok=True)


def access_secrets(secret_id,version="1"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/ford-f3ff981824ba0ca90a815557/secrets/{secret_id}/versions/{version}"
    response = client.access_secret_version(name=name)
    response = response.payload.data.decode('UTF-8').replace("\r", "").replace("\n", "")
    return response
# Initialize bq tool handler with BigQuery client and hardcoded vector db for now which we can make configurable later
bq = MultiTableHandler(
    client=connector,
    project_id=chat_config.PROJECT_ID,
    dataset_id=chat_config.BQ_DATASET,
    vector_db="FAISS",
)
dialect = bq.dialect

dataset_names = [chat_config.BQ_DATASET]


SYSTEM_PROMPT = """You are a SQL expert. Task is to transform natural question to SQL query. Use the source context provided to answer the users question. 
+If you don't know the answer, just say that you don't know, don't try to make up an answer.

** Important Instructions:**
1. **PV Definition**: Expansion of PV is Passenger Vehicle. To get the details of PV, filter for 
```
WHERE NOT (LOWER(segment) LIKE '%commercial%' or LOWER(segment) LIKE '%pickup%')
```
2. ** CV Definition**: Expansion of CV is Commercial Vehicle. To get the details of CV, filter for 
```
WHERE (LOWER(segment) LIKE '%commercial%' or LOWER(segment) LIKE '%pickup%')
``` 
3. **Premium Definition**: Segment column holds information about Premium segment. To get the details of Premium Segment, filter for 
```
LOWER(segment) LIKE '%premium%'
```
4. **ICE Details**: `ev_flag` column holds information about ICE vehicles. To get the details of ICE vehicles, Filter for 
```
LOWER(ev_flag)='null'
```
5. **NEV Definition:** Expansion of NEV is New Energy Vehicle. Its combination of EV,EREV,PHEV and FCEV. To get the details of NEV, Filter for 
```
LOWER(ev_flag) IN ('ev','phev','erev','fcev')
```
6. ** Domestic Brands in China:** `vehicledomestic` column holds information about whether brand is Local in China, if the value is domestic then its a chinese brand, else Global brand. Use this information to classify chinese brand or global brand
7. **Import/Domestic Vehicles: ** `datatype` columns holds information about whether vehicle production happens within china, if its within china then its considered as domestic, else import.
8 ** Alias: ** When using columns from with clause views, DO not use it directly inside ORDER BY CLAUSE, FIRST ALIAS IT AND THEN USE THE ALIASED NAME IN ORDER BY CLAUSE.
9 **Average Transaction Price/Transaction Price Formula:** SUM(avg_tp_cny_weight * volume_with_tp)/SUM(volume_with_tp)
10 ** JSON Considerartion :** When you return JSON every back-slash inside a string must itself be
escaped (use \\)
"""
refresh_browser = False
async def setup_vector_store_granularity(embeddings):
    df = pd.read_csv("Granularity_Identification.csv")
    df_groupby = df.groupby(by=['Table '])
    tables_list = []
    for table_name, table_df in df_groupby:
        column_table = []
        for index, row in table_df.iterrows():
            column = {row["Column"]: row["Unique Values"]}
            column_table.append(column)
        cleaned_table_name = str(table_name).replace("(", "").replace(")", "").replace(",", "").replace("'", "")
        tables_dict = {cleaned_table_name: column_table}
        tables_list.append(tables_dict)

    table_lookup = {}
    docs=[]
    for tbl in tables_list:
        table_name = list(tbl.keys())[0]
        columns = tbl[table_name]
        table_lookup[table_name] = columns
    tables = df['Table '].unique().tolist()
    for table_name in tables:
        column_granularity = [columns for columns in table_lookup[table_name]]
        col_text = [f"{c}:{v}" for col in column_granularity for c, v in col.items()]
        for col in column_granularity:
            for c, v in col.items():
                column_bm25 = f"{c}:{v}"
                docs.append(Document(page_content=column_bm25,
                                     metadata={"table_name": table_name}
                                     )
                            )

        # print(col_text)
        # embeddings=cl.user_session.get("embeddings")
        vectorstore = FAISS.from_texts(col_text, embeddings)
        vectorstore.save_local(f"granularity/{table_name}_faiss_index")

        logger.info(f"FAISS index created and saved for table: {table_name}")
    return vectorstore,docs


async def query_vector_granularity_store(table_name, query, k, docs):
    table = ""
    if "ford-f3ff981824ba0ca90a815557" in table_name:
        table = table_name.split(".")[2]
    elif "china_sales" in table_name:
        table = table_name.split(".")[1]
    else:
        table=table_name
    vectorstore = FAISS.load_local(f"granularity/{table}_faiss_index", cl.user_session.get("embeddings"),
                                   allow_dangerous_deserialization=True)
    print(f"Table Name:{table}")
    related_docs = []
    for i in docs:
        j = i.metadata
        if j['table_name'] == table:
            related_docs.append(i)
    print(f"**********Related Documents",related_docs)

    # results = vectorstore.similarity_search(query, k=k)
    similarity_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results=[]
    try:
        bm25_retriever = BM25Retriever.from_documents(related_docs)
        bm25_retriever.k = k
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, similarity_retriever],
            weights=[0.6, 0.4],
            # retriever_search_kwargs={"k": k}
        )
        results = ensemble_retriever.get_relevant_documents(query)
        print(f"*********Inside query_vector_store*****:{results}")
    except Exception as e:
        logger.info(f"Exception inside query vector:{e}")
        if 'invalid' in str(e).lower() or 'unauthorized' in str(e).lower():
            cl.user_session.set("refresh_browser",True)
    return results


# loading the fewshot examples
# fewshot_json_path = os.path.join(PARENT_DIR, "ui/templates", "fewshot_multi_table.json")
fewshot_json_path="fewshot_multi_table.json"
examples = load_json(fewshot_json_path)
examples_updated = []
for each in examples:
    output = each["output"].format(project_id=chat_config.PROJECT_ID)
    examples_updated.append({"input": each["input"], "output": output})

for example in examples_updated:
    SYSTEM_PROMPT += f'Input: "{example["input"]}"\nOutput: "{example["output"]}"\n'

# provide additional prompt instruction for bigquery tool
additional_sql_prompt = """
Use the correct project id, dataset name while querying
project id: `ford-f3ff981824ba0ca90a815557.china_sales`
dataset_name: `china_sales`
Example Big Query: SELECT model FROM `ford-f3ff981824ba0ca90a815557.china_sales.china_vehicle_sales`

**Special Questions involving priceband**
1. Never include the range of left and right side limit in WHERE CLAUSE and follow below query strategy to parse the price_Range_en column when asked for certain range of priceband.
Example: Share of MTU within 200k-250k in 2024 
```
WITH 
Price_Ranged_Parsed AS (

SELECT
    segment,
    volume,
    price_range_en,
    
    CASE
      
      WHEN price_range_en LIKE '%-%' THEN
        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+\.?\d*)(?:k|M)') AS FLOAT64) *
        CASE
          WHEN price_range_en LIKE '%k-%k%' THEN 1000
          WHEN price_range_en LIKE '%M-%M%' THEN 1000000
          ELSE NULL
        END
      WHEN price_range_en LIKE '>%' THEN
        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^>(\d+\.?\d*)(?:k|M)') AS FLOAT64) *
        CASE
          WHEN price_range_en LIKE '%k' THEN 1000
          WHEN price_range_en LIKE '%M' THEN 1000000
          ELSE NULL
        END
      
      WHEN price_range_en LIKE '%k' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+)k') AS INT64) * 1000
      WHEN price_range_en LIKE '%M' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+\.?\d*)M') AS FLOAT64) * 1000000
      ELSE NULL 
    END AS parsed_start_price,

    
    CASE
      
      WHEN price_range_en LIKE '%-%' THEN
        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'-(\d+\.?\d*)(?:k|M)$') AS FLOAT64) *
        CASE
          WHEN price_range_en LIKE '%k-%k%' THEN 1000
          WHEN price_range_en LIKE '%M-%M%' THEN 1000000
          ELSE NULL
        END
      -- Case for values like '>50k' or '>0.5M': set upper bound to a very large number (effectively infinity)
      WHEN price_range_en LIKE '>%' THEN 999999999999.0
      -- Case for single exact values like '200k' or '0.5M': start and end are the same
      WHEN price_range_en LIKE '%k' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+)k') AS INT64) * 1000
      WHEN price_range_en LIKE '%M' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+\.?\d*)M') AS FLOAT64) * 1000000
      ELSE NULL -- Handles any other unparsed formats
    END AS parsed_end_price
  FROM
    `ford-f3ff981824ba0ca90a815557.china_sales.china_vehicle_sales`
  
  
    WHERE year = 2024
  
)
,medium_traditional_utility_sales AS (
  SELECT price_range_en,SUM(volume) AS segment_volume
  FROM Price_Ranged_Parsed
  WHERE LOWER(segment) = LOWER('MEDIUM TRADITIONAL UTILITY')
  AND parsed_start_price IS NOT NULL AND parsed_end_price IS NOT NULL AND parsed_start_price <250000 AND parsed_end_price >200000
  GROUP BY price_range_en
),
  total_sales AS (
    SELECT SUM(volume) AS total_volume
    FROM Price_Ranged_Parsed
    WHERE parsed_start_price IS NOT NULL AND parsed_end_price IS NOT NULL AND parsed_start_price <300000 AND parsed_end_price >250000
    
  )
SELECT
  medium_traditional_utility_sales.segment_volume AS volume,
  ROUND(SAFE_DIVIDE(medium_traditional_utility_sales.segment_volume, total_sales.total_volume) * 100, 2) AS `Share Percentage`
FROM medium_traditional_utility_sales
CROSS JOIN total_sales 

```
**Priceband Share/Priceband Revenue Share:**
1. Share or Revenue Share of different Pricebands: Ratio of volume or revenue (msrp*volume) belonging to the identified granularity( Granularity can be `segment`,`ev_flag`,`model`,`brand`,`manufacturer`,`datatype,`vehicledomestic`,etc..) across each priceband to the total volume or revenue (volume*msrp)belonging to the identified granularity
   For Share use volume
   For revenue share use volume*msrp
   Example: Share of Lexus across different pricebands in 2024
   SQL Solution:
   ```
   WITH lexus_priceband_volume AS (
  SELECT
    price_range_en,
    SUM(volume) AS lexus_volume
  FROM `ford-f3ff981824ba0ca90a815557.china_sales.china_vehicle_sales`
  WHERE LOWER(brand) = LOWER('lexus') --identified granularity
    AND year = 2024
  GROUP BY price_range_en
),
all_brands_priceband_volume AS ( -- total sales of identified granularity
  SELECT
    SUM(volume) AS total_volume
  FROM `ford-f3ff981824ba0ca90a815557.china_sales.china_vehicle_sales`
  WHERE year = 2024
  AND LOWER(brand) = LOWER('lexus') -- identified granularity
  
)
SELECT
  lexus_priceband_volume.price_range_en AS price_band,
  lexus_priceband_volume.lexus_volume AS lexus_volume,
  all_brands_priceband_volume.total_volume AS total_volume,
  ROUND(SAFE_DIVIDE(lexus_priceband_volume.lexus_volume, all_brands_priceband_volume.total_volume) * 100, 2) AS lexus_volume_share_percentage
FROM lexus_priceband_volume
CROSS JOIN all_brands_priceband_volume
ORDER BY lexus_priceband_volume.lexus_volume DESC

   ```
Explanation: Here identified granularity is `brand` which is `lexus` and both numerator and denominator has brand filter, while numerator contains volume of lexus brand across each priceband and denominator has total volume of lexus priceband.
This applies for all granularities and its general to use this approach to get share across different pricebands.
2. **Share of particular granularity(it can be `segment`,`ev_flag`,`model`,`brand`,`manufacturer`,`datatype,`vehicledomestic`,etc..) within a priceband range**
    Formula: Ratio of volume or revenue (msrp*volume) for the identified granularity within the priceband range to the total volume or revenue (msrp*volume) within the priceband range
    Example: Share of MTU for the priceband range 250k-300k in 2024
    SQL Solution:
    ```
    WITH 
Price_Ranged_Parsed AS (

SELECT
    segment,
    volume,
    price_range_en,
    
    CASE
      
      WHEN price_range_en LIKE '%-%' THEN
        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+\.?\d*)(?:k|M)') AS FLOAT64) *
        CASE
          WHEN price_range_en LIKE '%k-%k%' THEN 1000
          WHEN price_range_en LIKE '%M-%M%' THEN 1000000
          ELSE NULL
        END
      WHEN price_range_en LIKE '>%' THEN
        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^>(\d+\.?\d*)(?:k|M)') AS FLOAT64) *
        CASE
          WHEN price_range_en LIKE '%k' THEN 1000
          WHEN price_range_en LIKE '%M' THEN 1000000
          ELSE NULL
        END
      
      WHEN price_range_en LIKE '%k' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+)k') AS INT64) * 1000
      WHEN price_range_en LIKE '%M' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+\.?\d*)M') AS FLOAT64) * 1000000
      ELSE NULL 
    END AS parsed_start_price,

    
    CASE
      
      WHEN price_range_en LIKE '%-%' THEN
        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'-(\d+\.?\d*)(?:k|M)$') AS FLOAT64) *
        CASE
          WHEN price_range_en LIKE '%k-%k%' THEN 1000
          WHEN price_range_en LIKE '%M-%M%' THEN 1000000
          ELSE NULL
        END
      -- Case for values like '>50k' or '>0.5M': set upper bound to a very large number (effectively infinity)
      WHEN price_range_en LIKE '>%' THEN 999999999999.0
      -- Case for single exact values like '200k' or '0.5M': start and end are the same
      WHEN price_range_en LIKE '%k' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+)k') AS INT64) * 1000
      WHEN price_range_en LIKE '%M' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+\.?\d*)M') AS FLOAT64) * 1000000
      ELSE NULL -- Handles any other unparsed formats
    END AS parsed_end_price
  FROM
    `ford-f3ff981824ba0ca90a815557.china_sales.china_vehicle_sales`
  
  
    WHERE year = 2024
  
)
,medium_traditional_utility_sales AS ( -- volume of identified granularity which is segment within priceband range
  SELECT price_range_en,SUM(volume) AS segment_volume
  FROM Price_Ranged_Parsed
  WHERE LOWER(segment) = LOWER('MEDIUM TRADITIONAL UTILITY')
  AND parsed_start_price IS NOT NULL AND parsed_end_price IS NOT NULL AND parsed_start_price <300000 AND parsed_end_price >250000
  GROUP BY price_range_en
),
  total_sales AS ( -- total volume in priceband range
    SELECT SUM(volume) AS total_volume
    FROM Price_Ranged_Parsed
    WHERE parsed_start_price IS NOT NULL AND parsed_end_price IS NOT NULL AND parsed_start_price <300000 AND parsed_end_price >250000
    
  )
SELECT
  medium_traditional_utility_sales.segment_volume AS volume,
  ROUND(SAFE_DIVIDE(medium_traditional_utility_sales.segment_volume, total_sales.total_volume) * 100, 2) AS `Share Percentage`
FROM medium_traditional_utility_sales
CROSS JOIN total_sales 

    ```
    
Explanation: Here identified granularity is `segment` which is `MEDIUM TRADITIONAL UTILITY` and both numerator and denominator has priceband filter, while numerator contains volume of Medium Traditional Utility segment within priceband range and denominator has total volume within priceband range.
This applies for all granularities and its general to use this approach to get share of different granularities within priceband range.
Example: Share of Escape within 150k-200k in 2024
SQL Solution:
```
WITH Price_Ranged_Parsed AS (
  SELECT
    model,
    volume,
    price_range_en,
    CASE
      WHEN price_range_en LIKE '%-%' THEN
        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+\.?\d*)(?:k|M)') AS FLOAT64) *
        CASE
          WHEN price_range_en LIKE '%k-%k%' THEN 1000
          WHEN price_range_en LIKE '%M-%M%' THEN 1000000
          ELSE NULL
        END
      WHEN price_range_en LIKE '>%' THEN
        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^>(\d+\.?\d*)(?:k|M)') AS FLOAT64) *
        CASE
          WHEN price_range_en LIKE '%k' THEN 1000
          WHEN price_range_en LIKE '%M' THEN 1000000
          ELSE NULL
        END
      WHEN price_range_en LIKE '%k' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+)k') AS INT64) * 1000
      WHEN price_range_en LIKE '%M' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+\.?\d*)M') AS FLOAT64) * 1000000
      ELSE NULL
    END AS parsed_start_price,
    CASE
      WHEN price_range_en LIKE '%-%' THEN
        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'-(\d+\.?\d*)(?:k|M)$') AS FLOAT64) *
        CASE
          WHEN price_range_en LIKE '%k-%k%' THEN 1000
          WHEN price_range_en LIKE '%M-%M%' THEN 1000000
          ELSE NULL
        END
      WHEN price_range_en LIKE '>%' THEN 999999999999.0
      WHEN price_range_en LIKE '%+%' THEN 999999999999.0
      WHEN price_range_en LIKE '%k' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+)k') AS INT64) * 1000
      WHEN price_range_en LIKE '%M' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+\.?\d*)M') AS FLOAT64) * 1000000
      ELSE NULL
    END AS parsed_end_price
  FROM
    `ford-f3ff981824ba0ca90a815557.china_sales.china_vehicle_sales`
  WHERE
    year = 2024
    
),
Escape_Sales AS (
  SELECT
    SUM(volume) AS escape_volume
  FROM
    Price_Ranged_Parsed
  WHERE
    parsed_start_price IS NOT NULL
    AND parsed_end_price IS NOT NULL
    AND parsed_start_price < 200000
    AND parsed_end_price > 150000
    AND LOWER(model) = LOWER('ford escape')
),
Total_Sales AS (
  SELECT
    SUM(volume) AS total_volume
  FROM
    Price_Ranged_Parsed
  WHERE
    parsed_start_price IS NOT NULL
    AND parsed_end_price IS NOT NULL
    AND parsed_start_price < 200000
    AND parsed_end_price > 150000
)
SELECT
  escape_volume AS Escape_Volume,
  total_volume AS Total_Volume,
  ROUND(SAFE_DIVIDE(escape_volume, total_volume) * 100, 2) AS Escape_Share_Percentage
FROM
  Escape_Sales
CROSS JOIN
  Total_Sales

```
Explanation: Here identified granularity is `model` which is `Ford Escape` and both numerator and denominator has priceband filter, while numerator contains volume of Ford Escape model within priceband range and denominator has total volume within priceband range.
This applies for all granularities and its general to use this approach to get share of different granularities within priceband range.
Example: Share of Lexus within priceband range 350k-400k in 2024
SQL Solution:
```
WITH 
Price_Ranged_Parsed AS (
  SELECT
    brand,
    volume,
    price_range_en,
    CASE
      WHEN price_range_en LIKE '%-%' THEN
        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+\.?\d*)(?:k|M)') AS FLOAT64) *
        CASE
          WHEN price_range_en LIKE '%k-%k%' THEN 1000
          WHEN price_range_en LIKE '%M-%M%' THEN 1000000
          ELSE NULL
        END
      WHEN price_range_en LIKE '>%' THEN
        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^>(\d+\.?\d*)(?:k|M)') AS FLOAT64) *
        CASE
          WHEN price_range_en LIKE '%k' THEN 1000
          WHEN price_range_en LIKE '%M' THEN 1000000
          ELSE NULL
        END
      WHEN price_range_en LIKE '%k' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+)k') AS INT64) * 1000
      WHEN price_range_en LIKE '%M' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+\.?\d*)M') AS FLOAT64) * 1000000
      ELSE NULL
    END AS parsed_start_price,
    CASE
      WHEN price_range_en LIKE '%-%' THEN
        SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'-(\d+\.?\d*)(?:k|M)$') AS FLOAT64) *
        CASE
          WHEN price_range_en LIKE '%k-%k%' THEN 1000
          WHEN price_range_en LIKE '%M-%M%' THEN 1000000
          ELSE NULL
        END
      WHEN price_range_en LIKE '>%' THEN 999999999999.0
      WHEN price_range_en LIKE '%+%' THEN 999999999999.0
      WHEN price_range_en LIKE '%k' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+)k') AS INT64) * 1000
      WHEN price_range_en LIKE '%M' THEN SAFE_CAST(REGEXP_EXTRACT(price_range_en, r'^(\d+\.?\d*)M') AS FLOAT64) * 1000000
      ELSE NULL
    END AS parsed_end_price
  FROM
    `ford-f3ff981824ba0ca90a815557.china_sales.china_vehicle_sales`
  WHERE
    year = 2024
    
),
lexus_priceband_volume AS ( -- volume of identified granularity which is segment within priceband range
  SELECT
    price_range_en,
    SUM(volume) AS lexus_volume
  FROM
    Price_Ranged_Parsed
  WHERE
    LOWER(brand) = LOWER('lexus')
    AND parsed_start_price IS NOT NULL
    AND parsed_end_price IS NOT NULL
    AND parsed_start_price < 400000
    AND parsed_end_price > 350000
    
  GROUP BY
    price_range_en
),
total_volume AS ( -- total volume in priceband, dont include identified granularity filter in denominator
  SELECT
    SUM(volume) AS total_volume
  FROM
    Price_Ranged_Parsed
  WHERE
    parsed_start_price IS NOT NULL
    AND parsed_end_price IS NOT NULL
    AND parsed_start_price < 400000
    AND parsed_end_price > 350000
)
SELECT
  lexus_priceband_volume.price_range_en AS price_band,
  lexus_priceband_volume.lexus_volume AS lexus_volume,
  total_lexus_volume.total_volume AS total_volume,
  ROUND(SAFE_DIVIDE(lexus_priceband_volume.lexus_volume, total_lexus_volume.total_volume) * 100, 2) AS lexus_volume_share_percentage
FROM
  lexus_priceband_volume
CROSS JOIN
  total_lexus_volume
ORDER BY
  lexus_priceband_volume.lexus_volume DESC
```
Explanation: Here identified granularity is `brand` which is `Lexus` and both numerator and denominator has priceband filter, while numerator contains volume of Lexus brand within priceband range and denominator has total volume within priceband range.
This applies for all granularities and its general to use this approach to get share of different granularities within priceband range
"""

# chat history disabled by default
chat_history = False


@cl.on_chat_start
async def start():
    msg = f"""Hi! I am powered by **FACT** *(Ford AI Chat Toolkit)*. I can assist you in transforming your queries to **{dialect}** SQL, generate answer, and suggest EDA for visualizations and insights.\
    \n`GPT-4o` is set as LLM. You can change this along with `dataset` of your choice from the Chat Settings <img src="public/chatsetting.png" alt="settings" width="25" height="25">.\
    For more details, please read the [README document](/readme)"""
    await cl.Message(content=msg).send()
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="dataset",
                label=f"{dialect} Datasets",
                values=dataset_names,
                initial_index=0,
                description="Select the dataset you want to query.",
            ),
            cl.input_widget.Select(
                id="model",
                label="Model",
                values=fetch_allowed_models(chat_config.METADATA_ENDPOINT)
                or model_list,
                initial_value="gpt-4.1-mini",
                description="Select the LLM Model for generating the answer.",
            ),
            cl.input_widget.Select(
                id="chat history",
                label="Chat History",
                values=["Enable", "Disable"],
                initial_value="Disable",
                description="Choose to enable or disable chat history.",
            ),
            cl.input_widget.Slider(
                id="temp",
                label="Temperature",
                initial=0,
                min=0,
                max=1,
                step=0.05,
                description="Lower the temperature, lower the randomness in answer.",
            ),
            cl.input_widget.Select(
                id="force_reset_vectordb",
                label="Force reset vectordb",
                values=["True", "False"],
                initial_value="False",
                description="Choose True to refresh vectordb",
            ),
        ]
    ).send()
    gpt_llm = FordLLM(
        model=settings["model"],
        context=SYSTEM_PROMPT,
        temperature=settings["temp"],
    )
    cl.user_session.set("gpt_llm", gpt_llm)
    await cl.Message(content="Please wait while we are setting up things for your session..").send()
    await setup_agent(settings)


@cl.step
async def setup_vectorstore(sql_tool: Any):
    """
    Function to setup the vector store.
    """
    await cl.context.current_step.stream_token("Setting up Vector Store...")
    sql_tool = await sql_tool
    return sql_tool

@cl.step(name="Setup Granularity")
async def set_granularity():
    token_fetcher = TokenFetcher()
    embeddings = OpenAIEmbeddings(
        api_key=token_fetcher.token,
        base_url="https://api.pivpn.core.ford.com/fordllmapi/api/v1",
        model="text-embedding-ada-002"
    )
    print("Token call for Embedding")
    granularity_llm = ChatOpenAI(
        api_key=token_fetcher.token,
        base_url="https://api.pivpn.core.ford.com/fordllmapi/api/v1",
        model_name="gpt-4.1-mini",
        temperature=0
    )
    print("Token call completed for Open AI")
    system_prompt = """ You are an helpful assitant. You are provided with different granularity followed by unique values present in it. Your task is to analyze the quetion
                    and identify the different levels of granularity which are of exact match from the columns associated with table. Input provided to you contains all relevant tables, So do not reject any tables. Respond in JSON format. 
                    There will be a scenario, where multiple tables will be involved, where one table may contain value in vehicle and other may contain same value in sub_vehicle column. DO TAKE ACCOUNT FOR THESE KIND OF SCENARIOS.
                    DO NOT HALLUCINATE WITH COLUMNS THAT ARE NOT ASSOCIATED WITH TABLE. Please find the below expansion and corresponding values in table.
                    1. column:segment,actual value in column: MEDIUM TRADITIONAL UTILITY, short-form:MTU
                    2. columm:segment,actual value in column: LARGE TRADITIONAL PREMIUM UTILITY,short-form:LTPU
                    Always Provide the complete name associated in the column even if the question mentions partial name.
                    Example: Share of Mufasa across each priceband
                    Explanation: Unique value found in model has HYUNDAI MUFASA, but question mentions Mufasa. You will have to include complete name mentioned in column's Unique Value
                    Response: 
                    [
                        {{
                            table_name: "china_vehicle_sales",
                            [
                                {{
                                    "column_name": "model",
                                    "value" : "HYUNDAI MUFASA"
                                
                                }}
                            
                            ],
                            table_name: "trim_info_vehicle_table",
                            [
                                {{
                                    "column_name": "model",
                                    "value" : "HYUNDAI MUFASA"
                                }}
                            ]
                        }},
                        "explanation": MUFASA is mentioned in the question, model column has matching value HYUNDAI MUFASA.
                    ]

                    ======Response Guidelines
                    {{
                        table_name:<Name of the table>,
                        [
                        {{
                        column_name:<Name of the column associated with the above table>,
                        value:<value match in column>,
                        }}
                        ]
                        explanation: <brief explanation about why this column is selected>
                    }}
                    1. Response should be valid JSON array which should be parsed in python using json.loads()
                    2. You Should never add any additional comments such as "Understood. I will provide the response strictly in JSON format without any additional comments."
                    3. Only JSON response is allowed

                    Example 1 : Share of Polestar within priceband 350k-400k in 2024
                    Response: [
                        {{
                            "table_name": "china_vehicle_sales",
                            "columns": [
                                {{
                                    "column_name": "brand",
                                    "value": "POLESTAR"
                                }}
                            ]
                        }},
                        {{
                            "table_name": "trim_info_vehicle_table",
                            "columns": [
                                {{
                                    "column_name": "brand",
                                    "value": "POLESTAR"
                                }},

                            ],

                        }},
                        
                        "explanation": "Brand value of POLESTAR exactly matches with the question being asked."
                    ]


                    question:
                    {question}

                    Granular Columns and Unique Values:
                    {columns}

                    """

    system_message_prompt_template = SystemMessagePromptTemplate.from_template(
        system_prompt
    )

    human_message_prompt_template = HumanMessagePromptTemplate.from_template(
        # "Strictly Reply with only 'Y' or 'N'. No additional comments should be added."
        "Strictly follow the JSON format. No additional comments should be added. Response should be in json format"
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt_template, human_message_prompt_template]
    )

    vectorstore, docs = await setup_vector_store_granularity(embeddings=embeddings)
    cl.user_session.set("chat_prompt", chat_prompt)
    cl.user_session.set("embeddings", embeddings)
    cl.user_session.set("granularity_llm", granularity_llm)
    cl.user_session.set("vectorstore", vectorstore)
    cl.user_session.set("docs", docs)

@cl.step(name="Set Tool")
async def set_tool(settings: cl.ChatSettings):
    """
    Function to set the tool. Here, the user's choice of LLM will get initialized using FordLLM.
    EDA tool will get initialized as yywell.
    """
    await cl.context.current_step.stream_token("Setting tool...")
    if settings["chat history"] == "Enable":
        chat_history = True
    else:
        chat_history = False
    try:
        sql_tool = cl.make_async(bq.tool_set)(
            llm=cl.user_session.get("gpt_llm"),
            dataset_id=settings["dataset"],
            tables_to_include=[],
            additional_prompt=additional_sql_prompt,
            chat_history=chat_history,
        )
    except Exception as e:
        raise_exception("set_tool", e)

    domain = None
    if chat_config.CHAT_DOMAIN:
        domain = chat_config.CHAT_DOMAIN
    eda_tool = EDAHandler(domain=domain, model_name=settings["model"]).eda_tool_set()
    tools = [sql_tool, eda_tool]
    cl.user_session.set("llm_model_name", settings["model"])
    cl.user_session.set("dataset", settings["dataset"])
    return tools


@cl.on_settings_update
async def setup_agent(settings: cl.ChatSettings):
    """
    Function to setup the agent.
    """
    prev_dataset = cl.user_session.get("prev_dataset")

    # for entry in os.listdir('.'):
    #     if 'VECTORDB' in entry:
    #         vectordb_path = entry
    #         prev_dataset = vectordb_path.split('_')[0]
    #         print(vectordb_path)
    #         print("previous dataset")
    #         print(prev_dataset)

    # Regenerate vector db if dataset is changed or user chooses to refresh vectordb. Do not refresh if this is the first run
    # of the application, in that case the user can do a forced reset but by default application will read existing vectordb
    if (str(settings["dataset"]) != prev_dataset and prev_dataset is not None) or settings["force_reset_vectordb"] == "True":
        schema_file_path = os.getenv("MULTITABLE_SCHEMA_FILE")
        if os.getenv("SAVE_VECTORDB") == "True" and os.path.exists(schema_file_path):
            logger.info("Removing store metadata json file")
            os.remove(schema_file_path)
        if os.getenv("SAVE_VECTORDB") == "True" and os.path.exists(os.getenv("VECTORDB_PATH")):
            logger.info("Removing stored vectordb file")
            shutil.rmtree(os.getenv("VECTORDB_PATH"))

        logger.info("Stored vectordb and schema files removed, creating new ones ...")

    sql_tool, eda_tool = await set_tool(settings)

    awaited_sql_tool = await setup_vectorstore(sql_tool)
    # set var for prev dataset
    cl.user_session.set("prev_dataset", settings["dataset"])

    output = f"Current Model Settings!\n```json\n{json.dumps(settings, indent=2)}\n```"
    # await cl.Message(content=output).send()
    cl.user_session.set("sql_tool", awaited_sql_tool)
    cl.user_session.set("eda_tool", eda_tool)
    cl.user_session.set("continue_eda", False)
    await set_granularity()
    await cl.Message(content=" Setup is completed. Start asking your questions").send()

def _safe_generate_sql(tool, question):
    """
    Wrapper that calls `tool.generate_sql` and never raises.
    It returns a dict with either 'result' or 'error'.
    """
    try:
        result = tool.generate_sql(question)
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": e}

@cl.step(name="Generate SQL", type="llm", language="sql")
async def generate_sql(message: str):
    """
    Step to generate the SQL query.

    Args:
        message (cl.Message): User message.
    """
    await cl.context.current_step.stream_token("Generating SQL...")
    sql_tool = cl.user_session.get("sql_tool")
    top_n_tables = sql_tool.retrieve_top_n_tables(message)
    tbl_list = top_n_tables.keys()
    print(f"*********Top N Tables*********:{tbl_list}")
    await cl.Message(content="Querying vector store for column matches...").send()

    related_column = {}
    for i, tbl in enumerate(tbl_list):
        response = await query_vector_granularity_store(tbl, message, 4, cl.user_session.get("docs"))
        logger.info(f"Response from Vector Store:{response}")
        related_column[tbl] = []
        column_details = {}
        for re in response:
            # print(re)
            # print("******************")
            column_name = re.page_content.split(":")[0]
            unique_values = re.page_content.split(":")[1]
            related_column[tbl].append({column_name: unique_values})

        if i % 3 == 0:
            await cl.Message(content=f"Processed {i + 1}/{len(tbl_list)} tables...").send()
        await asyncio.sleep(0)
    chat_prompt = cl.user_session.get("chat_prompt")
    logger.info(f"***********{related_column}*********")
    req = chat_prompt.format_prompt(
        question=message,
        columns=related_column
    )
    granularity_llm = cl.user_session.get("granularity_llm")
    loop = asyncio.get_running_loop()
    try:
        resp = await loop.run_in_executor(None, granularity_llm.invoke, req)
    except Exception as e:
        if 'invalid' in str(e).lower() or 'unauthorized' in str(e).lower():
            cl.user_session.set("refresh_browser", True)
    # resp = granularity_llm.invoke(req)
    message += f"""You are provided with important information about granularity which will help you to solve the question in json format.
            Below information contains column name and its matching value which will help you identify correct columns to filter or query.
            {resp.content}"""
    logger.info(resp.content)
    # try:
    #     loop = asyncio.get_running_loop()
    #     response = await loop.run_in_executor(None, sql_tool.generate_sql, message)
    #     # response = sql_tool.generate_sql(question=message)
    # except ValueError as e:
    #     logger.error(f"SQL query generation failed due to failed LLM call: {e}")
    #     await cl.Message(
    #         content=f"Generate SQL failed due to failed LLM call: {e}", language="json"
    #     ).send()
    # except Exception as e:
    #     await cl.Message(content="Error occured in SQL Generation Step. Please try again or start new chat").send()
    #     logger.error(f"Error in generate sql step: {e}")
    #     if 'invalid' in str(e).lower() or 'unauthorized' in str(e).lower():
    #         cl.user_session.set("refresh_browser", True)
    #     await cl.Message(
    #         content=f"Error in generating SQL: {e}", language="json"
    #     ).send()
    #     return {"error": str(e)}
    wrapped_call = partial(_safe_generate_sql, sql_tool, message)
    outcome = await loop.run_in_executor(None, wrapped_call)
    if outcome["error"] is not None:
        await cl.context.current_step.fail("SQL generation error")
        await cl.Message(
            content="❌ Failed to generate SQL. "
                    "Please re-try or re-phrase your request.",
        ).send()
        # you can inspect outcome["error"] here if you want
        return
    response = outcome["result"]
    if not isinstance(response, dict) or "top_n_tables" not in response or "sql_query" not in response:
        # await cl.context.current_step.fail("Invalid format from SQL tool")
        await cl.Message(
            content="❌ The assistant produced an invalid response format. "
                    "Please ask again.",
        ).send()
        return
    top_n_tables = response.get("top_n_tables")
    sql_query = response.get("sql_query")
    cl.user_session.set("top_n_tables", top_n_tables)
    validated_sql = ""
    if isinstance(sql_query, dict):
        await cl.Message(content=sql_query, language="json").send()
        return sql_query
    elif isinstance(sql_query, str) and sql_query != "":
        # await cl.Message(content=f"Generated SQL: {sql_query}", language="json").send()
        await cl.Message(content="Generating SQL Please wait...").send()
        try:
            validated_sql = await validate_sql(sql_query)
            return validated_sql
        except Exception as e:
            raise_exception("generate_sql", e)

    elif isinstance(sql_query, str) and sql_query == "":
        await cl.Message(
            content=f"Something went wrong with the query generation process.{response}",
            language="json",
        ).send()
        logger.error("No SQl query generated")
        return sql_query
    return sql_query


# write a step validate sql
@cl.step(name="Validate SQL", type="llm", language="sql")
async def validate_sql(sql: str):
    """
    Step to validate the SQL query.

    Args:
        sql (str): SQL query to be validated.
    """
    await cl.context.current_step.stream_token("Validating SQL...")
    sql_tool = cl.user_session.get("sql_tool")
    top_n_tables = cl.user_session.get("top_n_tables")
    await cl.Message(content="Validating the SQL query, please wait...").send()
    try:
        # response = sql_tool.query_validation(
        #     query=sql, top_n_tables=top_n_tables, error=False
        # )
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None,
                                              partial(sql_tool.query_validation, query=sql, top_n_tables=top_n_tables,
                                                      error=False))
        if isinstance(response, dict) and "error" in response.keys():
            return {"error": response["error"]}
        validated_query = response["sql_query"]
        explanation = response["explanation"]
        if isinstance(validated_query, str) and validated_query == "":
            validated_query = sql
            await cl.Message(
                content="Validated SQL is empty because: " + str(explanation),
                language="json",
            ).send()
        if isinstance(validated_query, str) and validated_query != "":
            await cl.Message(
                content=f"Validated SQL: {validated_query}", language="sql"
            ).send()
        else:
            await cl.Message(content=validated_query, language="sql").send()
    except Exception as e:
        raise_exception("validate_sql", e)

    return validated_query


@cl.step(name="Execute SQL")
async def run_sql(sql: str):
    """
    Step to execute the SQL query.

    Args:
        sql (str): SQL query to be processed
    """
    await cl.context.current_step.stream_token("Running SQL...")
    sql_tool = cl.user_session.get("sql_tool")
    df = sql_tool.get_results_df(sql)
    if isinstance(df, str):
        try:
            # validated_query, explanation = sql_tool.query_validation(
            #     query=sql,
            #     top_n_tables=cl.user_session.get("top_n_tables"),
            #     error=True,
            #     error_message=df,
            # )
            loop = asyncio.get_running_loop()

            validated_query, explanation = await loop.run_in_executor(None,
                                                                      partial(sql_tool.query_validation, query=sql,
                                                                              top_n_tables=cl.user_session.get(
                                                                                  "top_n_tables"),
                                                                              error=True,
                                                                              error_message=df))
            if validated_query == "":
                await cl.Message(content=df, language="json").send()
                raise RuntimeError(df)
            else:
                await cl.Message(content=explanation, language="json").send()
                return validated_query

        except Exception as e:
            raise_exception("run_sql", e)

    cl.user_session.set("sql_result", df)
    return df


@cl.step(name="EDA")
async def run_eda_suggest():
    """
    Step to suggest EDA plots.
    """
    await cl.context.current_step.stream_token("Running EDA Suggest...")
    eda_tool = cl.user_session.get("eda_tool")
    df = cl.user_session.get("sql_result")
    _, goals, warning_msg = eda_tool.set_up(df=df)
    if warning_msg:
        await cl.Message(content=warning_msg).send()

    goals_reformat_dict = {}
    goals_reformat_str = "The following are possible plots you can generate:\n"
    for g in goals:
        goals_reformat_dict[str(g.index)] = {
            "question": g.question,
            "visualization": g.visualization,
            "rationale": g.rationale,
        }
        goals_reformat_str += f"### Goal {g.index}: \n `Question`: {g.question}\n `Chart`: {g.visualization}\n `Rationale`: {g.rationale}\n\n"

    await cl.Message(content=goals_reformat_str).send()
    cl.user_session.set("goals_reformat_dict", goals_reformat_dict)
    return goals


@cl.step(name="Generate Plot")
async def on_generate_plot():
    """
    Step to generate the plot, save it and then provide insights on the plot.
    """
    eda_tool = cl.user_session.get("eda_tool")
    goal = cl.user_session.get("goal")
    if isinstance(goal, str):
        goal = {"question": goal, "visualization": goal, "rationale": ""}

    await cl.context.current_step.stream_token("Running Plot Generation...")
    chart = eda_tool.get_chart(goal=goal)
    if len(chart) >= 1:
        img_file_name = get_chart_filename(chart[0].code)
        file_name = os.path.join(local_dir, f"{img_file_name}.png")
        chart[0].savefig(file_name)
        image = cl.Image(
            path=file_name, name=img_file_name, size="large", display="inline"
        )
        img_id = await cl.Message(
            content="Processed image!",
            elements=[image],
        ).send()
        explanation = eda_tool.plot_explain(encode_image(file_name), goal["question"])
        # await cl.Message(content=chart[0].code, language="python").send()
        explain_id = await cl.Message(
            content="Chart Explanation",
            elements=[cl.Text(content=explanation, display="inline")],
        ).send()
        await revert_sql_operation()
    else:
        await cl.Message(content="Unable to process this query").send()
        img_id, explain_id = "", ""


async def revert_sql_operation():
    revert_to_sql_action = [
        cl.Action(
            name="revert_sql", payload={"value": "True"}, label="Click for SQL query."
        )
    ]
    await cl.Message(
        content="**Revert back to SQL Query Execution?** Click the button below, else *continue asking plot based question for existing SQL output.*",
        elements=revert_to_sql_action,
    ).send()


@cl.action_callback("revert_sql")
async def on_revert_sql(choice: cl.Action):
    """
    Function to handle user choice to revert SQL query.

    Args:
        choice (cl.Action): User choice of selecting to perform SQL operation.
    """
    if choice.payload["value"] == "True":
        cl.user_session.set("continue_sql", True)
        # reset the df variable place holder
        cl.user_session.set("sql_result", None)
        cl.user_session.set("continue_eda", False)
        await cl.Message(
            content="You can now perform SQL query generation and execution!"
        ).send()


@cl.action_callback("goal_execute")
async def on_goal_execute(goal_id: cl.Action):
    """
    Action to execute the goal.

    Args:
        goal_id (cl.Action): Goal ID between 0, 1, 2.
    """
    goals_reformat_dict = cl.user_session.get("goals_reformat_dict")
    goal = goals_reformat_dict[goal_id.payload["value"]]
    cl.user_session.set("goal", goal)
    cl.user_session.set("goal_question", goal["question"])
    await on_generate_plot()


@cl.action_callback("eda_suggest")
async def on_eda_suggest():
    """
    Action to suggest EDA plots.
    """
    goals = await run_eda_suggest()
    cl.user_session.set("continue_eda", True)
    goals_actions = []
    for g in goals:
        goals_actions.append(
            cl.Action(
                name="goal_execute",
                payload={"value": str(g.index)},
                label=f"Goal {g.index}",
                tooltip="Click me to get the chart.",
            )
        )

    await cl.Message(
        content="Click one of the buttons for EDA chart OR **write your own query**",
        elements=goals_actions,
    ).send()
    await revert_sql_operation()


async def execute_sql(message: str):
    """
    Function to execute the SQL query and trigger action to select EDA analysis post SQL query execution.

    Args:
        message (cl.Message): User message.
    """
    sql = await generate_sql(message)
    if isinstance(sql, dict) and "explanation" in sql.keys():
        return "", sql["explanation"]
    elif isinstance(sql, dict) and "error" in sql.keys():
        return "", sql["error"]
    elif isinstance(sql, str) and sql == "":
        return "", "No query was generated"
    elif not isinstance(sql, str):
        return "", sql

    results = await run_sql(sql)

    if isinstance(results, str):
        await cl.Message(
            content=f"Query execution resulted in error - {results}. No results were generated.",
            language="json",
        ).send()
        return "", "No results were generated."

    csv_file_name = f"dataframe_{datetime.now().strftime(format='%Y_%m_%d_%H_%M_%S')}"
    csv_file_path = os.path.join(curr_dir, "data", f"{csv_file_name}.csv")
    results.to_csv(csv_file_path, index=False)
    download_elements = [
        cl.File(
            name=f"Download_{csv_file_name}",
            content=open(csv_file_path, "rb").read(),
            display="inline",
            mime="text/csv",
        ),
    ]
    query_id = await cl.Message(
        content=results.to_markdown(), elements=download_elements
    ).send()
    # show the SQL summary generated by GPT only if the result output follows the condition
    if results.shape[1] < 5 and results.shape[0] < 20:
        sql_tool = cl.user_session.get("sql_tool")
        summary_result = sql_tool.summarize_sql_output(
            message, sql, results.to_markdown()
        )
        # if isinstance(summary_result,dict)
        await cl.Message(
            content="SQL Result Summary",
            elements=[cl.Text(content=summary_result, display="inline")],
        ).send()

    if results.shape[0] > 5:
        eda_actions = [
            cl.Action(
                name="eda_suggest",
                payload={"value": "eda_suggest"},
                label="EDA Analysis",
                tooltip="Click here to perform EDA plots based on SQL output.",
            )
        ]
        await cl.Message(
            content="Based on this result, you can perform **EDA plot visualization**. Click on the button below to proceed.",
            elements=eda_actions,
        ).send()

    return query_id, sql


@cl.on_message
async def main(message: cl.Message):
    """
    Main function to handle the message from the user.

    Args:
        message (cl.Message): User message.
    """
    llm_model_name = cl.user_session.get("llm_model_name")
    if cl.user_session.get("continue_sql"):
        query_id, sql = await execute_sql(message.content)
        cl.user_session.set("continue_sql", False)
    elif cl.user_session.get("continue_eda"):
        cl.user_session.set("goal_question", message.content)
        query_id = await on_generate_plot()
        sql = ""
    else:
        query_id, sql = await execute_sql(message.content)
    if not isinstance(sql, dict):
        bq_logger.insert(store_logs(query_id, message.content, sql, llm_model_name))


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_app_user: cl.User,
) -> Optional[cl.User]:
    print(f">>>>>>>>>provider_id={provider_id}>>>>>>>>>")
    print(f">>>>>>>>>app_user={default_app_user.identifier}>>>>>>>>>")
    print(f">>>>>>>>>raw_user_data={raw_user_data}>>>>>>>>>")
    return default_app_user


# if __name__ == "__main__":
#     from chainlit.cli import run_chainlit
#     run_chainlit(__file__)
