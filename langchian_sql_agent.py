# langchain_sql_agent.py
# 通过Langchain构建一个SQL Agent，能够将自然语言查询转换为SQL语句并执行
import time
print("程序启动")
start = time.time()

import os
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# 参数配置
TEXT_TO_SQL_MODEL = "glm-4.6"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'mysql',
    'database': 'prestige_db'
}

# 连接数据库
db_uri = f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
db = SQLDatabase.from_uri(db_uri)
print("数据库连接测试:", db_uri)
llm = ChatOpenAI(
    openai_api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model=TEXT_TO_SQL_MODEL
)

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=False,
)

try:
    while True:
        user_input = input("请输入你的问题（输入quit或者exit退出）：")
        if user_input.strip().lower() == "quit" or user_input.strip().lower() == "exit":
            break
        response = agent_executor.invoke({"input": user_input})
        print("Agent response:", response)
        print("Output:", response.get("output"))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("执行出错：", e)


