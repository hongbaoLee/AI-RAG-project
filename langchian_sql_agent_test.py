# langchain_sql_agent.py
# 通过Langchain构建一个SQL Agent，能够将自然语言查询转换为SQL语句并执行。
# openai-tools模式下，"glm-4.6"模型可以很好地支持SQL Agent运行。但有时候输出时会丢失数据。
# deepseek-v3.2-exp模型可以支持zero-shot-react-description模式，但是速度很慢，中间由于格式解析问题会报错。详细原因（ChatGPT解释）：
# Langchain SQL Agent 需要 LLM 按照特定格式（如“Thought: ...\nAction: ...\nAction Input: ...”）一步步推理和调用工具（如sql_db_query），最后才输出Final Answer。
# LLM（如 deepseek-v3.2-exp）直接给出了最终答案（甚至是伪造的表格），没有走工具链，Langchain无法解析这种输出。
# 这类问题在国产大模型和未专门为function calling微调的模型上非常常见。
# 解决方法是使用支持function calling的模型，或者对模型进行微调，使其能够按照预期格式输出。
# qwen系列模型，在各种模式下测试均无法正确运行SQL Agent。
#
# 结论：国产大模型在function calling和agent方面的能力还有待提升，建议使用OpenAI的GPT-4等模型以获得更稳定的体验。
# 在程序中还是让大模型生成SQL语句，然后在程序中直接执行SQL语句，避免使用agent模式。
#

import time
print("程序启动")
start = time.time()

import os
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# 参数配置
TEXT_TO_SQL_MODEL = "glm-4.6"    #可选模型：glm-4.6, deepseek-v3.2-exp等
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
    agent_type="openai-tools",    # 使用openai-tools模式，可选模式：openai-functions, tool-calling, zero-shot-react-description
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


