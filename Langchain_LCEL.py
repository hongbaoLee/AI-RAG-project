# Langchian 示例：使用 DashScope API 调用大模型生成创意口号
# 主要学习Langchian的表达式语言 (LCEL) 语法
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 参数配置
LLM_MODEL = "qwen-plus"    
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


# 1. 初始化模型
llm = ChatOpenAI(
    openai_api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model=LLM_MODEL
)
# 2. 创建提示模板
prompt = ChatPromptTemplate.from_template("请给出一个关于 {topic} 的创意口号。")

# 3. 创建输出解析器
output_parser = StrOutputParser()

# 4. 使用 LangChain 表达式语言 (LCEL) 链接组件
# 这是现代 LangChain 的核心语法，用 | 符号连接
chain = prompt | llm | output_parser

# 5. 调用链
topic = "一家新开的咖啡店"
response = chain.invoke({"topic": topic})

print(response) 
# 可能的输出: "唤醒你的每一天，从一杯好咖啡开始。"
