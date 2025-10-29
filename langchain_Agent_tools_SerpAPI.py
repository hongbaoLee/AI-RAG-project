# langchain_Agent_tools_SerpAPI.py
# 使用 SerpAPI 联网搜索工具的 LangChain Agent 示例
# SerpAPI需要自己封装为Agent的Tools
import os
from langchain.tools import tool
from serpapi import GoogleSearch
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub


# 参数配置
LLM_MODEL = "qwen-plus"    
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
SERPAPI_API_KEY = "ec6e9ebf8e0bf594d94c62b1a2cc2f48620cf6f1cbb3f993877de68e4fd95aff"
@tool("serpapi_baidu_search")
def serpapi_baidu_search(query: str) -> str:    
    """使用 SerpAPI 百度引擎联网搜索，返回前几条结果摘要。"""
    # print(f"SerpAPI Tool 被调用，查询：{query}")  # 调试用
    params = {
        "q": query,
        "engine": "google",  # 可修改为"bing"，"baidu"等
        "api_key": SERPAPI_API_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    # print(results)  # 调试用，查看完整返回结果
    if "organic_results" in results:
        return "\n".join([item.get("snippet", "") for item in results["organic_results"][:3]])
    return "未找到相关内容。"


# 初始化 SerpAPI 工具
tools = [serpapi_baidu_search]

# 初始化 LLM
llm = ChatOpenAI(
    openai_api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model=LLM_MODEL
)

# 拉取 prompt
prompt = hub.pull("hwchase17/openai-functions-agent")

# 创建 Agent 和执行器
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 用户输入问题
while True:
    question = input("请输入您的问题（输入 quit 或 exit 退出）：")
    if question.strip().lower() in ("quit", "exit"):
        print("程序已退出。")
        break
    response = agent_executor.invoke({"input": question})
    print(response['output'])