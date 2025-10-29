# langchain_Agent_tools_so360.py
# 使用so.com 360搜索工具的 LangChain Agent 示例
# 学习Langchain的Agent工具调用功能，自定义360搜索作为Agent的工具
import os
import requests
from bs4 import BeautifulSoup          # 引入BeautifulSoup，解析spider返回的HTML文档
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
# from langchain_community.tools import DuckDuckGoSearchRun # 引入工具
from langchain_openai import ChatOpenAI
from langchain import hub

# 参数配置
LLM_MODEL = "qwen-plus"    
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 360搜索函数
def so_search(query, num_results=5):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    params = {"q": query}
    url = "https://www.so.com/s"
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for h3 in soup.find_all("h3")[:num_results]:
        title = h3.get_text(strip=True)
        summary = ""
        # 优先找下一个<p>，否则找父节点下的所有<p>或<div>
        sib = h3.find_next_sibling()
        while sib and not summary:
            if sib.name == "p":
                summary = sib.get_text(strip=True)
            elif sib.name == "div":
                summary = sib.get_text(strip=True)
            sib = sib.find_next_sibling()
        results.append({"title": title, "summary": summary})
    useful_results = [
        res for res in results
        if "图片" not in res['title'] and "还搜了" not in res['title']
    ]
    return useful_results
# 定义Agent的工具
@tool("so_search_tool")                # 不要return_direct=True参数
def so_search_tool(query: str) -> str:
    """
    使用360搜索抓取互联网最新信息，返回前几条有效结果的标题和摘要。
    """
    results = so_search(query)
    if not results:
        return "未找到相关内容。"
    return "\n\n".join([
        f"{idx+1}. {res['title']}\n{res['summary'] if res['summary'] else '[无摘要]'}"
        for idx, res in enumerate(results)
    ])

# 1. 初始化模型
llm = ChatOpenAI(
    openai_api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model=LLM_MODEL
)

# 2. 定义工具列表
# tools = [DuckDuckGoSearchRun()]
tools = [so_search_tool]
# 3. 获取预置的 Agent Prompt
prompt = hub.pull("hwchase17/openai-functions-agent")

# 4. 创建 Agent
# 这个 Agent 知道如何使用函数调用（Tool Calling）来与工具交互
agent = create_tool_calling_agent(llm, tools, prompt)

# 5. 创建 Agent 执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)     # verbose=True 可以看到思考过程

if __name__ == "__main__":
    while True:
        question = input("请输入您的问题（输入 quit 或 exit 退出）：")
        if question.strip().lower() in ("quit", "exit"):
            print("程序已退出。")
            break
        # 6. 执行 Agent
        response = agent_executor.invoke({"input": question})
        print(response['output'])





