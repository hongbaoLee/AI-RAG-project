import streamlit as st
from embedding_alli import ask_company_rules, collection, load_pdf_to_vector_db, PDF_PATH


# 设置页面标题和图标
st.set_page_config(
    page_title="📘 公司制度智能助手",
    page_icon="📘",
    layout="centered"
)

# 页面标题
st.title("📘 公司制度智能问答系统")
st.markdown("由通义千问 + 阿里云百炼驱动，支持全文检索与智能回答")

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 检查是否已加载数据
if collection.count() == 0:
    st.warning(f"⚠️ 检测到知识库为空，正在从 `{PDF_PATH}` 导入数据...")
    try:
        load_pdf_to_vector_db(PDF_PATH)
        st.success("✅ 数据导入成功！")
        st.rerun()
    except Exception as e:
        st.error(f"❌ 导入失败：{str(e)}")
        st.stop()
else:
    st.success(f"✅ 已加载 {collection.count()} 条制度内容，可开始提问")

# 显示聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入你的问题，例如：年假怎么申请？"):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 显示 AI 思考过程
    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答..."):
            response = ask_company_rules(prompt)
        st.write(response)

    # 添加 AI 消息
    st.session_state.messages.append({"role": "assistant", "content": response})

# 清除聊天按钮
if st.button("🗑️ 清除聊天记录"):
    st.session_state.messages = []
    st.rerun()

# 提示语
st.markdown("---")
st.markdown("💡 **提示**：本系统基于公司内部 PDF 文件构建，所有数据本地存储，安全可靠。")