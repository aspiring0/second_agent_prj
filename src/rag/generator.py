#src/rag/generator.py

#链接openai的chat模型，进行回答生成
from langchain_openai import ChatOpenAI

#话术模板，构建AI提示词
from langchain_core.prompts import ChatPromptTemplate
# 输出解析器，将模型输出转换为字符串
from langchain_core.output_parsers import StrOutputParser

#  RunnablePassthrough 用于将输入直接传递给输出，不进行任何处理
from langchain_core.runnables import RunnablePassthrough

from src.rag.retriever import VectorRetriever
from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("RAG_Generator")

class RAGGenerator:
    def __init__(self):
        self.retriever = VectorRetriever()

        self.llm = ChatOpenAI(
            model_name=settings.CHAT_MODEL,
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_BASE_URL

        )
        # --- 3. 定义提示词 (Prompt) ---
        # from_template 方法允许我们用 {variable} 的格式挖坑，后面再填填空题。
        self.prompt_template = ChatPromptTemplate.from_template("""
        你是企业内部知识库助手。请根据下面的【上下文】回答用户的问题。
        请遵循以下规则：
        1. 只要【上下文】中包含了与问题相关的任何事实（例如定义、数据、描述），就请根据这些事实进行回答。
        2. 不要死板地寻找完美匹配。如果用户问“见解”或“理解”，请基于上下文中的事实进行总结。
        3. 只有当上下文中完全没有提及问题的主题时，才回答“未找到信息”
        
        【上下文】:
        {context}  <-- 这里一会儿会填入我们从数据库查到的文档
        
        【用户问题】:
        {question} <-- 这里会填入用户在终端输入的问题
        """)
        
    def _format_docs(self, docs):
        """
        数据清洗
        docs是一个列表
        将文档列表格式化为字符串，每个文档占一行
        """
        return "\n\n".join([doc.page_content for doc in docs])

    def get_answer(self, question: str):
        """
        生成回答
        question: 用户输入的问题
        """
        logger.info(f"正在生成回答... 问题: {question}")

        # 1. 初始化检索器
        docs = self.retriever.query(question,top_k=3)
        # 兜底逻辑：如果数据库是空的，或者啥也没查到，直接返回，省点 API 钱
        if not docs:
            return logger.warning("⚠️ 知识库中没有任何相关文档。")
        else:
            logger.info(f"检索到 {len(docs)} 个相关文档")
        # --- 🔴 新增调试打印 ---
        # 这一段可以让你在控制台看到检索到的具体内容，排查为什么 AI 觉得没答案
        print("\n" + "="*20 + " [调试] 检索到的上下文 " + "="*20)
        for i, (doc, score) in enumerate(docs):
            print(f"📄 片段 {i+1} (匹配分 {score:.2f}):\n{doc.page_content.strip()[:100]}...") # 只看前100字
        print("="*60 + "\n")
        # ---------------------
        # 将查到的对象列表 (docs) 里的分数去掉，只保留文档对象，然后清洗成字符串
        # docs 里的结构是 [(Document, score), (Document, score)...]
        # 列表推导式 [doc for doc, score in docs] 取出了其中的 Document
        # 然后用 _format_docs 方法将它们格式化为字符串
        context = self._format_docs([doc for doc, score in docs])

        logger.info(f"检索上下文长度: {len(context)} 字符")

        # 2. 构建链
        # 链的工作流程：
        # 1. 从用户输入 question 开始
        # 2. 调用 prompt_template 格式化，将 context 填充到模板中
        # 3. 调用 llm 模型生成回答
        # 4. 用 StrOutputParser 解析模型输出，将其转换为字符串
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        try:
            logger.info("调用 LLM 生成回答中...")
            # --- 步骤 C: 执行 (Invoke) ---
            # invoke 是启动键。
            # 我们传入一个字典，字典里的 key (context, question) 必须对应 self.prompt 里挖的那个坑 {context}, {question}。
            answer = rag_chain.invoke({"context": context, "question": question})
            logger.info(f"LLM 生成的回答: {answer}")
            return answer
        except Exception as e:
            logger.error(f"LLM 调用出错: {e}")
            return "生成回答时出错，请稍后重试。"