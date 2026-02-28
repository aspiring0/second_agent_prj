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
        self.prompt_template = ChatPromptTemplate.from_template("""你是一个知识库问答助手。你的任务是根据检索到的上下文内容回答用户问题。

【检索到的上下文】:
{context}

【用户问题】:
{question}

【回答规则 - 必须严格遵守】:
1. 你必须基于上面的【检索到的上下文】来回答，这是你唯一的信息来源
2. 禁止说"未找到"、"无法回答"、"没有相关信息"等拒绝性语句
3. 如果上下文中确实有内容，你就必须对这些内容进行总结、整理或解释
4. 即使上下文与问题不是100%匹配，也要尽力从上下文中提取有价值的信息回答
5. 如果用户问的是某个文件，请检查上下文中的【来源】信息，找到匹配的内容进行整理

现在请基于上下文内容直接给出回答:""")
        
    def _format_docs(self, docs):
        """
        数据清洗
        docs是一个列表
        将文档列表格式化为字符串，包含来源信息
        """
        formatted = []
        for i, doc in enumerate(docs):
            # 获取来源信息
            source = doc.metadata.get('source', '未知来源')
            formatted.append(f"【文档{i+1} 来源: {source}】\n{doc.page_content}")
        return "\n\n".join(formatted)

    def get_answer(self, question: str, session_id=None, project_id="default"):
        """
        生成回答
        question: 用户输入的问题
        """
        logger.info(f"🤖 收到问题: {question} (Session: {session_id})")

        # 1. 初始化检索器
        docs = self.retriever.query(question, project_id=project_id, top_k=3)
        # 兜底逻辑：如果数据库是空的，或者啥也没查到，直接返回
        if not docs:
            logger.warning("⚠️ 知识库中没有任何相关文档。")
            return "抱歉，知识库中没有找到与您问题相关的内容。"
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
        # 直接使用 prompt | llm | parser 的简单链结构
        # invoke 时直接传入包含 context 和 question 的字典
        rag_chain = self.prompt_template | self.llm | StrOutputParser()

        try:
            logger.info("调用 LLM 生成回答中...")
            logger.info(f"📝 上下文内容预览: {context[:200]}...")
            # --- 步骤 C: 执行 (Invoke) ---
            # invoke 是启动键。
            # 我们传入一个字典，字典里的 key (context, question) 必须对应 self.prompt 里挖的那个坑 {context}, {question}。
            answer = rag_chain.invoke({"context": context, "question": question})
            logger.info(f"✅ LLM 生成的回答: {answer}")
            return answer
        except Exception as e:
            logger.error(f"LLM 调用出错: {e}")
            return "生成回答时出错，请稍后重试。"