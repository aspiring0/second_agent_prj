#src/rag/generator.py
"""
RAG生成器模块 - 负责基于检索结果生成回答
包含：
1. 相关性判断机制
2. 拒绝回答机制（不知道能力）
3. Embedding缓存（待实现）
4. 统一提示词管理
"""
#话术模板，构建AI提示词
from langchain_core.prompts import ChatPromptTemplate
# 输出解析器，将模型输出转换为字符串
from langchain_core.output_parsers import StrOutputParser

from src.rag.retriever import VectorRetriever
from src.agent.prompts import (
    get_rag_generator_prompt, 
    get_relevance_check_prompt,
    PromptManager
)
from config.settings import settings
from src.utils.logger import setup_logger
from src.utils.model_manager import model_manager
from typing import List, Tuple, Optional
import time

logger = setup_logger("RAG_Generator")

# 相关性阈值配置
RELEVANCE_THRESHOLD = 0.15  # 低于此分数认为不相关（放宽，减少误拒）
SCORE_THRESHOLD = 2.0      # 向量距离阈值（越小越相关）

class RAGGenerator:
    def __init__(self, enable_relevance_check: bool = True):
        self.retriever = VectorRetriever()
        self.enable_relevance_check = enable_relevance_check

        # LLM 实例延迟获取，确保每次使用最新的 API Key
        # 不在 __init__ 中缓存，避免 UI 更新 Key 后仍用旧实例

        # --- 使用统一的提示词管理模块 ---
        # 从 PromptManager 获取提示词模板，便于统一管理和版本控制
        self.prompt_template = get_rag_generator_prompt()
        self.relevance_prompt = get_relevance_check_prompt()

        # --- Sprint 1: Reranker 重排序 ---
        self.enable_reranker = settings.ENABLE_RERANKER
        self.reranker = None
        if self.enable_reranker:
            try:
                from src.rag.reranker import Reranker
                self.reranker = Reranker(backend=settings.RERANKER_BACKEND)
                logger.info(f"✅ Reranker 已启用，后端: {settings.RERANKER_BACKEND}")
            except Exception as e:
                logger.warning(f"Reranker 初始化失败: {e}，将不使用重排序")
                self.enable_reranker = False

        logger.info("RAG Generator 初始化完成，使用统一提示词管理")

    def _get_llm(self):
        """延迟获取 LLM 实例，每次调用时从 model_manager 获取最新的"""
        return model_manager.get_chat_model(temperature=0.1)

    def _format_docs(self, docs: List) -> str:
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
    
    def _format_docs_with_scores(self, docs: List[Tuple]) -> str:
        """格式化带分数的文档列表"""
        formatted = []
        for i, (doc, score) in enumerate(docs):
            source = doc.metadata.get('source', '未知来源')
            formatted.append(f"【文档{i+1} 来源: {source} 相关度: {score:.2f}】\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    def check_relevance(self, question: str, context: str) -> float:
        """
        使用LLM判断检索内容与问题的相关性
        返回0-1之间的相关性分数
        """
        if not context or len(context.strip()) < 10:
            return 0.0
        
        try:
            chain = self.relevance_prompt | self._get_llm() | StrOutputParser()
            result = chain.invoke({"question": question, "context": context[:1000]})
            
            # 解析分数
            score_str = result.strip().replace("\n", "")
            # 尝试提取数字
            import re
            match = re.search(r'[\d.]+', score_str)
            if match:
                score = float(match.group())
                return min(1.0, max(0.0, score))
            return 0.5  # 默认中等相关
        except Exception as e:
            logger.warning(f"相关性判断失败: {e}，默认返回0.5")
            return 0.5
    
    def check_relevance_by_score(self, docs: List[Tuple]) -> float:
        """
        基于向量检索分数判断相关性（快速方法，不需要额外LLM调用）
        返回0-1之间的相关性分数
        """
        if not docs:
            return 0.0
        
        # 取前3个结果的平均分
        avg_score = sum(score for _, score in docs[:3]) / len(docs[:3])
        
        # Chroma使用余弦距离，距离越小越相关
        # 转换为0-1的相关性分数（距离0=相关度1，距离2=相关度0）
        relevance = max(0, 1 - avg_score / 2)
        return relevance
    
    def should_deny(self, question: str, docs: List[Tuple], use_llm_check: bool = False) -> Tuple[bool, str]:
        """
        判断是否应该拒绝回答（返回"不知道"）
        
        Args:
            question: 用户问题
            docs: 检索到的文档列表 [(doc, score), ...]
            use_llm_check: 是否使用LLM进行相关性判断（更准确但更慢）
            
        Returns:
            (should_deny, reason): 是否拒绝，拒绝原因
        """
        # 1. 没有检索到任何文档
        if not docs:
            return True, "no_results"
        
        # 2. 基于向量分数快速判断
        score_relevance = self.check_relevance_by_score(docs)
        if score_relevance < RELEVANCE_THRESHOLD:
            logger.info(f"⚠️ 向量相关性过低: {score_relevance:.2f} < {RELEVANCE_THRESHOLD}")
            return True, "low_vector_relevance"
        
        # 3. 检查最相关文档的分数
        best_score = docs[0][1] if docs else 999
        if best_score > SCORE_THRESHOLD:
            logger.info(f"⚠️ 最佳匹配分数过低: {best_score:.2f} > {SCORE_THRESHOLD}")
            return True, "low_best_score"
        
        # 4. 可选：使用LLM进行更精确的判断
        if use_llm_check:
            context = self._format_docs_with_scores(docs)
            llm_relevance = self.check_relevance(question, context)
            if llm_relevance < RELEVANCE_THRESHOLD:
                logger.info(f"⚠️ LLM相关性过低: {llm_relevance:.2f}")
                return True, "low_llm_relevance"
        
        return False, "relevant"

    def get_answer(self, question: str, session_id=None, project_id="default") -> str:
        """
        生成回答（带相关性判断和拒绝机制）
        question: 用户输入的问题
        """
        start_time = time.time()
        logger.info(f"🤖 收到问题: {question} (Session: {session_id})")

        # 1. 检索 → Rerank → 生成
        docs = self.retriever.query(question, project_id=project_id, top_k=10)

        # Sprint 1: 检索后重排序
        if self.enable_reranker and self.reranker and docs:
            docs = self.reranker.rerank(question, docs, top_k=3)
            logger.info(f"Rerank 后保留 {len(docs)} 条结果")
        
        # 2. 判断是否应该拒绝回答
        if self.enable_relevance_check:
            should_deny, deny_reason = self.should_deny(question, docs, use_llm_check=False)
            if should_deny:
                latency = (time.time() - start_time) * 1000
                logger.info(f"⏱️ 拒绝回答 (原因: {deny_reason}, 耗时: {latency:.0f}ms)")
                return self._generate_denial_response(question, deny_reason)
        
        # 兜底逻辑：如果没有启用相关性检查，使用旧逻辑
        if not docs:
            logger.warning("⚠️ 知识库中没有任何相关文档。")
            return "抱歉，知识库中没有找到与您问题相关的内容。"
        
        logger.info(f"检索到 {len(docs)} 个相关文档")
        
        # 调试打印
        print("\n" + "="*20 + " [调试] 检索到的上下文 " + "="*20)
        for i, (doc, score) in enumerate(docs):
            print(f"📄 片段 {i+1} (匹配分 {score:.2f}):\n{doc.page_content.strip()[:100]}...")
        print("="*60 + "\n")
        
        context = self._format_docs_with_scores(docs)
        logger.info(f"检索上下文长度: {len(context)} 字符")

        # 3. 生成回答
        rag_chain = self.prompt_template | self._get_llm() | StrOutputParser()

        try:
            logger.info("调用 LLM 生成回答中...")
            answer = rag_chain.invoke({"context": context, "question": question})
            latency = (time.time() - start_time) * 1000
            logger.info(f"✅ LLM 生成的回答 (耗时: {latency:.0f}ms): {answer[:100]}...")
            return answer
        except Exception as e:
            logger.error(f"LLM 调用出错: {e}")
            return "生成回答时出错，请稍后重试。"
    
    def _generate_denial_response(self, question: str, reason: str) -> str:
        """生成拒绝回答的响应"""
        denial_responses = {
            "no_results": "抱歉，我在知识库中没有找到与您问题相关的内容。请尝试换一种方式提问，或者确认问题是否与知识库内容相关。",
            "low_vector_relevance": "抱歉，您的问题似乎与知识库内容关联度不高。请尝试提供更具体的问题描述，或确认问题是否属于知识库的范围。",
            "low_best_score": "抱歉，我没有找到足够相关的信息来回答您的问题。请尝试换一种表述方式。",
            "low_llm_relevance": "抱歉，根据我的分析，您的问题与知识库内容相关性较低。请确认问题是否与已上传的文档相关。"
        }
        return denial_responses.get(reason, "抱歉，我无法回答这个问题。")
    
    def get_answer_with_relevance(self, question: str, session_id=None, project_id="default") -> dict:
        """
        生成回答并返回相关性信息（用于测试和评估）
        """
        start_time = time.time()
        
        # 检索
        docs = self.retriever.query(question, project_id=project_id, top_k=3)
        
        # 计算相关性
        score_relevance = self.check_relevance_by_score(docs)
        should_deny, deny_reason = self.should_deny(question, docs, use_llm_check=False)
        
        # 生成回答
        if should_deny:
            answer = self._generate_denial_response(question, deny_reason)
        elif not docs:
            answer = "抱歉，知识库中没有找到与您问题相关的内容。"
        else:
            context = self._format_docs([doc for doc, score in docs])
            rag_chain = self.prompt_template | self._get_llm() | StrOutputParser()
            try:
                answer = rag_chain.invoke({"context": context, "question": question})
            except Exception as e:
                answer = f"生成回答时出错: {str(e)}"
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "question": question,
            "answer": answer,
            "relevance_score": score_relevance,
            "should_deny": should_deny,
            "deny_reason": deny_reason,
            "docs_count": len(docs),
            "latency_ms": latency
        }
