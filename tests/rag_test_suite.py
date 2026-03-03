# tests/rag_test_suite.py
"""
RAG系统完整测试套件 - RAG Test Suite
基于PROJECT_DOCUMENTATION.md构建的标准化测试问答集
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

from src.metrics.collector import metrics_collector
from src.metrics.performance import performance_tracker
from src.metrics.quality import quality_evaluator
from src.utils.logger import setup_logger

logger = setup_logger("RAG_TEST")


# ==================== 基于PROJECT_DOCUMENTATION.md的标准化测试问答集 ====================
# 这个数据集基于项目文档构建，包含25个标准问答对

PROJECT_DOC_TEST_CASES = [
    # ==================== 基础概念类 (5题) ====================
    {
        "id": "doc_001",
        "query": "这个项目是什么？",
        "ground_truth": "这是一个企业级RAG（检索增强生成）多智能体问答系统，结合了RAG技术、多智能体架构（基于LangGraph）、知识库管理和Web UI。",
        "category": "factual",
        "difficulty": "easy",
        "keywords": ["RAG", "多智能体", "问答系统", "企业级"]
    },
    {
        "id": "doc_002",
        "query": "RAG是什么？它的核心流程是什么？",
        "ground_truth": "RAG是检索增强生成（Retrieval-Augmented Generation）技术。核心流程包括：1)索引阶段：文档切分、向量化、存入向量数据库；2)检索阶段：问题向量化、相似度搜索；3)生成阶段：Prompt组合上下文和问题，LLM生成答案。",
        "category": "factual",
        "difficulty": "easy",
        "keywords": ["RAG", "检索增强生成", "索引", "检索", "生成"]
    },
    {
        "id": "doc_003",
        "query": "LangGraph是什么？在本项目中有什么作用？",
        "ground_truth": "LangGraph是多智能体编排框架，基于状态图（StateGraph）实现。本项目中用于编排Researcher（研究员）和Writer（作家）两个Agent的协作流程，支持条件边和工具调用。",
        "category": "factual",
        "difficulty": "medium",
        "keywords": ["LangGraph", "StateGraph", "多智能体", "编排"]
    },
    {
        "id": "doc_004",
        "query": "项目中使用了哪些技术栈？",
        "ground_truth": "技术栈包括：前端Streamlit、服务层ChatService/KBService/DocumentService、Agent层LangGraph+LangChain、RAG层ETL+VectorStore+Retriever+Generator、存储层ChromaDB+SQLite、模型层OpenAI API（GPT-3.5/4 + text-embedding-3-small）。",
        "category": "factual",
        "difficulty": "easy",
        "keywords": ["技术栈", "Streamlit", "LangGraph", "ChromaDB", "OpenAI"]
    },
    {
        "id": "doc_005",
        "query": "多智能体是如何协作的？",
        "ground_truth": "采用流水线协作模式：Researcher（研究员）负责分析用户意图、决定是否需要工具、调用工具获取信息；然后传递给Writer（作家）节点，整理信息并生成最终回答。",
        "category": "factual",
        "difficulty": "medium",
        "keywords": ["多智能体", "协作", "Researcher", "Writer", "流水线"]
    },
    
    # ==================== 技术实现类 (8题) ====================
    {
        "id": "doc_006",
        "query": "ETL模块是如何处理文档的？",
        "ground_truth": "ETL模块（etl.py）负责文档加载和切分。支持多种格式（TXT、PDF、DOCX、MD、代码文件），使用临时文件技术处理上传文件，采用RecursiveCharacterTextSplitter进行智能切分，支持中文和Markdown分隔符策略。",
        "category": "technical",
        "difficulty": "medium",
        "keywords": ["ETL", "文档加载", "切分", "RecursiveCharacterTextSplitter"]
    },
    {
        "id": "doc_007",
        "query": "文本切分使用了什么策略？CHUNK_SIZE和CHUNK_OVERLAP是多少？",
        "ground_truth": "使用RecursiveCharacterTextSplitter进行递归切分。默认CHUNK_SIZE=800字符，CHUNK_OVERLAP=100字符。中文分隔符优先级：段落>行>句号>其他标点；Markdown分隔符包括代码块、标题等。",
        "category": "technical",
        "difficulty": "medium",
        "keywords": ["切分策略", "CHUNK_SIZE", "CHUNK_OVERLAP", "分隔符"]
    },
    {
        "id": "doc_008",
        "query": "向量数据库是如何实现的？如何实现知识库隔离？",
        "ground_truth": "使用ChromaDB作为向量数据库，通过OpenAI的text-embedding-3-small模型进行向量化。知识库隔离通过元数据过滤（Metadata Filtering）实现：入库时添加project_id到metadata，检索时通过filter参数过滤。",
        "category": "technical",
        "difficulty": "medium",
        "keywords": ["ChromaDB", "向量数据库", "知识库隔离", "Metadata Filtering", "project_id"]
    },
    {
        "id": "doc_009",
        "query": "检索模块是如何工作的？默认返回多少个结果？",
        "ground_truth": "VectorRetriever使用余弦相似度进行语义检索，通过similarity_search_with_score方法查询。默认top_k=3，返回相似度最高的3个文档块，支持通过filter参数进行元数据过滤。",
        "category": "technical",
        "difficulty": "medium",
        "keywords": ["检索", "VectorRetriever", "余弦相似度", "top_k"]
    },
    {
        "id": "doc_010",
        "query": "生成模块是如何构建Prompt的？",
        "ground_truth": "RAGGenerator使用ChatPromptTemplate构建Prompt模板，包含【检索到的上下文】、【用户问题】和【回答规则】。使用LCEL（LangChain Expression Language）链式调用：prompt_template | llm | StrOutputParser()。",
        "category": "technical",
        "difficulty": "medium",
        "keywords": ["生成", "Prompt", "ChatPromptTemplate", "LCEL"]
    },
    {
        "id": "doc_011",
        "query": "Agent中有哪些可用的工具？",
        "ground_truth": "工具包括：ask_knowledge_base（知识库检索）、list_knowledge_base_files（列出文件）、search_by_filename（按文件名搜索）、general_qa（通用问答）、summarize_text（总结）、translate_text（翻译）、analyze_code（代码分析）、get_current_time（获取时间）、calculate_expression（数学计算）。",
        "category": "technical",
        "difficulty": "medium",
        "keywords": ["工具", "ask_knowledge_base", "general_qa", "Tool"]
    },
    {
        "id": "doc_012",
        "query": "ToolNode是如何工作的？",
        "ground_truth": "ToolNode是LangGraph预置的工具执行节点。当LLM响应中包含tool_calls时，ToolNode自动识别并执行对应工具，返回ToolMessage，然后回到Researcher节点继续处理。",
        "category": "technical",
        "difficulty": "hard",
        "keywords": ["ToolNode", "tool_calls", "ToolMessage", "工具执行"]
    },
    {
        "id": "doc_013",
        "query": "AgentState是如何定义的？消息是如何聚合的？",
        "ground_truth": "AgentState使用TypedDict定义，包含messages和next_step字段。messages使用Annotated[List[BaseMessage], operator.add]声明，表示新消息会追加到列表而非覆盖，这是LangGraph的状态聚合机制。",
        "category": "technical",
        "difficulty": "hard",
        "keywords": ["AgentState", "TypedDict", "Annotated", "operator.add", "状态聚合"]
    },
    
    # ==================== 架构设计类 (5题) ====================
    {
        "id": "doc_014",
        "query": "系统的整体架构是什么样的？",
        "ground_truth": "分层架构：用户界面(Streamlit) → Service层(ChatService/KBService/DocumentService) → Agent层(LangGraph状态机) → RAG层(ETL/VectorStore/Retriever/Generator) → 存储层(ChromaDB+SQLite) → 模型层(OpenAI API)。",
        "category": "architecture",
        "difficulty": "medium",
        "keywords": ["架构", "分层", "Service", "Agent", "RAG"]
    },
    {
        "id": "doc_015",
        "query": "Service层有哪些服务？各自的职责是什么？",
        "ground_truth": "Service层包括：KnowledgeBaseService（知识库CRUD、会话管理、统计信息）、ChatService（消息存取、Agent调用、流式响应）、DocumentService（文件上传处理、入库流程）。",
        "category": "architecture",
        "difficulty": "medium",
        "keywords": ["Service", "KnowledgeBaseService", "ChatService", "DocumentService"]
    },
    {
        "id": "doc_016",
        "query": "数据流向是怎样的？从用户提问到获得回答",
        "ground_truth": "流程：用户提问→ChatService保存消息→Agent Graph→Researcher分析→判断是否需要工具→ToolNode执行工具（如ask_knowledge_base）→VectorRetriever检索→返回结果→Researcher继续→Writer生成回答→保存响应→返回用户。",
        "category": "architecture",
        "difficulty": "medium",
        "keywords": ["数据流", "流程", "Researcher", "Writer", "ToolNode"]
    },
    {
        "id": "doc_017",
        "query": "SQLite数据库中有哪些表？",
        "ground_truth": "SQLite包含4个表：projects（知识库元信息）、sessions（会话管理）、messages（消息历史，包含role和content）、project_files（文件记录，包含chunks_count）。",
        "category": "architecture",
        "difficulty": "easy",
        "keywords": ["SQLite", "表结构", "projects", "sessions", "messages"]
    },
    {
        "id": "doc_018",
        "query": "LangGraph图是如何编排的？有哪些节点和边？",
        "ground_truth": "图编排：入口→researcher节点→条件判断(should_continue)→tools节点或writer节点→tools回到researcher→writer到END。节点包括researcher、writer、tools；边包括条件边和固定边。",
        "category": "architecture",
        "difficulty": "hard",
        "keywords": ["LangGraph", "图编排", "节点", "边", "条件边"]
    },
    
    # ==================== 使用指南类 (4题) ====================
    {
        "id": "doc_019",
        "query": "如何启动Web应用？",
        "ground_truth": "使用命令 streamlit run web_app.py 启动Web应用。也可以通过 python main.py 启动命令行交互模式。",
        "category": "usage",
        "difficulty": "easy",
        "keywords": ["启动", "Web应用", "streamlit", "web_app.py"]
    },
    {
        "id": "doc_020",
        "query": "如何添加新的文档类型支持？",
        "ground_truth": "在src/rag/etl.py的_select_loader方法中添加新的分支，例如：elif suffix == '.xlsx': return YourExcelLoader(file_path)。需要安装对应的Loader依赖。",
        "category": "usage",
        "difficulty": "medium",
        "keywords": ["扩展", "文档类型", "Loader", "etl.py"]
    },
    {
        "id": "doc_021",
        "query": "如何切换LLM模型或Embedding模型？",
        "ground_truth": "在config/settings.py中修改：CHAT_MODEL = 'gpt-4' 切换对话模型；EMBEDDING_MODEL = 'text-embedding-3-large' 切换向量化模型。",
        "category": "usage",
        "difficulty": "easy",
        "keywords": ["配置", "模型", "CHAT_MODEL", "EMBEDDING_MODEL", "settings.py"]
    },
    {
        "id": "doc_022",
        "query": "环境变量如何配置？",
        "ground_truth": "创建.env文件，配置：OPENAI_API_KEY=sk-xxxx（API密钥）和OPENAI_API_BASE=https://api.openai.com/v1（API地址，支持代理或兼容接口）。",
        "category": "usage",
        "difficulty": "easy",
        "keywords": ["环境变量", ".env", "OPENAI_API_KEY", "OPENAI_API_BASE"]
    },
    
    # ==================== 故障排查类 (3题) ====================
    {
        "id": "doc_023",
        "query": "为什么检索不到内容？",
        "ground_truth": "可能原因：1)文档未正确入库（检查data/vector_db/目录）；2)project_id不匹配（确认当前知识库）；3)向量数据库损坏（删除data/vector_db/重新入库）。",
        "category": "troubleshooting",
        "difficulty": "medium",
        "keywords": ["故障", "检索不到", "入库", "project_id"]
    },
    {
        "id": "doc_024",
        "query": "如何提高检索质量？",
        "ground_truth": "优化方向：1)调整CHUNK_SIZE和CHUNK_OVERLAP参数；2)使用更好的Embedding模型；3)增加重排序（Rerank）步骤；4)优化切分策略（按段落、按章节）。",
        "category": "troubleshooting",
        "difficulty": "medium",
        "keywords": ["优化", "检索质量", "CHUNK_SIZE", "Rerank", "切分策略"]
    },
    {
        "id": "doc_025",
        "query": "多智能体为什么会循环调用？如何排查？",
        "ground_truth": "排查方法：1)检查should_continue函数的逻辑是否正确；2)确认Researcher的System Prompt是否正确引导；3)添加最大迭代次数限制防止无限循环。",
        "category": "troubleshooting",
        "difficulty": "hard",
        "keywords": ["故障", "循环调用", "should_continue", "迭代次数"]
    },
]


@dataclass
class RAGTestResult:
    """RAG测试结果"""
    test_id: str
    query: str
    expected_answer: str
    actual_answer: str
    retrieved_chunks: List[str]
    retrieval_scores: List[float]
    
    # 检索指标
    precision: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    
    # 生成指标
    faithfulness: float = 0.0
    relevance: float = 0.0
    completeness: float = 0.0
    
    # 性能指标
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # 元数据
    category: str = ""
    difficulty: str = ""
    passed: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


class RAGTestSuite:
    """RAG系统测试套件"""
    
    def __init__(self, output_dir: str = None):
        from config.settings import settings
        self.output_dir = Path(output_dir or (settings.BASE_DIR / "metrics" / "rag_tests"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_cases = PROJECT_DOC_TEST_CASES
        self.results: List[RAGTestResult] = []
        
        logger.info(f"✅ RAG测试套件初始化完成，共 {len(self.test_cases)} 个测试用例")
    
    def run_test(
        self,
        retriever_func,
        generator_func,
        project_id: str = "default",
        top_k: int = 3
    ) -> List[RAGTestResult]:
        """
        运行完整的RAG测试
        
        Args:
            retriever_func: 检索函数 (query, project_id, top_k) -> [(doc, score), ...]
            generator_func: 生成函数 (query, context, project_id) -> response
            project_id: 知识库ID
            top_k: 检索数量
        """
        logger.info(f"🚀 开始RAG测试，共 {len(self.test_cases)} 个用例")
        
        self.results = []
        
        for i, case in enumerate(self.test_cases):
            logger.info(f"📝 测试 {i+1}/{len(self.test_cases)}: {case['id']} - {case['query'][:30]}...")
            
            result = self._run_single_test(
                case=case,
                retriever_func=retriever_func,
                generator_func=generator_func,
                project_id=project_id,
                top_k=top_k
            )
            
            self.results.append(result)
        
        return self.results
    
    def _run_single_test(
        self,
        case: Dict,
        retriever_func,
        generator_func,
        project_id: str,
        top_k: int
    ) -> RAGTestResult:
        """运行单个测试用例"""
        query = case["query"]
        expected = case["ground_truth"]
        
        result = RAGTestResult(
            test_id=case["id"],
            query=query,
            expected_answer=expected,
            actual_answer="",
            retrieved_chunks=[],
            retrieval_scores=[],
            category=case.get("category", "general"),
            difficulty=case.get("difficulty", "medium")
        )
        
        # 1. 检索阶段
        retrieval_start = time.time()
        try:
            retrieved = retriever_func(query, project_id, top_k)
            result.retrieved_chunks = [doc.page_content[:200] for doc, _ in retrieved]
            result.retrieval_scores = [score for _, score in retrieved]
            
            # 评估检索质量（基于关键词匹配）
            result.precision, result.recall = self._evaluate_retrieval_keywords(
                query, retrieved, case.get("keywords", [])
            )
            result.mrr = self._calculate_mrr(retrieved, case.get("keywords", []))
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            retrieved = []
        
        result.retrieval_latency_ms = (time.time() - retrieval_start) * 1000
        
        # 2. 生成阶段
        generation_start = time.time()
        try:
            context = "\n".join([doc.page_content for doc, _ in retrieved])
            response = generator_func(query, context, project_id)
            result.actual_answer = response
            
            # 评估生成质量
            result.faithfulness = self._evaluate_faithfulness(response, context)
            result.relevance = self._evaluate_relevance(query, response)
            result.completeness = self._evaluate_completeness(response, expected)
            
        except Exception as e:
            logger.error(f"生成失败: {e}")
            result.actual_answer = f"ERROR: {str(e)}"
        
        result.generation_latency_ms = (time.time() - generation_start) * 1000
        result.total_latency_ms = result.retrieval_latency_ms + result.generation_latency_ms
        
        # 判断是否通过（综合评分 > 0.6）
        avg_score = (result.relevance + result.completeness) / 2
        result.passed = avg_score > 0.6
        
        return result
    
    def _evaluate_retrieval_keywords(
        self, 
        query: str, 
        retrieved: List, 
        expected_keywords: List[str]
    ) -> Tuple[float, float]:
        """基于关键词评估检索质量"""
        if not expected_keywords or not retrieved:
            return 0.5, 0.5
        
        # 检查检索结果中包含多少关键词
        found_keywords = set()
        for doc, _ in retrieved:
            content_lower = doc.page_content.lower()
            for kw in expected_keywords:
                if kw.lower() in content_lower:
                    found_keywords.add(kw)
        
        precision = len(found_keywords) / len(retrieved) if retrieved else 0
        recall = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
        
        return precision, recall
    
    def _calculate_mrr(self, retrieved: List, expected_keywords: List[str]) -> float:
        """计算MRR"""
        if not expected_keywords or not retrieved:
            return 0.0
        
        for i, (doc, _) in enumerate(retrieved):
            content_lower = doc.page_content.lower()
            for kw in expected_keywords:
                if kw.lower() in content_lower:
                    return 1.0 / (i + 1)
        
        return 0.0
    
    def _evaluate_faithfulness(self, response: str, context: str) -> float:
        """评估忠实度"""
        if not context:
            return 0.5
        
        response_words = set(response.lower())
        context_words = set(context.lower())
        
        if not response_words:
            return 0.0
        
        overlap = response_words & context_words
        return len(overlap) / len(response_words)
    
    def _evaluate_relevance(self, query: str, response: str) -> float:
        """评估相关性"""
        query_words = set(query.lower())
        response_words = set(response.lower())
        
        if not query_words:
            return 0.5
        
        overlap = query_words & response_words
        return min(1.0, len(overlap) / len(query_words) + 0.3)
    
    def _evaluate_completeness(self, response: str, expected: str) -> float:
        """评估完整性"""
        if not expected:
            return 0.7 if len(response) > 20 else 0.3
        
        expected_words = set(expected.lower())
        response_words = set(response.lower())
        
        overlap = expected_words & response_words
        return len(overlap) / len(expected_words) if expected_words else 0.5
    
    def get_summary(self) -> Dict:
        """获取测试摘要"""
        if not self.results:
            return {"error": "没有测试结果"}
        
        # 检索指标汇总
        avg_precision = sum(r.precision for r in self.results) / len(self.results)
        avg_recall = sum(r.recall for r in self.results) / len(self.results)
        avg_mrr = sum(r.mrr for r in self.results) / len(self.results)
        
        # 生成指标汇总
        avg_faithfulness = sum(r.faithfulness for r in self.results) / len(self.results)
        avg_relevance = sum(r.relevance for r in self.results) / len(self.results)
        avg_completeness = sum(r.completeness for r in self.results) / len(self.results)
        
        # 性能指标汇总
        avg_retrieval_latency = sum(r.retrieval_latency_ms for r in self.results) / len(self.results)
        avg_generation_latency = sum(r.generation_latency_ms for r in self.results) / len(self.results)
        avg_total_latency = sum(r.total_latency_ms for r in self.results) / len(self.results)
        
        # 通过率
        passed_count = sum(1 for r in self.results if r.passed)
        pass_rate = passed_count / len(self.results)
        
        # 按分类统计
        by_category = {}
        for r in self.results:
            cat = r.category or "unknown"
            if cat not in by_category:
                by_category[cat] = {"count": 0, "passed": 0, "avg_relevance": []}
            by_category[cat]["count"] += 1
            if r.passed:
                by_category[cat]["passed"] += 1
            by_category[cat]["avg_relevance"].append(r.relevance)
        
        for cat in by_category:
            rels = by_category[cat]["avg_relevance"]
            by_category[cat]["avg_relevance"] = sum(rels) / len(rels) if rels else 0
            by_category[cat]["pass_rate"] = by_category[cat]["passed"] / by_category[cat]["count"]
        
        return {
            "total_tests": len(self.results),
            "passed": passed_count,
            "failed": len(self.results) - passed_count,
            "pass_rate": pass_rate,
            
            "retrieval_metrics": {
                "avg_precision": round(avg_precision, 3),
                "avg_recall": round(avg_recall, 3),
                "avg_mrr": round(avg_mrr, 3)
            },
            
            "generation_metrics": {
                "avg_faithfulness": round(avg_faithfulness, 3),
                "avg_relevance": round(avg_relevance, 3),
                "avg_completeness": round(avg_completeness, 3)
            },
            
            "performance_metrics": {
                "avg_retrieval_latency_ms": round(avg_retrieval_latency, 2),
                "avg_generation_latency_ms": round(avg_generation_latency, 2),
                "avg_total_latency_ms": round(avg_total_latency, 2)
            },
            
            "by_category": by_category
        }
    
    def export_report(self, filename: str = None) -> str:
        """导出测试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"rag_test_report_{timestamp}.json"
        report_path = self.output_dir / filename
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "test_cases": self.test_cases,
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 测试报告已导出: {report_path}")
        return str(report_path)
    
    def print_summary(self):
        """打印测试摘要"""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("📊 RAG系统测试报告")
        print("="*70)
        
        print(f"\n📈 总体结果:")
        print(f"   总测试数: {summary['total_tests']}")
        print(f"   通过: {summary['passed']}")
        print(f"   失败: {summary['failed']}")
        print(f"   通过率: {summary['pass_rate']:.1%}")
        
        print(f"\n🎯 检索指标:")
        rm = summary['retrieval_metrics']
        print(f"   平均精确率: {rm['avg_precision']:.3f}")
        print(f"   平均召回率: {rm['avg_recall']:.3f}")
        print(f"   平均MRR: {rm['avg_mrr']:.3f}")
        
        print(f"\n📝 生成指标:")
        gm = summary['generation_metrics']
        print(f"   平均忠实度: {gm['avg_faithfulness']:.3f}")
        print(f"   平均相关性: {gm['avg_relevance']:.3f}")
        print(f"   平均完整性: {gm['avg_completeness']:.3f}")
        
        print(f"\n⚡ 性能指标:")
        pm = summary['performance_metrics']
        print(f"   平均检索延迟: {pm['avg_retrieval_latency_ms']:.2f} ms")
        print(f"   平均生成延迟: {pm['avg_generation_latency_ms']:.2f} ms")
        print(f"   平均总延迟: {pm['avg_total_latency_ms']:.2f} ms")
        
        print(f"\n📂 按分类统计:")
        for cat, stats in summary.get('by_category', {}).items():
            print(f"   {cat}: {stats['count']}个用例, 通过率 {stats['pass_rate']:.1%}")
        
        print("="*70)


def create_test_dataset_file():
    """创建测试数据集文件"""
    from config.settings import settings
    
    data_dir = settings.BASE_DIR / "data" / "test_datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = data_dir / "project_doc_test_cases.json"
    
    dataset = {
        "name": "PROJECT_DOCUMENTATION测试集",
        "description": "基于PROJECT_DOCUMENTATION.md构建的标准化测试问答集",
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "test_cases": PROJECT_DOC_TEST_CASES,
        "statistics": {
            "total_cases": len(PROJECT_DOC_TEST_CASES),
            "by_category": {},
            "by_difficulty": {}
        }
    }
    
    # 统计
    for case in PROJECT_DOC_TEST_CASES:
        cat = case.get("category", "unknown")
        diff = case.get("difficulty", "unknown")
        
        dataset["statistics"]["by_category"][cat] = dataset["statistics"]["by_category"].get(cat, 0) + 1
        dataset["statistics"]["by_difficulty"][diff] = dataset["statistics"]["by_difficulty"].get(diff, 0) + 1
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 测试数据集已创建: {output_path}")
    print(f"   总用例数: {len(PROJECT_DOC_TEST_CASES)}")
    print(f"   分类: {dataset['statistics']['by_category']}")
    print(f"   难度: {dataset['statistics']['by_difficulty']}")
    
    return str(output_path)


if __name__ == "__main__":
    # 创建测试数据集文件
    create_test_dataset_file()
    
    print("\n" + "="*70)
    print("📋 测试用例预览")
    print("="*70)
    
    for case in PROJECT_DOC_TEST_CASES[:5]:
        print(f"\n[{case['id']}] [{case['category']}] [{case['difficulty']}]")
        print(f"Q: {case['query']}")
        print(f"A: {case['ground_truth'][:100]}...")