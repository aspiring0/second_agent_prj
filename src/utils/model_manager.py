# src/utils/model_manager.py
"""
模型管理器 - Model Manager
统一管理所有可用的大语言模型和Embedding模型
支持动态切换不同的模型提供商
"""
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("MODEL_MANAGER")


class ModelProvider(Enum):
    """模型提供商"""
    OPENAI = "openai"
    AZURE = "azure"
    DEEPSEEK = "deepseek"
    ZHIPU = "zhipu"
    MOONSHOT = "moonshot"
    CUSTOM = "custom"


@dataclass
class ChatModelConfig:
    """对话模型配置"""
    id: str                          # 模型ID（唯一标识）
    name: str                        # 显示名称
    model_name: str                  # 实际调用的模型名
    provider: ModelProvider          # 提供商
    base_url: Optional[str] = None   # API基础URL（None则使用默认）
    api_key_env: str = "OPENAI_API_KEY"  # API Key环境变量名
    max_tokens: int = 4096           # 最大输出token
    temperature: float = 0.1         # 默认温度
    description: str = ""            # 模型描述
    supports_tools: bool = True      # 是否支持工具调用
    supports_vision: bool = False    # 是否支持视觉


@dataclass
class EmbeddingModelConfig:
    """Embedding模型配置"""
    id: str                          # 模型ID
    name: str                        # 显示名称
    model_name: str                  # 实际调用的模型名
    provider: ModelProvider          # 提供商
    base_url: Optional[str] = None   # API基础URL
    api_key_env: str = "OPENAI_API_KEY"
    dimension: int = 1536            # 向量维度
    description: str = ""


# ==================== 预定义模型列表 ====================

CHAT_MODELS: Dict[str, ChatModelConfig] = {
    # OpenAI 模型
    "gpt-4o": ChatModelConfig(
        id="gpt-4o",
        name="GPT-4o",
        model_name="gpt-4o",
        provider=ModelProvider.OPENAI,
        max_tokens=4096,
        description="OpenAI最新旗舰模型，性能强大，支持多模态",
        supports_vision=True
    ),
    "gpt-4o-mini": ChatModelConfig(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        model_name="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        max_tokens=4096,
        description="轻量版GPT-4o，性价比高",
        supports_vision=True
    ),
    "gpt-4-turbo": ChatModelConfig(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        model_name="gpt-4-turbo",
        provider=ModelProvider.OPENAI,
        max_tokens=4096,
        description="GPT-4增强版，支持128K上下文"
    ),
    "gpt-4": ChatModelConfig(
        id="gpt-4",
        name="GPT-4",
        model_name="gpt-4",
        provider=ModelProvider.OPENAI,
        max_tokens=4096,
        description="OpenAI旗舰模型，推理能力强"
    ),
    "gpt-3.5-turbo": ChatModelConfig(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        model_name="gpt-3.5-turbo",
        provider=ModelProvider.OPENAI,
        max_tokens=4096,
        description="快速且经济的通用模型"
    ),
    
    # DeepSeek 模型
    "deepseek-chat": ChatModelConfig(
        id="deepseek-chat",
        name="DeepSeek Chat",
        model_name="deepseek-chat",
        provider=ModelProvider.DEEPSEEK,
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        max_tokens=4096,
        description="DeepSeek对话模型，中文能力强",
        supports_tools=True
    ),
    "deepseek-reasoner": ChatModelConfig(
        id="deepseek-reasoner",
        name="DeepSeek Reasoner",
        model_name="deepseek-reasoner",
        provider=ModelProvider.DEEPSEEK,
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        max_tokens=4096,
        description="DeepSeek推理模型，适合复杂任务",
        supports_tools=False
    ),
    
    # 智谱AI 模型
    "glm-4": ChatModelConfig(
        id="glm-4",
        name="GLM-4",
        model_name="glm-4",
        provider=ModelProvider.ZHIPU,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="ZHIPU_API_KEY",
        max_tokens=4096,
        description="智谱AI旗舰模型，中文理解强",
        supports_tools=True
    ),
    "glm-4-flash": ChatModelConfig(
        id="glm-4-flash",
        name="GLM-4 Flash",
        model_name="glm-4-flash",
        provider=ModelProvider.ZHIPU,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="ZHIPU_API_KEY",
        max_tokens=4096,
        description="智谱AI快速模型，免费额度大",
        supports_tools=True
    ),
    
    # Moonshot 模型
    "moonshot-v1-8k": ChatModelConfig(
        id="moonshot-v1-8k",
        name="Moonshot V1 8K",
        model_name="moonshot-v1-8k",
        provider=ModelProvider.MOONSHOT,
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        max_tokens=4096,
        description="月之暗面对话模型，长文本能力突出",
        supports_tools=False
    ),
    "moonshot-v1-32k": ChatModelConfig(
        id="moonshot-v1-32k",
        name="Moonshot V1 32K",
        model_name="moonshot-v1-32k",
        provider=ModelProvider.MOONSHOT,
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        max_tokens=8192,
        description="月之暗面长文本模型",
        supports_tools=False
    ),
}

EMBEDDING_MODELS: Dict[str, EmbeddingModelConfig] = {
    # OpenAI Embedding
    "text-embedding-3-small": EmbeddingModelConfig(
        id="text-embedding-3-small",
        name="OpenAI Embedding v3 Small",
        model_name="text-embedding-3-small",
        provider=ModelProvider.OPENAI,
        dimension=1536,
        description="OpenAI最新小维度Embedding，性价比高"
    ),
    "text-embedding-3-large": EmbeddingModelConfig(
        id="text-embedding-3-large",
        name="OpenAI Embedding v3 Large",
        model_name="text-embedding-3-large",
        provider=ModelProvider.OPENAI,
        dimension=3072,
        description="OpenAI最新大维度Embedding，效果最好"
    ),
    "text-embedding-ada-002": EmbeddingModelConfig(
        id="text-embedding-ada-002",
        name="OpenAI Ada 002",
        model_name="text-embedding-ada-002",
        provider=ModelProvider.OPENAI,
        dimension=1536,
        description="OpenAI经典Embedding模型"
    ),
    
    # 智谱 Embedding
    "embedding-2": EmbeddingModelConfig(
        id="embedding-2",
        name="智谱 Embedding-2",
        model_name="embedding-2",
        provider=ModelProvider.ZHIPU,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="ZHIPU_API_KEY",
        dimension=1024,
        description="智谱AI Embedding模型"
    ),
}


class ModelManager:
    """
    模型管理器
    负责创建和管理所有模型实例
    """
    
    def __init__(self):
        self._chat_cache: Dict[str, BaseChatModel] = {}
        self._embedding_cache: Dict[str, Embeddings] = {}
        self._current_chat_model: str = settings.CHAT_MODEL
        self._current_embedding_model: str = settings.EMBEDDING_MODEL
        
        logger.info("模型管理器初始化完成")
        logger.info(f"   当前Chat模型: {self._current_chat_model}")
        logger.info(f"   当前Embedding模型: {self._current_embedding_model}")
    
    # ==================== 模型列表获取 ====================
    
    def list_chat_models(self) -> List[ChatModelConfig]:
        """获取所有可用的对话模型列表"""
        return list(CHAT_MODELS.values())
    
    def list_embedding_models(self) -> List[EmbeddingModelConfig]:
        """获取所有可用的Embedding模型列表"""
        return list(EMBEDDING_MODELS.values())
    
    def get_chat_model_config(self, model_id: str) -> Optional[ChatModelConfig]:
        """获取对话模型配置"""
        return CHAT_MODELS.get(model_id)
    
    def get_embedding_model_config(self, model_id: str) -> Optional[EmbeddingModelConfig]:
        """获取Embedding模型配置"""
        return EMBEDDING_MODELS.get(model_id)
    
    # ==================== 当前模型设置 ====================
    
    def set_current_chat_model(self, model_id: str) -> bool:
        """设置当前使用的对话模型"""
        if model_id not in CHAT_MODELS:
            logger.error(f"未知的对话模型: {model_id}")
            return False
        
        config = CHAT_MODELS[model_id]
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            logger.error(f"缺少API Key: {config.api_key_env}")
            return False
        
        self._current_chat_model = model_id
        logger.info(f"切换对话模型: {config.name}")
        return True
    
    def set_current_embedding_model(self, model_id: str) -> bool:
        """设置当前使用的Embedding模型"""
        if model_id not in EMBEDDING_MODELS:
            logger.error(f"未知的Embedding模型: {model_id}")
            return False
        
        config = EMBEDDING_MODELS[model_id]
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            logger.error(f"缺少API Key: {config.api_key_env}")
            return False
        
        self._current_embedding_model = model_id
        # 清空Embedding缓存，因为切换了模型
        self._embedding_cache.clear()
        logger.info(f"切换Embedding模型: {config.name}")
        return True
    
    def get_current_chat_model_id(self) -> str:
        """获取当前对话模型ID"""
        return self._current_chat_model
    
    def get_current_embedding_model_id(self) -> str:
        """获取当前Embedding模型ID"""
        return self._current_embedding_model
    
    # ==================== 模型实例获取 ====================
    
    def get_chat_model(
        self, 
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        获取对话模型实例
        
        Args:
            model_id: 模型ID，None则使用当前模型
            temperature: 温度参数
            **kwargs: 其他参数传递给ChatOpenAI
        """
        target_model = model_id or self._current_chat_model
        config = CHAT_MODELS.get(target_model)
        
        if not config:
            logger.warning(f"未找到模型配置 {target_model}，使用默认配置")
            # 使用默认配置
            return ChatOpenAI(
                model=target_model,
                temperature=temperature or 0.1,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_BASE_URL,
                **kwargs
            )
        
        # 获取API Key
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"请设置环境变量: {config.api_key_env}")
        
        # 获取Base URL
        base_url = config.base_url or os.getenv("OPENAI_API_BASE")
        
        # 构建缓存键
        cache_key = f"{target_model}_{temperature}_{hash(frozenset(kwargs.items()))}"
        
        if cache_key not in self._chat_cache:
            self._chat_cache[cache_key] = ChatOpenAI(
                model=config.model_name,
                temperature=temperature or config.temperature,
                openai_api_key=api_key,
                openai_api_base=base_url,
                max_tokens=config.max_tokens,
                **kwargs
            )
            logger.debug(f"创建新的Chat模型实例: {config.name}")
        
        return self._chat_cache[cache_key]
    
    def get_embedding_model(self, model_id: Optional[str] = None) -> Embeddings:
        """
        获取Embedding模型实例
        
        Args:
            model_id: 模型ID，None则使用当前模型
        """
        target_model = model_id or self._current_embedding_model
        config = EMBEDDING_MODELS.get(target_model)
        
        if not config:
            logger.warning(f"未找到Embedding模型配置 {target_model}，使用默认配置")
            return OpenAIEmbeddings(
                model=target_model,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_BASE_URL
            )
        
        # 获取API Key
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"请设置环境变量: {config.api_key_env}")
        
        # 获取Base URL
        base_url = config.base_url or os.getenv("OPENAI_API_BASE")
        
        if target_model not in self._embedding_cache:
            self._embedding_cache[target_model] = OpenAIEmbeddings(
                model=config.model_name,
                openai_api_key=api_key,
                openai_api_base=base_url
            )
            logger.debug(f"创建新的Embedding模型实例: {config.name}")
        
        return self._embedding_cache[target_model]
    
    # ==================== 工具和状态 ====================
    
    def get_model_status(self) -> Dict:
        """获取模型状态信息"""
        chat_config = CHAT_MODELS.get(self._current_chat_model)
        embedding_config = EMBEDDING_MODELS.get(self._current_embedding_model)
        
        return {
            "current_chat_model": {
                "id": self._current_chat_model,
                "name": chat_config.name if chat_config else self._current_chat_model,
                "provider": chat_config.provider.value if chat_config else "unknown"
            },
            "current_embedding_model": {
                "id": self._current_embedding_model,
                "name": embedding_config.name if embedding_config else self._current_embedding_model,
                "dimension": embedding_config.dimension if embedding_config else 1536
            },
            "available_chat_models": len(CHAT_MODELS),
            "available_embedding_models": len(EMBEDDING_MODELS)
        }
    
    def check_model_available(self, model_id: str) -> tuple[bool, str]:
        """
        检查模型是否可用
        
        Returns:
            (是否可用, 原因说明)
        """
        # 检查对话模型
        if model_id in CHAT_MODELS:
            config = CHAT_MODELS[model_id]
            api_key = os.getenv(config.api_key_env)
            if api_key:
                return True, "可用"
            else:
                return False, f"缺少API Key: {config.api_key_env}"
        
        # 检查Embedding模型
        if model_id in EMBEDDING_MODELS:
            config = EMBEDDING_MODELS[model_id]
            api_key = os.getenv(config.api_key_env)
            if api_key:
                return True, "可用"
            else:
                return False, f"缺少API Key: {config.api_key_env}"
        
        return False, f"未知模型: {model_id}"
    
    def update_api_key(self, api_key: str, base_url: Optional[str] = None) -> bool:
        """
        更新 API Key 和 Base URL，清除缓存的模型实例

        Args:
            api_key: 新的 API Key
            base_url: 新的 API Base URL，None 则不修改

        Returns:
            是否更新成功
        """
        if not api_key or not api_key.strip():
            logger.error("API Key 不能为空")
            return False

        # 更新环境变量，使后续 os.getenv 读取到新值
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        if base_url is not None and base_url.strip():
            os.environ["OPENAI_API_BASE"] = base_url.strip()

        # 同步更新 settings 对象（单例已加载，需要手动刷新）
        settings.OPENAI_API_KEY = api_key.strip()
        if base_url is not None and base_url.strip():
            settings.OPENAI_BASE_URL = base_url.strip()

        # 清除所有缓存的模型实例，下次 get_chat_model / get_embedding_model 时重新创建
        self.clear_cache()

        logger.info("API Key 和 Base URL 已更新，模型缓存已清除")
        return True

    def get_available_models(self) -> List[Dict]:
        """
        返回可用模型列表（预设列表）

        Returns:
            模型信息字典列表，每项包含 id, name, provider 字段
        """
        models = []
        for model_id, config in CHAT_MODELS.items():
            models.append({
                "id": config.id,
                "name": config.name,
                "provider": config.provider.value,
                "model_name": config.model_name,
                "description": config.description,
            })
        return models

    def test_connection(self, api_key: str, base_url: Optional[str] = None, model_name: str = "gpt-4o-mini") -> tuple[bool, str]:
        """
        测试 API Key 是否有效

        Args:
            api_key: 要测试的 API Key
            base_url: API Base URL，None 使用默认
            model_name: 用于测试的模型名称

        Returns:
            (是否成功, 提示信息)
        """
        try:
            test_base_url = base_url or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            test_llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=test_base_url,
                max_tokens=10,
                temperature=0,
            )
            # 发送简单请求测试连接
            test_llm.invoke("Hi")
            return True, "连接成功，API Key 有效"
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"API Key 测试失败: {error_msg}")
            return False, f"连接失败: {error_msg}"

    def clear_cache(self):
        """清空模型缓存"""
        self._chat_cache.clear()
        self._embedding_cache.clear()
        logger.info("模型缓存已清空")


# 全局单例
model_manager = ModelManager()