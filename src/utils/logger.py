# src/utils/logger.py
import logging
import sys
# 引入刚才写好的配置，直接拿到日志存放路径，体现了配置集中的好处
from config.settings import settings

def setup_logger(name):
    """
    配置并返回一个 logger 对象
    :param name: 模块名称 (比如 'RAG_ETL')，方便知道日志是哪个模块打印的
    """
    
    # 1. 确保日志文件夹存在，如果不存在会自动创建 (mkdir -p)
    if not settings.LOG_DIR.exists():
        settings.LOG_DIR.mkdir(parents=True)

    # 2. 创建 logger 实例
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO) # 设置最低记录级别，低于 INFO 的调试信息会被忽略

    # 3. 如果这个 logger 已经被配置过（防止重复打印），直接返回
    if logger.handlers:
        return logger

    # 4. 定义日志格式：时间 - 模块名 - 级别 - 内容
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 5. 处理器 A：输出到控制台 (屏幕)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 6. 处理器 B：输出到文件 (app.log)
    # encoding='utf-8' 防止中文乱码
    file_path = settings.LOG_DIR / "app.log"
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # 7. 装载处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger