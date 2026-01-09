# test_structure.py
from config.settings import settings
from src.utils.logger import setup_logger

# 初始化日志，名字叫 'SystemCheck'
logger = setup_logger("SystemCheck")

def check_system():
    logger.info("🚀 开始系统自检...")
    
    # 1. 检查路径配置
    logger.info(f"项目根目录: {settings.BASE_DIR}")
    logger.info(f"数据目录: {settings.DATA_DIR}")
    
    # 2. 检查 API Key (只显示前4位，保护隐私)
    key = settings.OPENAI_API_KEY
    if key:
        logger.info(f"API Key 状态: 已加载 ({key[:4]}...)")
    else:
        logger.error("API Key 状态: ❌ 未找到！")

    # 3. 模拟写入文件日志
    logger.info("这条消息应该同时出现在屏幕上和 logs/app.log 文件里。")
    logger.info("✅ 基础架构验证完成！")

if __name__ == "__main__":
    check_system()