import os
import logging
from datetime import datetime

def setup_logger(log_dir, log_filename=None):
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件路径
    if log_filename:
        log_file = os.path.join(log_dir, log_filename)
    else:
        log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 配置日志
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 检查是否已有相同的处理器
    has_file_handler = any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file for h in logger.handlers)
    has_console_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    
    # 创建文件处理器
    if not has_file_handler:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 创建控制台处理器
    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger
