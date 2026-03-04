from datetime import datetime
import logging 
import os
import json
# 读取JSON文件

  
def setup_logging(log_dir: str):
    log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def redirect_logging_to_dir(logger, new_log_dir: str, exp_name: str = None):
    """
    将日志重定向到新目录
    
    Args:
        logger: 现有的logger对象
        new_log_dir: 新的日志目录
        exp_name: 实验名称（可选，用于日志文件名）
    
    Returns:
        新的日志文件路径
    """
    os.makedirs(new_log_dir, exist_ok=True)
    
    # 构建日志文件名
    if exp_name:
        log_filename = f"training_{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    log_filepath = os.path.join(new_log_dir, log_filename)
    
    # 移除现有的文件处理器（保留控制台处理器）
    handlers_to_remove = []
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handlers_to_remove.append(handler)
    
    for handler in handlers_to_remove:
        handler.close()
        logger.removeHandler(handler)
    
    # 添加新的文件处理器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    new_file_handler = logging.FileHandler(log_filepath)
    new_file_handler.setLevel(logging.INFO)
    new_file_handler.setFormatter(formatter)
    logger.addHandler(new_file_handler)
    
    return log_filepath