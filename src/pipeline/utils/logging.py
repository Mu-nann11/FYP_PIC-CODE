"""
统一日志工具
支持控制台输出 + 文件日志，自动按项目结构组织日志目录
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from pipeline.config import BASE_DIR

# 默认日志目录
DEFAULT_LOG_DIR = BASE_DIR / "results" / "pipeline_logs"


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    name: str = "tma_pipeline",
    quiet: bool = False,
) -> logging.Logger:
    """
    配置日志系统

    Args:
        log_dir: 日志输出目录，默认使用 results/pipeline_logs
        level: 日志级别，默认 INFO
        name: logger 名称
        quiet: True 则禁用控制台输出（仅写入文件）
    """
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件名包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 控制台处理器（可选）
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "[%(levelname)s] %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "tma_pipeline") -> logging.Logger:
    """
    获取已配置的 logger 实例
    如果 logger 尚未配置，返回一个基础配置
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 尚未配置，进行基础配置
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger
