import logging
from pathlib import Path
from typing import Optional, Union


def get_logger(config_or_name: Union[dict, str], log_file: Optional[Path] = None, level=logging.INFO):
    """
    支持两种调用方式：
      get_logger(config)                  → 使用 config 配置（兼容 main.py）
      get_logger("batch_run", log_file=…) → 简化调用（兼容 batch_run_all_to_one.py）
    """
    if isinstance(config_or_name, dict):
        config = config_or_name
        name = "fiji_stitcher"
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger
        level_name = str(config.get("LOG_LEVEL", "INFO")).upper()
        logger.setLevel(getattr(logging, level_name, logging.INFO))
        stitched_parent = Path(config["STITCHED_PARENT_DIR"])
        log_dir = stitched_parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = log_dir / config["RUN_LOG_FILENAME"]
    else:
        name = config_or_name
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger
        logger.setLevel(level)
        if log_file is not None:
            log_dir = log_file.parent
            log_dir.mkdir(parents=True, exist_ok=True)
            logfile = log_file
        else:
            logfile = None

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if logfile is not None:
        fh = logging.FileHandler(str(logfile), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if logfile is not None:
        logger.info("Logging to: %s", logfile)
    return logger
