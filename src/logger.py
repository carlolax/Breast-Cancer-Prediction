import os
import logging
from datetime import datetime

LOG_DIR = "/Users/carlo/Developer/notebooks/Breast-Cancer-Prediction/logs" # Changes this with your preferred logs directory

def setup_logger(log_name, log_level=logging.INFO):
    logger = logging.getLogger(f'breast-cancer-prediction.{log_name}')
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    console_handler = logging.StreamHandler()

    console_handler.setLevel(log_level)
    console_handler.setFormatter(logger_formatter)
    logger.addHandler(console_handler)
    os.makedirs(LOG_DIR, exist_ok=True)

    log_file = os.path.join(LOG_DIR, f"{log_name}.log")
    file_handler = logging.FileHandler(log_file)

    file_handler.setLevel(log_level)
    file_handler.setFormatter(logger_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return logger
