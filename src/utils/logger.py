import logging
from configparser import ConfigParser

def setup_logger(level=logging.INFO):
    logging.basicConfig(
        level=level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("WatermarkPipeline")

logger = setup_logger()
