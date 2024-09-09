from logger import setup_logging
import pandas as pd
import config.proc_img as module_config
from config.parse_config import ConfigParser
from simce.utils import read_json
from config.proc_img import CURSO
from pathlib import Path
import logging
def get_logger(name, verbosity=2):
    log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
    msg_verbosity = f'verbosity option {verbosity} is invalid. Valid options are {log_levels.keys()}.'
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    return logger


setup_logging(Path('pruebas/'))
log = get_logger('prueba')
log.info('hola')

self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
config_dict = read_json('config/model.json')
config = ConfigParser(config_dict)