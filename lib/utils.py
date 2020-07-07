import logging


def config_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    basic_format = '%(message)s'
    formatter = logging.Formatter(basic_format)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    fhlr = logging.FileHandler(log_dir)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger

