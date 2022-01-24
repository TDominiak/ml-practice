import logging

logger = logging.getLogger('ImageClassifier')


def setup_logger():
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
