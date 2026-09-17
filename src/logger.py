
import logging
import os


class Logger:
    def __init__(self,name: str, log_file: str, level: str):
        """
        Logger class for logging messages to a file.
        Args:
            name (str): The name of the logger.
            log_file (str): The path to the log file.
            level (str): The logging level (e.g., "debug", "info", "warning", "error", "critical").
        """
        log_dir = "./logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.dict_level = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
            }

        self.name = name
        self.log_file = log_file
        self.level = self.dict_level.get(level.lower(), logging.DEBUG)

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        self.setup(log_dir)


    def setup(self, log_dir: str):

        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)


        file_handle = logging.FileHandler(os.path.join(log_dir, self.log_file),mode='w')
        file_handle.setLevel(self.level)
        file_handle.setFormatter(formatter)
        self.logger.addHandler(file_handle) 

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)
