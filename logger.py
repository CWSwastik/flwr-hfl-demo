import csv
import os
from pathlib import Path
import datetime
import sys
import logging
from config import EXPERIMENT_NAME

class RemoveWarningsFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARN


# Get the library's logger
lib_logger = logging.getLogger("flwr")
lib_logger.setLevel(logging.INFO)  # set low to allow everything through

# Create a handler that filters only INFO logs
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.addFilter(RemoveWarningsFilter())
handler.setFormatter(
    logging.Formatter("\x1b[32m%(name)s - %(levelname)s\x1b[0m: %(message)s")
)

# Clear existing handlers (optional depending on library behavior)
lib_logger.handlers = []
lib_logger.propagate = False  # don't pass logs to root logger
lib_logger.addHandler(handler)


class Logger:
    def __init__(self, subfolder, file_path, headers, init_file=True):
        # 1. Setup paths using os.path for strict cross-platform compatibility
        # Get directory of the current script (logger.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build path: script_dir/logs/EXPERIMENT_NAME/subfolder
        self.logs_dir = os.path.join(script_dir, "logs", EXPERIMENT_NAME, subfolder)
        
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Final full path to the log file
        self.file_path = os.path.join(self.logs_dir, file_path)

        if not headers or headers[0].lower() != "timestamp":
            headers.insert(0, "timestamp")
        self.headers = headers

        if init_file:
            if not os.path.exists(self.file_path):
                self._init_file()

    def _init_file(self):
        try:
            with open(self.file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
        except Exception as e:
            print(f"Error initializing log file {self.file_path}: {e}")

    def log(self, row_dict):
        row_dict["timestamp"] = datetime.datetime.now().isoformat()
        try:
            # Open in 'a' (Append) mode
            with open(self.file_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writerow(row_dict)
        except Exception as e:
            print(f"Error appending to log file {self.file_path}: {e}")
