import csv
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
        script_dir = Path(__file__).resolve().parent
        logs_dir = script_dir / "logs" / EXPERIMENT_NAME / subfolder
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = logs_dir / file_path

        # Ensure the first column is "timestamp"
        if headers[0].lower() != "timestamp":
            headers.insert(0, "timestamp")
        self.headers = headers
        if init_file:
            self._init_file()

    def _init_file(self):
        # if not self.file_path.exists():
        with self.file_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log(self, row_dict):
        # Add current timestamp to the row
        row_dict["timestamp"] = datetime.datetime.now().isoformat()
        with self.file_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(row_dict)
