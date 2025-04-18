import csv
from pathlib import Path
import datetime


class Logger:
    def __init__(self, subfolder, file_path, headers):
        script_dir = Path(__file__).resolve().parent
        logs_dir = script_dir / "logs" / subfolder
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = logs_dir / file_path

        # Ensure the first column is "timestamp"
        if headers[0].lower() != "timestamp":
            headers.insert(0, "timestamp")
        self.headers = headers
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
