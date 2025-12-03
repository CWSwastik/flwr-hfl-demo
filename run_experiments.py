import json
import subprocess
import shutil
import os
import re
import time

CONFIG_FILE = "config.py"
BACKUP_FILE = "config_backup.py"


def read_config():
    with open(CONFIG_FILE, "r") as f:
        return f.read()


def write_config(text):
    with open(CONFIG_FILE, "w") as f:
        f.write(text)


def update_config(config_text, updates):
    for key, value in updates.items():
        if isinstance(value, str):
            replacement = f'{key} = "{value}"'
        else:
            replacement = f"{key} = {value}"

        pattern = rf"{key}\s*=.*"
        print(f"pattern: {pattern}, replacement: {replacement}")
        if re.search(pattern, config_text):
            config_text = re.sub(pattern, replacement, config_text, count=1)
        else:
            config_text += f"\n{replacement}\n"

    return config_text


def run_simulation():
    print("Running simulate.py...")
    result = subprocess.run(["python", "simulate.py"])
    if result.returncode != 0:
        print("ERROR running simulate.py")
    print("Finished simulate.py.")


def main():
    if not os.path.exists(BACKUP_FILE):
        shutil.copy(CONFIG_FILE, BACKUP_FILE)

    original_config = read_config()

    # ToDo: Accept experiments file as cli argument
    with open("experiments.json", "r") as f:
        experiments = json.load(f)

    for i, exp in enumerate(experiments):
        print(f"\n==============================")
        print(f" Running Experiment {i+1}/{len(experiments)}")
        print(f"==============================")
        print(exp)

        new_config = update_config(original_config, exp)
        write_config(new_config)

        run_simulation()
        print((f" Finished Experiment {i+1}/{len(experiments)}, starting next exp in 5 sec"))
        time.sleep(5)

    print("\nAll experiments finished! Restoring config.py")

    write_config(original_config)


if __name__ == "__main__":
    main()
