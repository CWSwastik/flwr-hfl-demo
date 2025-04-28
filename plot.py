import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
log_path = "./logs/central/central_server.log"
df = pd.read_csv(log_path)

# Plot settings
plt.figure(figsize=(12, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(df["round"], df["loss"], marker="o", color="red")
plt.title("Loss Over Rounds")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(df["round"], df["accuracy"], marker="o", color="green")
plt.title("Accuracy Over Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True)

# Layout and show
plt.tight_layout()
plt.show()
