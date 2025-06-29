import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 檔案
df = pd.read_csv("C:\\Users\\Yajin\\dream\\temp\\anya\\train_metrics.csv")

# 繪製折線圖
plt.figure(figsize=(10, 6))

# PSNR 曲線
# plt.plot(df["step"], df["psnr"], label="PSNR", color="blue")

# SSIM 曲線
# plt.plot(df["step"], df["ssim"], label="SSIM", color="green")

# LPIPS 曲線
plt.plot(df["step"], df["lpips"], label="LPIPS", color="red")

# 設定標題與標籤
plt.xlabel("Step")
plt.ylabel("Metric Value")
plt.title("Training Metrics Over Steps(Anya)")
plt.legend()
plt.grid()

# 顯示圖表
plt.show()
