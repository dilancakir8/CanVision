import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

results = pd.read_csv("autoencoder_results.csv")

plt.figure(figsize=(6,4))
plt.hist(results["ReconstructionError"], bins=50, alpha=0.7)
plt.xlabel("Rekonstrüksiyon Hatası (Autoencoder)")
plt.ylabel("Frekans")
plt.title("Autoencoder Hata Dağılımı")
plt.show()

summary = results["Anomaly"].value_counts().rename({0:"Normal", 1:"Anomali"}).reset_index()
summary.columns = ["Durum", "Sayı"]

plt.figure(figsize=(6,4))
sns.barplot(x="Durum", y="Sayı", data=summary)
plt.title("Autoencoder Normal vs Anomali Dağılımı")
plt.show()

norm_error = (results["ReconstructionError"] - results["ReconstructionError"].min()) / \
             (results["ReconstructionError"].max() - results["ReconstructionError"].min())

results["HealthScore"] = 1 - norm_error

plt.figure(figsize=(10,4))
plt.plot(results["Zaman"], results["HealthScore"], label="Health Score")
plt.xlabel("Zaman")
plt.ylabel("Health Score")
plt.title("Autoencoder Health Score Zaman Grafiği")
plt.legend()
plt.show()

results["HealthScoreTrend"] = results["HealthScore"].rolling(window=10).mean()

plt.figure(figsize=(10,4))
plt.plot(results["Zaman"], results["HealthScore"], alpha=0.6, label="Raw Health Score")
plt.plot(results["Zaman"], results["HealthScoreTrend"], color="red", label="Trend (Moving Average)")
plt.xlabel("Zaman")
plt.ylabel("Health Score")
plt.title("Autoencoder Health Score Trend Analizi")
plt.legend()
plt.show()

threshold = 0.3 
plt.figure(figsize=(10,4))
plt.plot(results["Zaman"], results["HealthScore"], label="Health Score")
plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
plt.xlabel("Zaman")
plt.ylabel("Health Score")
plt.title("Autoencoder Erken Uyarı Mekanizması")
plt.legend()
plt.show()

spike_count = np.sum(results["ReconstructionError"] > (results["ReconstructionError"].mean() + 2*results["ReconstructionError"].std()))
print("Autoencoder kritik spike sayısı:", spike_count)