import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import numpy as np

data = pd.read_csv("CANVision_temiz_veri.csv")
data = data.head(1000)

X = data[["Zaman", "CAN_ID", "Sinyal"]]

iso_model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_model.fit(X)
y_pred = iso_model.predict(X)
data["prediction"] = y_pred
normal = data[data["prediction"] == 1]
anomaly = data[data["prediction"] == -1]

data["AnomalyScore"] = -iso_model.score_samples(X)


summary = data["prediction"].value_counts()
summary_table = pd.DataFrame({
    "Durum": ["Normal", "Anomali"],
    "Sayı": [summary.get(1, 0), summary.get(-1, 0)]
})

plt.figure(figsize=(6,4))
sns.barplot(x="Durum", y="Sayı", data=summary_table, palette=["blue","red"])
plt.title("Isolation Forest: Normal vs Anomali")
plt.show()

CAN_ID_summary = data[data["prediction"] == -1]["CAN_ID"].value_counts().reset_index()
CAN_ID_summary.columns = ["CAN_ID", "Anomali Sayısı"]

plt.figure(figsize=(10,6))
sns.barplot(x="CAN_ID", y="Anomali Sayısı", data=CAN_ID_summary, color="red")
plt.title("Isolation Forest: CAN_ID Bazlı Anomali Dağılımı")
plt.xticks(rotation=90)
plt.show()

norm_score = (data["AnomalyScore"] - data["AnomalyScore"].min()) / \
             (data["AnomalyScore"].max() - data["AnomalyScore"].min())
data["HealthScore"] = 1 - norm_score

plt.figure(figsize=(10,4))
plt.plot(data["Zaman"], data["HealthScore"], label="Health Score")
plt.xlabel("Zaman")
plt.ylabel("Health Score")
plt.title("Isolation Forest Health Score Zaman Grafiği")
plt.legend()
plt.show()

data["HealthScoreTrend"] = data["HealthScore"].rolling(window=10).mean()

plt.figure(figsize=(10,4))
plt.plot(data["Zaman"], data["HealthScore"], alpha=0.6, label="Raw Health Score")
plt.plot(data["Zaman"], data["HealthScoreTrend"], color="red", label="Trend (Moving Average)")
plt.xlabel("Zaman")
plt.ylabel("Health Score")
plt.title("Isolation Forest Health Score Trend Analizi")
plt.legend()
plt.show()

threshold = 0.3
plt.figure(figsize=(10,4))
plt.plot(data["Zaman"], data["HealthScore"], label="Health Score")
plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
plt.xlabel("Zaman")
plt.ylabel("Health Score")
plt.title("Isolation Forest Erken Uyarı Mekanizması")
plt.legend()
plt.show()

spike_threshold = data["AnomalyScore"].mean() + 2*data["AnomalyScore"].std()
spike_count = np.sum(data["AnomalyScore"] > spike_threshold)
print("Isolation Forest kritik spike sayısı:", spike_count)