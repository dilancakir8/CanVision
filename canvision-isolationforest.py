import pandas as pd
from sklearn.ensemble import IsolationForest
data = pd.read_csv("CANVision_temiz_veri.csv")

X = data[["Zaman", "CAN_ID", "Sinyal"]]     

iso_model= IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_model.fit(X)
y_pred = iso_model.predict(X)

print("Isolation Forest Sonucu")
print(y_pred)