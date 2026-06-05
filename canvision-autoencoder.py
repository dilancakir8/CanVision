import pandas as pd
import numpy as np
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


data = pd.read_csv("CANVision_temiz_veri.csv")

X = data.values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

input_dim = X_scaled.shape[1]  
encoding_dim = 8

input_layer = Input(shape=(input_dim,))
encoder = Dense(16, activation="relu")(input_layer)
encoder = Dense(encoding_dim, activation="relu")(encoder)

decoder = Dense(16, activation="relu")(encoder)
decoder = Dense(input_dim, activation="sigmoid")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.2
)

X_pred = autoencoder.predict(X_scaled)

mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

threshold = np.mean(mse) + 2*np.std(mse)

anomalies = mse > threshold
print("Toplam anomali sayısı:", np.sum(anomalies))
import pandas as pd

results = pd.DataFrame({
    "Zaman": data["Zaman"],
    "CAN_ID": data["CAN_ID"],
    "Sinyal": data["Sinyal"],
    "ReconstructionError": mse,
    "Anomaly": anomalies.astype(int)
})


results.to_csv("autoencoder_results.csv", index=False)
print("Sonuçlar 'autoencoder_results.csv' dosyasına kaydedildi.")
