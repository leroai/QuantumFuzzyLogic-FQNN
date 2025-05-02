# Bitcoin Veri Ön İşleme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Veri Toplama (gerçek uygulamada API'lerden alınabilir)
# Son 365 gün için yapay Bitcoin fiyat verisi oluşturma
np.random.seed(42)  # Tekrarlanabilirlik için
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
dates = pd.date_range(start=start_date, end=end_date, freq='D')
n = len(dates)

# Sentetik fiyat verisi oluşturma
start_price = 30000
trend = np.linspace(0, 5000, n)
seasonality = 1000 * np.sin(np.arange(n) * (2 * np.pi / 7))
noise = np.random.normal(0, 2000, n)
prices = start_price + trend + seasonality + noise
prices = np.maximum(prices, 100)

# OHLCV veri çerçevesi oluşturma
df = pd.DataFrame({
    'Open': prices * np.random.uniform(0.98, 1.0, n),
    'High': prices * np.random.uniform(1.0, 1.05, n),
    'Low': prices * np.random.uniform(0.95, 0.99, n),
    'Close': prices,
    'Volume': 10000000 + 5000000 * np.random.randn(n) + 100000 * prices / start_price
}, index=dates)

# 2. Eksik Değerleri İşleme
# Örnek olarak bazı eksik değerler ekleyelim
df.loc[df.index[10:15], 'Close'] = np.nan
df.loc[df.index[30:32], 'Volume'] = np.nan

# Eksik değerleri doldurma - ileri + geri doldurma kombinasyonu
df_filled = df.fillna(method='ffill').fillna(method='bfill')
df = df_filled.copy()

# 3. Veri Normalizasyonu
scaler_minmax = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler_minmax.fit_transform(df),
    columns=df.columns,
    index=df.index
)

print("Normalize edilmiş veri:")
print(df_normalized.head())
