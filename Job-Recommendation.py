import requests
import pandas as pd

url = 'https://api.coingecko.com/api/v3/coins/markets'
params = {
    "ids:" : 'bitcoin',
    "vs_currency": 'usd'
}
response = requests.get(url, params=params)
data = response.json()
df = pd.DataFrame(data)
print(df)