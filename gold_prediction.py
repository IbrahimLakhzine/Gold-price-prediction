import pandas as pd
from prophet import Prophet


df = pd.read_excel(r'C:/Users/ibrahim.l/Desktop/gold_monthly.xlsx')
#print(df.tail())
df.columns=["ds","y"]

model = Prophet()
model.fit(df)
period=12*30
future = model.make_future_dataframe(periods=period,freq='M')
forecast=model.predict(future)

print(forecast[['ds', 'yhat']].tail())

figure=model.plot(forecast)