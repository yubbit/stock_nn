import pandas as pd
import numpy as np

# Convert the PSE data into a time series
data = pd.read_csv('../data/pse_data.csv', header=0)

pse_data = pd.DataFrame()

ticker_list = data['long_name'].unique()
for ticker in ticker_list:
    temp = data[data['long_name'] == ticker]
    temp = temp.drop_duplicates('date', 'last')
    pse_data = pse_data.append(temp)

pse_data = pse_data.pivot('date', 'long_name')
pse_data[pse_data < 0] = 0

pse_data['open'] = pse_data['open'].fillna(method='pad')
pse_data['high'] = pse_data['high'].fillna(method='pad')
pse_data['low'] = pse_data['low'].fillna(method='pad')
pse_data['close'] = pse_data['close'].fillna(method='pad')

# Generate the output data set. This will flag positive if the stock price goes above a certain
# percentage any time within a 90-day window, without going below a certain percentage
price_max = pse_data['close'].rolling(90).max()
price_max = price_max.shift(-90)
price_max = price_max / pse_data['close'] - 1
price_max = price_max.fillna(0)
price_max = price_max.applymap(lambda x: 0 if x < 0.2 else 1)

price_min = pse_data['close'].rolling(90).min()
price_min = price_min.shift(-90)
price_min = price_min / pse_data['close'] - 1
price_min = price_min.fillna(0)
price_min = price_min.applymap(lambda x: 0 if x < -0.1 else 1)

output_data = price_max * price_min
output_data.to_csv('data/output_data.csv')

# Process the PSE input data
pse_data = pse_data.pct_change()
pse_data = (pse_data - pse_data.mean()) / pse_data.std() # Get the percentage change and normalize the data
pse_data.fillna(0)

pse_data.index = pd.to_datetime(pse_data.index)


# Convert the World Bank data into a time series
wb_data = pd.read_csv('../data/wb_data.csv', skiprows=3)
wb_data = wb_data.drop(['Country Name', 'Country Code', 'Indicator Code', 'Unnamed: 61'], axis=1)
wb_data = wb_data.transpose()

wb_data.columns = wb_data.ix[0]
wb_data = wb_data.drop(wb_data.index[0]) 

wb_data.index = pd.to_datetime(wb_data.index)


# Process the World Bank data
wb_data.index = wb_data.index + pd.DateOffset(years=1) # Add one to each index, to coincide with data availability
wb_data = wb_data['1982-01-01':]
wb_data = wb_data.replace(0, np.nan)

ignore_data = []
for i in range(len(wb_data.columns)):
    if wb_data[wb_data.columns[i]].count() / len(wb_data[wb_data.columns[i]]) < 0.85: # Count non-NaN data
        ignore_data.append(wb_data.columns[i])

for i in ignore_data:
    wb_data = wb_data.drop(i, axis=1)

wb_data = wb_data.fillna(method='pad')

wb_data = wb_data.pct_change()
wb_data = (wb_data - wb_data.mean()) / wb_data.std() # Get the percentage change and normalize the data

wb_data = wb_data['1983-01-01':]
wb_data = wb_data.fillna(0)

wb_data = wb_data.reindex(pse_data.index, method='pad')


# Generate the input data set
input_data = pd.concat([pse_data['close'], pse_data['volume'], wb_data], axis=1)
input_data = input_data.fillna(0)
input_data.to_csv('data/input_data.csv')



