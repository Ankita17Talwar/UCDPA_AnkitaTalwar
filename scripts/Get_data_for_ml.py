import time
import pandas as pd
import yfinance as yf
import os

#################################################################################
# Load stock Symbols : we use equity.csv (list of active stocks listed on BSE)
################################################################################
stock_data = pd.read_csv('../dataset/Equity.csv')

print(stock_data.head())
print(stock_data.shape)  # (4230, 10)
print(stock_data.columns)
# Index(['Security Code', 'Issuer Name', 'Security Id', 'Security Name',
#        'Status', 'Group', 'Face Value', 'ISIN No', 'Industry', 'Instrument'],
#       dtype='object')

stock_symbl = pd.DataFrame()
stock_symbl['Symbol'] = stock_data['Issuer Name'] + '.bo'
print(stock_symbl.shape)  # (4230, 1)

# define the size of the chunk
n = 500

# break the data frame into chunks : Below will give the iterator over the dataframe
symbols_chunk = [stock_symbl[i:i + n] for i in range(0, stock_symbl.shape[0], n)]

print(list(symbols_chunk[0]['Symbol']))

#################################################################################
# Get Stock Data from yahoo finance
################################################################################
# stock_info = pd.DataFrame()
i = 0

for chunk in symbols_chunk:

    stock_info = pd.DataFrame()
    for ticker in list(chunk['Symbol']):

        print(ticker)
        ticker_name = yf.Ticker(ticker)
        try:
            ticker_info = ticker_name.info
            # print(ticker_info)
            # parse the dict response into a DataFrame
            ticker_df = pd.DataFrame.from_dict(ticker_info.items()).T
            # sets column names to first row of Datframe
            ticker_df.columns = ticker_df.iloc[0]
            # drop first line - containing column names
            ticker_df = ticker_df.drop(ticker_df.index[0])
            stock_info = stock_info.append(ticker_df)
            time.sleep(3)
            print(f'{ticker} + Complete')
        except (IndexError, ValueError) as e:
            print(f'{ticker} + Data Not Found')
        except TypeError as ex:
            print(str(ex))

    # print(stock_info.shape)
    if not os.path.isfile('../output_files/BES_Data.csv'):
        stock_info.to_csv('../output_files/BES_Data.csv', header=True)
        print('write' + str(i))
    else:
        stock_info.to_csv('../output_files/BES_Data.csv', mode='a', header=False)
        print('append' + str(i))

    i += 1
