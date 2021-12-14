import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
from functools import reduce

URL='https://www.screener.in/screens/29729/top-1000-stocks/?limit=15&page=1'


page = requests.get(URL)

soup = BeautifulSoup(page.content, 'lxml')

# inspect the location of the table first. We can see that table is located
# under <table> tag and class 'data-table text-nowrap striped'

#  Obtain information from table tag
table1 = soup.find('table', class_='data-table text-nowrap striped')
# print(table1)

list_data =[]

for page in range (1,62):
    URL = 'https://www.screener.in/screens/29729/top-1000-stocks/?limit=15&page='+str(page)

    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'lxml')

    table1 = soup.find('table', class_='data-table text-nowrap striped')

    if page ==1:
        # Create column List
        # Inspect Location of each column. Each column name is present under th
        # Obtain every title of columns with tag <th>
        headers = []
        for i in table1.find_all('th'):
            title = i.text
            print(title)
            title = re.sub('^\n\s*', '', title)  # Data contain text enclosed in \n followed by space
            title = re.sub('\n.*', '', title)    # Remove Trailing \n and any symbol after that
            headers.append(title)

            print(len(headers))  # 24 its giving duplicates as page is displaying column list twice

            unique_list = reduce(lambda l, x: l.append(x) or l if x not in l else l, headers, [])
            print(len(unique_list))
            print(unique_list)

    # now we can fill it with items in each column.
    # Create a for loop, but first we need to identify the location of the row and item column first.
    # row is located under tag <tr> and items are located under tag <td> .
    # This is applied to all rows and items within the table.
    for j in table1.find_all('tr')[1:]:
        try:
            row_data = j.find_all('td')
            row = [(re.sub('^\n\n\s*', '', re.sub('\n\s*\n', '', i.text))).strip() for i in row_data]
            # length = len(data)
            print(row)
            list_data.append(row)
        except AttributeError as e:
            print('Error:' + str(e))
            pass





    #print(length)
    #print(len(row)) 12
    #data = data.append(row)
    #data.loc[length] = row

data = pd.DataFrame(list_data, columns= unique_list)
data.drop(['S.No.'], axis=1, inplace=True)
print(data.shape)  # (1569, 11)

# drop rows with Nulls
data.dropna(how='any',inplace=True)
print (data.shape)  # (1506, 11)
data.to_csv('../output_files/stock_metrics_scanner.csv')

