import pandas as pd
import missingno as mno
import matplotlib.pyplot as plt

data = pd.read_csv('../dataset/titanic_dataset.csv')

#print(data.head())

print(data.info())

# Visualize Missing Data
mno.matrix(data)
plt.show()

print(data.isnull().sum())

updated_data = data
updated_data['Age'] = updated_data['Age'].fillna(updated_data['Age'].mean())
updated_data.info()


# print(updated_data.iloc[[0]])

print(updated_data[updated_data.duplicated()])

df = pd.DataFrame([['ABC', 1],
                   ['ABC', 1],
                   ['ABC', 2],
                   ['C', '3'],
                   ['D', '3']
                   ], columns=['NAME','ID'])
print(df[df.duplicated()])
print(df[df.duplicated(keep="first")])
print(df[df.duplicated(keep="last")])
print(df[df.duplicated(keep="first", subset=['ID'])])
print(df[df.duplicated(keep="last", subset=['ID'])])

df.drop_duplicates(subset=['ID'], keep='first', inplace=True)
print(df)