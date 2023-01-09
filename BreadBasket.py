import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df=pd.read_csv('BreadBasket_DMS.csv')
print(df.head())

print("unique items: ", df['Item'].nunique())
print(df['Item'].unique())

###### item==coffee or item==bread
print(df[(df['Item']=='Coffee') | (df['Item']=='Bread')])

####### check null
print(df.isnull().sum())



######## check item==none
print(df[(df['Item']=='NONE')])



######### delete item==none
df.drop(df[df['Item']=='NONE'].index, inplace=True)



######### split Date to day, month, year
df['day']=df['Date'].apply(lambda x: x.split('-')[2])
df['month']=df['Date'].apply(lambda x: x.split('-')[1])
df['year']=df['Date'].apply(lambda x: x.split('-')[0])

print(df.head())



######### show items most sold
most_sold=df['Item'].value_counts().head(10)
print(most_sold)



### Chart
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
color = plt.cm.rainbow(np.linspace(0, 1, 10))
most_sold.plot(kind='bar',color = color)
plt.xticks(rotation = 90 )
plt.title('item most sold')

plt.subplot(1,2,2)
most_sold.plot(kind='pie')
plt.xticks(rotation = 90 )
plt.show()



# months=['10','11','12','1','2','3','4']
months=df['month'].unique()
Transactions = df.groupby('month')['Transaction'].nunique()
color = plt.cm.rainbow(np.linspace(0, 1, 10))
plt.bar(x=months, height=Transactions)
plt.title('month sale')
plt.xlabel("10/2016-4/2017")
plt.ylabel('Transaction')
plt.show()

###### Asociation_rules and apriori algorithm
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori

list_transaction = []

for i in df['Transaction'].unique():
    tlist=list(set(df[df['Transaction']==i]['Item']))
    if(len(tlist)>0):
        list_transaction.append(tlist)
print(len(list_transaction))


print(list_transaction)



te = TransactionEncoder()
te_ary=te.fit(list_transaction).transform(list_transaction)
df2=pd.DataFrame(te_ary,columns=te.columns_)

# print(df2.sample(3))


frequent_itemsets = apriori(df2,min_support=0.01,use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift',min_threshold=0.1)
frequent_itemsets['length']=frequent_itemsets['itemsets'].apply(lambda x: len(x))   

rules.sort_values('confidence',ascending=False)
print(rules)