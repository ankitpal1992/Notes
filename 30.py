import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
a=pd.read_csv('USA_Housing.csv')
print(a.columns)
print(a.head(5))
print(a.info())
print(a.describe())
sns.pairplot(a)
plt.show()
sns.heatmap(a.corr())
plt.show()
x=a[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y=a['Price']
from sklearn.cross_validation import train_test_split
x_trn,x_test,y_trn,y_test=train_test_split(x,y,test_size=0.4,random_state=101)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_trn,y_trn)
print(lm.intercept_)
print(lm.coef_)
pdt=lm.predict(x_test)
print(pdt)
plt.hist(pdt,bins=70)
plt.show()
print(x_trn.columns)
d=pd.DataFrame(lm.coef_,x.columns,columns=['Coeff'])
print(d)
plt.scatter(y_test,pdt)
plt.show()
sns.distplot(y_test-pdt)
plt.show()
