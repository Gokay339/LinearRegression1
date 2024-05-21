import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from google.colab import files
uploaded = files.upload()

fuel = pd.read_csv('FuelConsumption.csv')
fuel.head()

fuel.isnull().sum()

print(fuel.dtypes)      #DATALARIN TİPİNİ GÖRMEYE YARAR


fuel.isnull().sum()


x = fuel['COEMISSIONS '].values
y = fuel['FUEL CONSUMPTION'].values

print(x.shape)
print(type(x))

uzunluk = len(x)
x = x.reshape((uzunluk,1))
print(x.shape)
print(type(x))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=42)            #RANDOMSTATE KOMUTU SÜREKLİ AYNI VERİLERLE EĞİTİM YAPMASINI SAĞLAR

from sklearn.linear_model import LinearRegression
modelregresyon = LinearRegression()
modelregresyon.fit(x_train,y_train)               #FİT UYDUR DEMEK

y_pred = modelregresyon.predict(x_test)

plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,modelregresyon.predict(x_train),color="blue")
plt.title("Eğitim Verileri")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

plt.scatter(x_test,y_test,color="green")
plt.plot(x_test,modelregresyon.predict(x_test),color="blue")
plt.title("Test Verileri")
plt.xlabel('Başın Çevre Uzunluğu (cm^3)')
plt.ylabel('Beyin Ağırlığı(gram)')
plt.show()

print('Eğim(Q1):', modelregresyon.coef_)                                       # Bu, lineer regresyon modelinin eğim katsayısını temsil eder.
print('Kesen(Q0):', modelregresyon.intercept_)                                 #Bu, lineer regresyon modelinin kesim noktası katsayısını temsil eder
print("y=%0.2f"%modelregresyon.coef_+"x+%0.2f"%modelregresyon.intercept_)    # Y = BX+C GİBİ

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score

print("R-Kare: ", r2_score(y_test, y_pred))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))           # DOĞRULUK ORANLARINI ÖLÇMEYE YARAR
print("MedAE: ", median_absolute_error(y_test, y_pred))
print("EVS: ", explained_variance_score(y_test, y_pred))

Doğruluk oranı : https://prnt.sc/aJ572xj8kK7G
