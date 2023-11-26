# Gerekli kütüphaneleri ekleyelim
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Verileri oluşturduğumuz yer
data = {
    'Yıl': [2018, 2017, 2016, 2015, 2014],
    'Gelir': [70, 50, 60, 40, 30],
    'Yatırım': [8, 6, 7, 5, 3]
}

df = pd.DataFrame(data)

# a) Doğrusal regresyon modeli oluşturma
X = df[['Yatırım']]
y = df['Gelir']

# Verileri eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modeli oluşturma
model = LinearRegression()
model.fit(X_train, y_train)

# b) Modelin açıklama gücünü değerlendirme
y_pred = model.predict(X_test)
r2_score = metrics.r2_score(y_test, y_pred)

print(f"R-kare (R^2) değeri: {r2_score}")

# c) İlişkiyi anakütleye genellemek
y_pred_genel = model.predict(X)

# d) Yatırımların artışının gelir üzerindeki etkisi
coef = model.coef_[0]
print(f"Yatırım artışında gelirdeki değişim: {coef} birim")

# e) Yatırım yapmak karlı bir karar mı?

if coef > 0:
    print("Yatırım yapmak karlı bir karar gibi görünüyor.")
else:
    print("Yatırım yapmak karlı bir karar gibi görünmüyor.")

# f) Polinom regresyon (ikinci derece)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

X_poly_pred = poly.fit_transform(X)

y_poly_pred = poly_model.predict(X_poly_pred)


coef_poly = poly_model.coef_

print(f"Polinom regresyon katsayıları: {coef_poly}")

# Grafiği çizelim
plt.scatter(X, y, color='blue', label='Veri Noktaları')
plt.plot(X, y_poly_pred, color='red', label='Polinom Regresyon (Derece 2)')
plt.xlabel('Yatırım')
plt.ylabel('Gelir')
plt.legend()
plt.show()
