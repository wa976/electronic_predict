import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from pandas import Series

fuel_price = pd.read_csv('HOME_전력거래_정산단가_연료원별.csv', encoding='euc-kr').to_numpy()
elec_price = pd.read_csv('HOME_전력거래_계통한계가격_가중평균SMP.csv', encoding='euc-kr').to_numpy()



fuel_price_1 = fuel_price[5:117,2:3]
fuel_price_2 = fuel_price[5:117,3:4]
fuel_price_3 = fuel_price[5:117,4:5]
fuel_price_4 = fuel_price[5:117,5:6]
fuel_price_5 = fuel_price[5:117,6:7]

index = elec_price[1:113,0:1]


# 데이터 시각화
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.plot(fuel_price_1, color='green', marker='o', linestyle='solid')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.plot(fuel_price_2, color='green', marker='o', linestyle='solid')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.plot(fuel_price_3, color='green', marker='o', linestyle='solid')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.plot(fuel_price_4, color='green', marker='o', linestyle='solid')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.plot(fuel_price_5, color='green', marker='o', linestyle='solid')

plt.show()


# 조건 충족
fuel_price = fuel_price[5:117,2:]
elec_price = elec_price[1:113,1:]

print(len(fuel_price))
print(len(elec_price))

train_x, test_x, train_y, test_y = train_test_split(fuel_price, elec_price, random_state=42)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

#정규화
ss = StandardScaler()
ss.fit(train_x)
train_poly = ss.transform(train_x)
test_poly = ss.transform(test_x)

# feature 증가
poly = PolynomialFeatures(degree=4, include_bias=False)
poly.fit(train_x)
train_poly = poly.transform(train_x)
test_poly = poly.transform(test_x)
print(train_poly.shape)
print(test_poly.shape)



#최적의 alpha 찾기
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(train_poly, train_y)
    train_score.append(lasso.score(train_poly, train_y))
    test_score.append(lasso.score(test_poly, test_y))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()


lasso = Lasso(alpha=0.001,max_iter=10000)
lasso.fit(train_poly, train_y)
print("Lasso train :", lasso.score(train_poly, train_y))
print("Lasso test  :", lasso.score(test_poly, test_y))
print("the number of the used features: {}".format(np.sum(lasso.coef_ != 0))) # 사용된 특성갯수

#다음달 값 예측
fuel = [[39.373545,128.8298592,141.1443383,307.5227262,161.9373965]]
my_predict = lasso.predict(poly.transform(fuel))
print("Lasso predict :",my_predict)


#최적의 alpha 찾기
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_poly, train_y)
    train_score.append(ridge.score(train_poly, train_y))
    test_score.append(ridge.score(test_poly, test_y))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()


ridge = Ridge(alpha=10)
ridge.fit(train_poly, train_y)
print("train :", ridge.score(train_poly, train_y))
print("test  :", ridge.score(test_poly, test_y))

#다음달 값 예측
fuel = [[39.373545,128.8298592,141.1443383,307.5227262,161.9373965]]
my_predict = ridge.predict(poly.transform(fuel))
print(my_predict)


#릿지 알파 구하기


#라쏘 알파구하기





"""
"""






