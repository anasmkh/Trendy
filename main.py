import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st

st.title("Trendy Inventory Forcasting Tool")
product_type = st.sidebar.selectbox('Categories in Trendy', ('Men', 'Women', 'Children'))
if product_type == 'Men':
    category = 0
elif product_type == 'Women':
    category = 1
elif product_type == 'Children':
    category = 2

Product_Subcategory = st.sidebar.selectbox('SubCategories in Trendy', ('T_shirt', 'Jeans', 'Jackets', 'Bags', 'Shoes'))
subcategory = 0
if Product_Subcategory == 'T_shirt' and category == 1:
    subcategory = 1
if Product_Subcategory == 'T_shirt' and category == 2:
    subcategory = 2
if Product_Subcategory == 'T_shirt' and category == 3:
    subcategory = 3

if Product_Subcategory == 'Jeans' and category == 1:
    subcategory = 4
if Product_Subcategory == 'Jeans' and category == 2:
    subcategory = 5
if Product_Subcategory == 'Jeans' and category == 3:
    subcategory = 6

if Product_Subcategory == 'Jackets' and category == 1:
    subcategory = 7
if Product_Subcategory == 'Jackets' and category == 2:
    subcategory = 8
if Product_Subcategory == 'Jackets' and category == 3:
    subcategory = 9

if Product_Subcategory == 'Bags' and category == 1:
    subcategory = 10
if Product_Subcategory == 'Bags' and category == 2:
    subcategory = 11
if Product_Subcategory == 'Bags' and category == 3:
    subcategory = 12

if Product_Subcategory == 'Shoes' and category == 1:
    subcategory = 16
if Product_Subcategory == 'Shoes' and category == 2:
    subcategory = 17
if Product_Subcategory == 'Shoes' and category == 3:
    subcategory = 18

product_code = '0'
if subcategory == 1:
    product_code = st.sidebar.selectbox('Product Code', (
    '1982', '1889', '1982', '1172', '1236', '1472', '1359', '1010', '1581', '1723', '1567', '1189',
    '1674', '1594', '1992'))

if subcategory == 2:
    product_code = st.sidebar.selectbox('Product Code', ('1728',
                                                         '1225',
                                                         '1250',
                                                         '1996',
                                                         '1253',
                                                         '1870',
                                                         '1767',
                                                         '1996',
                                                         '1396',
                                                         '1832',
                                                         '1998',
                                                         '1966',
                                                         '1587',
                                                         '1038',
                                                         '1411',))

if subcategory == 3:
    product_code = st.sidebar.selectbox('Product Code', ('1361', '1761','1644','1923'))


if subcategory == 4:
    product_code = st.sidebar.selectbox('Product Code', ('1777','1428','1197','1526','1108'))

if subcategory == 5:
    product_code = st.sidebar.selectbox('Product Code', ('1614','1345','1270', '1825','1302',
                                                        '1694',
                                                        '1142',
                                                        '1636',
                                                        '1350',
                                                        '1142'))

if subcategory == 6:
    product_code = st.sidebar.selectbox('Product Code', ('1794',
                                                        '1012',
                                                        '1507',
                                                        '1601',
                                                        '1438',
                                                        '1199',
                                                        '1594'
))

if subcategory == 7:
    product_code = st.sidebar.selectbox('Product Code', ('1576','1050'))

if subcategory == 8:
    product_code = st.sidebar.selectbox('Product Code', ('1780',
                                                        '1047',
                                                        '1520',
                                                        '1343'
))

if subcategory == 9:
    product_code = st.sidebar.selectbox('Product Code', ('1320','1828','1028'))

if subcategory == 10:
    product_code = st.sidebar.selectbox('Product Code', ('1218',
                                                            '1768',
                                                            '1525',
                                                            '1780',
                                                            '1977',
                                                            '1686',
                                                            '1267'
))

if subcategory == 11:
    product_code = st.sidebar.selectbox('Product Code', ('1585',
                                                            '1776',
                                                            '1525',
                                                            '1307',
                                                            '1089',
                                                            '1377',
                                                            '1796',
                                                            '1947',
                                                            '1185'
))

if subcategory == 12:
    product_code = st.sidebar.selectbox('Product Code', ('1291','1695','1213','1130'))

if subcategory == 16:
    product_code = st.sidebar.selectbox('Product Code', ('1528','1841','1760','1997','1667','1475'))
if subcategory == 17:
    product_code = st.sidebar.selectbox('Product Code', ('1812',
                                                            '1935',
                                                            '1941',
                                                            '1095'))
if subcategory == 18:
    product_code = st.sidebar.selectbox('Product Code', ('1202','1375','1317','1393','1541','1585'))

Yearly_saled = st.sidebar.number_input('Yearly_saled', 0, 1000000)
Product_cost = st.sidebar.number_input('Product_cost', 0, 1000000)
Holding_cost = st.sidebar.number_input('Holding_cost', 0, 1000000)

dict = {'category_id': [category], 'subcategory_id': [subcategory], 'product_code': [product_code],
        'Yearly_saled': [Yearly_saled],
        'Product_cost': [Product_cost], 'Holding_cost': [Holding_cost]}
dict = pd.DataFrame(dict)


def load_data():
    data = pd.read_csv('inventory.csv')
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data


if st.button('show_data'):
    # st.plotly_chart(load_data())
    st.write(load_data())


def split(df):
    y = df.order
    x = df.drop(columns=['id', 'product_name', 'product_quantity', 'product_size', 'product_color',
                         ' product_price ', 'order'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=50)  # 18 is also good value
    return x_train, x_test, y_train, y_test


df = load_data()
x_train, x_test, y_train, y_test = split(df)

model = LinearRegression(copy_X=False, n_jobs=None, positive=True, fit_intercept=True, normalize=True)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
if st.button('Show Model Accuracy'):
    st.write(accuracy * 100, '%')

y_pred = model.predict(dict)
print(y_pred)

if st.button('Get Prediction'):
    y = np.round(y_pred)
    result = str(y)
    st.write('You Need to Order:', result, 'peace of: ', 'product_code', product_code, ' ', product_type,
             Product_Subcategory)
    # st.write(y)
