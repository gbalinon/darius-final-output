import numpy as np
import pandas as pd
import os
#from sklearn.externals
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache
def get_input():
    data = pd.read_csv("diamonds_regression.csv")
    return data

df = get_input()

### Set Sidebar Options
st.sidebar.title('About')
st.sidebar.info('Change parameters to see how insurance prices change.')
st.sidebar.title('Parameters')
# Carat
carat = st.sidebar.slider('Carat', 0.0, 2.0, 0.2)
# Depth
depth = st.sidebar.slider('Depth', 0.0, 100.0, 61.0)


### Price Output
st.subheader('Output Diamond Price')
filename = 'diamond_model.sav'
loaded_model = joblib.load(filename)

# Predict diamond price by carat and depth
prediction = round(loaded_model.predict([[carat, depth]])[0])

st.write(f"Suggested Diamond Price is: {prediction}")

chart_data = df[['carat', 'price']]

#@st.cache
def plot_line_sns():
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(x = "carat", y = "price", data = chart_data)
    st.pyplot(fig)

#@st.cache
def plot_pair_sns():
    fig2 = sns.pairplot(df[['carat', 'price']])
    st.pyplot(fig2)
    
def plot_count_sns():
    fig3 = plt.figure(figsize=(10, 4))
    sns.countplot(x = "cut", data = df)
    st.pyplot(fig3)

st.title = "Diamond Pricing App"
st.write("""From the data below, we built a machine learning—based pricing model to get price of a diamond with given carat and depth. """)


st.sidebar.title('Options')
if st.sidebar.checkbox ("Show Data (20 rows)"):
    st.write(df.head(20))
    
if st.sidebar.checkbox ("Show line graph"):
    plot_line_sns()
    
if st.sidebar.checkbox ("Show pair plot"):
    plot_pair_sns()

if st.sidebar.checkbox ("Show bar graph"):
    plot_count_sns()
    
