# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:36:13 2024

@author: GK986HL
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
 
df = pd.read_csv(r'C:\Users\GK986HL\Downloads\week5 day2 class content\Iris.csv')
df.head()
df.info()
 
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
 
classifier = RandomForestClassifier()
 
# Fit the model to the data
classifier.fit(X, y)
 
# Create the Streamlit app
st.title("Iris Species Classifier")
 
# Sidebar sliders for input features
sepal_length = st.sidebar.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))
 
# Predict the species
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = classifier.predict(input_data)
 
# Display the predicted species
st.write(f"Predicted Species: {prediction[0]}")