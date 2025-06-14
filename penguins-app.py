import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App
         
This app predicts the **Palmer Penguin** species!

Data obtained from [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.         
""") # [text](link) is the markdown syntax for hyperlinks

st.sidebar.header("User Input Features")

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into a dataframe. Either using CSV or sidebar data
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else :
    def user_input_features() :
        island = st.sidebar.selectbox('Island',('Biscoe', 'Dream','Torgersen')) # label, then categorical values
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9) # label, max, min, default
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'sex' : sex,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,}
        return pd.DataFrame(data, index=[0])
    
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# useful for encoding phase
penguins_raw = pd.read_csv("penguins_cleaned.csv")
penguins = penguins_raw.drop(columns=['species']) # no target data (X input data format)
df = pd.concat([input_df, penguins], axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=0)
    df = df.drop(columns=col)
df = df[:1]

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None: # if theres an uploaded file, display the input dataframe
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in the classifier model through pickle
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# make predictions
prediction = load_clf.predict(df)
prediction_probas = load_clf.predict_proba(df)


# prediction is the label index, so use indexing to choose the correct species
st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction probability')
st.write(prediction_probas)
