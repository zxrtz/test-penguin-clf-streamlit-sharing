import pandas as pd
penguins = pd.read_csv("penguins_cleaned.csv")

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy()
target = 'sex'
encode = ['sex', 'island']

# encode all features, concatenate, and delete unencoded features
for col in encode :
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

# Separating X and yh
X = df.drop(columns='species')
Y = df['species']

# Build Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X,Y)

# Save RF model
import pickle
pickle.dump(rfc, open('penguins_clf.pkl', 'wb')) # creates file, and writes the model to it

