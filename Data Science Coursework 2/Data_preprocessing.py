import pandas as pd
from Exploratory_data_analysis import categorical_features, df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Convert categorical features into one hot encoding
for i in range(len(categorical_features)):
    dummies = pd.get_dummies(df[categorical_features[i]],drop_first=True)
    df = df.drop(categorical_features[i],axis=1)
    df = pd.concat([df,dummies],axis=1)

# Separate dataframe into features and labels
X = df.drop('target',axis=1).values
y = df['target'].values

# Split our dataset into training and test samples in a ratio of 80 to 20. Seed is set for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=81)

# Scale our data to be in range [0,1]. Only fit the scaler to the train data and not to the test data to prevent data leakage
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

