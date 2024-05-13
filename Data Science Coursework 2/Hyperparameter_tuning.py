from Data_preprocessing import X_train, y_train
import keras
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_model(neurons1, neurons2, neurons3, dropout_rate, batch_size=32, epochs=10):
    """
    function to create our neural network given the prescribed hyperparameters
    """
    keras.utils.set_random_seed(81)
    model = Sequential()
    model.add(Dense(neurons1,  activation='relu', input_shape=(16,)))  
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons3, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create a KerasClassifier
model = KerasClassifier(model=create_model,dropout_rate=0.1,neurons1=64,neurons2=32,neurons3=16)

# Define the parameter grid. These are the hyperparameters we will tune
param_grid = {
    'neurons1': [40, 80, 160],  
    'neurons2': [20, 40, 80],   
    'neurons3': [10, 20, 40],  
    'dropout_rate': [0.1, 0.2],  
    'batch_size': [32, 64],  
    'epochs': [20, 30]  
}

# Create the GridSearchCV object
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=KFold(n_splits=3))

# Perform the grid search
grid_result = grid.fit(X_train, y_train)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

