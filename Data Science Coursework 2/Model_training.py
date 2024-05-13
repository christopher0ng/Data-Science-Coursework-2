from Data_preprocessing import X_train, y_train
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

keras.utils.set_random_seed(81)
nn_model =Sequential()
nn_model.add(Dense(80,  activation='relu', input_shape=(16,)))  
nn_model.add(Dropout(0.1))
nn_model.add(Dense(80, activation='relu'))
nn_model.add(Dropout(0.1))
nn_model.add(Dense(20, activation='relu'))
nn_model.add(Dropout(0.1))
nn_model.add(Dense(units=1, activation='sigmoid'))
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(x=X_train, y=y_train, epochs=30, batch_size=32)

# Save our trained model and its weights

# Serialize model to JSON
nn_model_json = nn_model.to_json()
with open("nn_model.json", "w") as json_file:
    json_file.write(nn_model_json)
nn_model.save_weights("trained_nn_model.weights.h5")
print("Model saved")