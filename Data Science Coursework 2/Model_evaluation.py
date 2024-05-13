from Data_preprocessing import X_test, y_test
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import classification_report

# Loading our trained model and weights
json_file = open('nn_model.json', 'r')
trained_model = json_file.read()
json_file.close()
trained_model_from_json = model_from_json(trained_model)
trained_model_from_json.load_weights("trained_nn_model.weights.h5")
print("Model loaded")

# Evaluating our model on the test set
prediction_prob = trained_model_from_json.predict(X_test)
predictions = np.where(prediction_prob > 0.5, 1,0).squeeze()
print(classification_report(y_test,predictions))

