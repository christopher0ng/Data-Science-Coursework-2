 # The use of neural networks in heart disease detection
 A simple neural network trained to predict whether a patient has a heart disease

 ## Data:
 Dataset is downloaded from: https://www.kaggle.com/datasets/mexwell/heart-disease-dataset/data
 Dataset folder contains csv file and documentation.pdf that gives more information about each column in the dataset

 ## How to run:
     
To install the required libraries and their versions

    pip install -r requirements.txt
    
To generate exploratory plots that are stored in figures folder

    python3 Exploratory_data_analysis.py

To preprocess data

    python3 Data_preprocessing.py

To perform hyperparameter tuning

    python3 Hyperparameter_tuning.py
 
To train the model 

    python3 Model_training.py

To evaluate our model 

    python3 Model_evaluation.py

Trained model weights are saved in files nn_model.json and trained_nn_model.weights.h5

## Results
Overview of methodology and analysis of results are found in Overview.pdf

