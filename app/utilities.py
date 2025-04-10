# here i will create a function that takes
# rf_model.pkl and predicts for new observations

import pickle
import pandas as pd

# get trained model
with open('model_store/rf_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# create function for prediction - takes in data and model
def predict(input_data, model):
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    input_df = pd.DataFrame([input_data], columns=feature_names)
    pred = model.predict(input_df)
    return pred

if __name__=="__main__":
    # Text to classify should be in a list.
    data = [0.3, 0.5, 0.1, 0.4]

    class_pred = predict(data,loaded_model)
    print(class_pred)