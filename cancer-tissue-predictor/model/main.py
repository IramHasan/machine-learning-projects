import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle

def get_clean_data():
    data = pd.read_csv("cancer-tissue-predictor/data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1) # data cleaning - remove id and unnamed:32 col from dataset
    data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1}) # data cleaning - change 'B' and 'M' to 0 and 1 for logistic regression
    return data

def create_model(data):
    # split data into predictors and target
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data into train and test sets with test size of 20%
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.2, random_state=42
        ) 
    
    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # test the model 
    y_pred = model.predict(X_test) # make model predictions using the test set
    print('Accuracy of model: ', accuracy_score(y_test, y_pred)) # evaluate the accuracy of the model with the test set
    print("Classification report: \n", classification_report(y_test, y_pred)) 

    return model, scaler


def main():
    
    data = get_clean_data()

    model, scaler = create_model(data)

    with open('cancer-tissue-predictor/model/model.pkl', 'wb') as f: # export purposes : write a binary file for the model using pickle.
        pickle.dump(model, f)

    with open('cancer-tissue-predictor/model/scaler.pkl', 'wb') as f: # export purposes : write a binary file for the scaler using pickle.
        pickle.dump(scaler, f)
    
if __name__ == '__main__':
    main()