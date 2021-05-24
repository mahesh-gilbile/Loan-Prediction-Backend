# -*- coding: utf-8 -*-

from flask import Flask,request,jsonify,request
import json
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score


app=Flask(__name__)
from flask_cors import CORS
CORS(app)

model1 = pickle.load(open('LSlib', 'rb'))
model2 = pickle.load(open('SVMlib', 'rb'))

@app.route('/')
def hey():
    return jsonify("It's working")

@app.route('/prediction',methods=['POST'])
def mahesh():
    data = request.data
    data = json.loads(data)
    applicantFirstName = data.get('applicantFirstName')
    applicantLastName = data.get('applicantLastName')
    applicantMobileNumber = data.get('applicantMobileNumber')
    applicantIncome = data.get('applicantIncome')
    coApplicantFirstName = data.get('coApplicantFirstName')
    coApplicantLastName = data.get('coApplicantLastName')
    coApplicantMobileNumber = data.get('coApplicantMobileNumber')
    coApplicantIncome = data.get('coApplicantIncome')
    totalIncome = data.get('totalIncome')
    loanAmount = data.get('loanAmount')
    loanAmountTerm = data.get('loanAmountTerm')
    creditScore = data.get('creditScore')

    creditScore = float(creditScore)
    if creditScore >= 0.5:
        creditScore = 1
    else:
        creditScore = 0

    loanAmount = int(loanAmount)
    loanAmountTerm = int(loanAmountTerm)

    ApplicantIncomelog = np.log(applicantIncome)
    LoanAmountlog = np.log(loanAmount)
    Loan_Ammount_Term_log = np.log(loanAmountTerm)
    Total_Income_log = np.log(totalIncome)


    #Logistic HardCode

    #read csv file of loan prediction
    df = pd.read_csv("Loan Prediction Dataset.csv")

    #numberic data cleaning
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

    #characterise data cleaning
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome'])
    df['LoanAmountLog'] = np.log(df['LoanAmount'])
    df['Loan_Amount_Term_Log'] = np.log(df['Loan_Amount_Term'])
    df['TotalIncomeLog'] = np.log(df['TotalIncome'])

    #drop columns
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'TotalIncome', 'Loan_ID', 'Gender', 'Married', 'Dependents', 'Self_Employed', 'Property_Area', 'Education']
    df = df.drop(columns=cols, axis=1)

    #Encoding
    from sklearn.preprocessing import LabelEncoder
    cols = ['Loan_Status']
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col])

    #Specifing Input And Out Attributes
    x = df.drop(columns = ['Loan_Status'], axis = 1)
    y = df['Loan_Status']

    #Spliting the data for  testing and training purpose
    from sklearn.model_selection import train_test_split
    x_train, x_test , y_train , y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

    
    #Customize logistic regression code
    class LogReg(): 
        """
        Custom made Logistic Regression class
        """
        def __init__(self, lr=0.01, n_iter= 1000): 
            self.lr = lr
            self.n_iter = n_iter 
            self.params = {}
        
        def param_init(self, X_train): 
            self.X_train = X_train
            """
            Initialize parameters 
            __________________ 
            Input(s)
            X: Training data
            """
            _, n_features = self.X_train.shape # shape of training data

            # initializing coefficents to 0 
            self.params["W"] = np.zeros(n_features)
            self.params["b"] = 0
            return self.params

        def get_z(self, x, W, b): 
            """
            Calculates Linear Function
            __________________
            Input(s)
            X: Training data
            W: Weight coefficients
            b: bias coefficients
            __________________
            Output(s)
            z: a Linear function
            """
            z = np.dot(x, W) + b
            return z
            
        def sigmoid(self , z):
            """
            Logit model
            _________________
            Input(s)
            z: Linear model 
            _________________
            Output(s)
            g: Logit function applied to linear model
            """
            g = 1 / (1 + np.exp(-z))
            return g 
            

        def gradient_descent(self, x_train, y_train, params, lr, n_iter): 
            self.x_train = x_train
            self.y_train = y_train
            self.params = params
            self.lr = lr
            self.n_iter = n_iter
            
            """
            Gradient descent to minimize cost function
            __________________ 
            Input(s)
            X: Training data
            y: Labels
            params: Dictionary contatining random coefficients
            alpha: Model learning rate
            __________________
            Output(s)
            params: Dictionary containing optimized coefficients
            """

            W = self.params["W"] 
            b = self.params["b"] 
            m = x_train.shape[0]

            for _ in range(self.n_iter): 
                # prediction with random weights
                g = self.sigmoid(self.get_z(x, W, b))
                # calculate the loss
                loss = -1/m * np.sum(y * np.log(g)) + (1 - y) * np.log(1 - g)
                # partial derivative of weights 
                dW = 1/m * np.dot(x.T, (g - y))
                db = 1/m * np.sum(g - y)
                # updates to coefficients
                W -= self.lr * dW
                b -= self.lr * db 
            
            self.params["W"] = W
            self.params["b"] = b
            return self

        def train(self, x_train, y_train):
            """
            Train model with Gradient decent
            __________________ 
            Input(s)
            X: Training data
            y: Labels
            alpha: Model learning rate
            n_iter: Number of iterations 
            __________________
            Output(s)
            params: Dictionary containing optimized coefficients
            """ 
            self.params = self.param_init(x_train)
            self.gradient_descent(x_train, y_train, self.params , self.lr, self.n_iter)
            return self 

        def predict(self, x_test):
            """
            Inference 
            __________________ 
            Input(s)
            X: Unseen data
            params: Dictionary contianing optimized weights from training
            __________________
            Output(s)
            y_preds: Predictions of model
            """  
            g = self.sigmoid(np.dot(x_test, self.params["W"]) + self.params["b"])
            return g


    #Training Model in customize logistic class
    model = LogReg()
    model.train(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)

    #Getting accuracy from customize LRC
    Accuracy_LRHard = str(accuracy_score(y_test, y_pred)*100)
    #Accuracy of logistic library
    Accuracy_LR = 77.27272727272727
    #Accuracy of Support Vector Machine Library
    Accuraccy_SVM = 77.92207792207793
     
    #Logistic Library Prediction
    logistic_lib = model1.predict([[creditScore, ApplicantIncomelog, LoanAmountlog, Loan_Ammount_Term_log, Total_Income_log]])

    #Support Vector Machince Library Code Prediction
    svm_lib = model2.predict([[creditScore, ApplicantIncomelog, LoanAmountlog, Loan_Ammount_Term_log, Total_Income_log]])

    #Hard Coded Logistic Regression Code Prediction
    logistic_hardcoded = model.predict([[creditScore, ApplicantIncomelog, LoanAmountlog, Loan_Ammount_Term_log, Total_Income_log]])

    logistic_hardcoded = (logistic_hardcoded > 0.5)

    if str(logistic_hardcoded) == '[False]':
        logistic_hardcoded = '0'
    else:
        logistic_hardcoded = '1'
    
    return jsonify(Accuracy_LRL = str(Accuracy_LR) , Accuraccy_SVM = str(Accuraccy_SVM) , Accuracy_LRHard = str(Accuracy_LRHard) , logistic_hardcoded = str(logistic_hardcoded[0]) , logistic_lib = str(logistic_lib[0]) , svm_lib = str(svm_lib[0]))

if __name__=="__main__":
    app.run(debug=True)
    