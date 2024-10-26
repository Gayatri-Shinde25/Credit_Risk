import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def ModelPreProcessing(df):
    # Missing Values Treatment
    df.fillna({'loan_int_rate': df['loan_int_rate'].median()}, inplace=True)
    df.fillna({'person_emp_length': df['person_emp_length'].median()}, inplace=True)
    
    # Outliers Treatment
    df = df[df['person_age'] <= 75]
    df = df[df['person_emp_length'] <= 47]
    
    # Categorical Variables Treatment
    grade_mapping = {
        'A': 0,
        'B': 0,
        'C': 0,
        'D': 1,
        'E': 1,
        'F': 1,
        'G': 1
    }
    df['loan_grade'] = df['loan_grade'].map(grade_mapping)

    # Manually encode person_home_ownership
    df['OWN'] = (df['person_home_ownership'] == 'OWN').astype(int)
    df['RENT'] = (df['person_home_ownership'] == 'RENT').astype(int)
    #df['person_home_ownership_MORTGAGE'] = (df['person_home_ownership'] == 'MORTGAGE').astype(int)
    df['OTHER'] = (df['person_home_ownership'] == 'OTHER').astype(int)
    
    # Manually encode loan_intent
    #df['loan_intent_DEBTCONSOLIDATION'] = (df['loan_intent'] == 'DEBTCONSOLIDATION').astype(int)
    df['EDUCATION'] = (df['loan_intent'] == 'EDUCATION').astype(int)
    df['HOMEIMPROVEMENT'] = (df['loan_intent'] == 'HOMEIMPROVEMENT').astype(int)
    df['MEDICAL'] = (df['loan_intent'] == 'MEDICAL').astype(int)
    df['PERSONAL'] = (df['loan_intent'] == 'PERSONAL').astype(int)
    df['VENTURE'] = (df['loan_intent'] == 'VENTURE').astype(int)

    # Manually encode cb_person_default_on_file
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

    # Drop original categorical columns
    df.drop(columns=['person_home_ownership', 'loan_intent'], inplace=True)

    # Feature scaling
    sc = StandardScaler()
    #df['person_age'] = sc.fit_transform(df[['person_age']])
    #df['person_income'] = sc.fit_transform(df[['person_income']])
    #df['person_emp_length'] = sc.fit_transform(df[['person_emp_length']])
    #df['loan_amnt'] = sc.fit_transform(df[['loan_amnt']])
    #df['loan_int_rate'] = sc.fit_transform(df[['loan_int_rate']])
    #df['loan_percent_income'] = sc.fit_transform(df[['loan_percent_income']])
    #df['cb_person_cred_hist_length'] = sc.fit_transform(df[['cb_person_cred_hist_length']])

    df = df[['person_age', 
              'person_income', 
              'person_emp_length', 
              'loan_amnt', 
              'loan_int_rate', 
              'loan_percent_income', 
              'cb_person_cred_hist_length', 
              'OTHER', 
              'OWN', 
              'RENT', 
              'EDUCATION', 
              'HOMEIMPROVEMENT', 
              'MEDICAL', 
              'PERSONAL', 
              'VENTURE', 
              'cb_person_default_on_file', 
              'loan_grade']]
    
    # Return the final dataset
    return df
