import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def ModelPreProcessing(df):

# Missing Values Treatment
    df.fillna({'loan_int_rate':df['loan_int_rate'].median()},inplace=True)
    df.fillna({'person_emp_length':df['person_emp_length'].median()},inplace=True)
    cr_data = df.copy()

# Outliers Treatment
    cr_age_rmvd = cr_data[cr_data['person_age']<=75]
    cr_age_rmvd.reset_index(drop=True, inplace=True)
    person_emp_rmvd = cr_age_rmvd[cr_age_rmvd['person_emp_length']<=47]
    person_emp_rmvd.reset_index(drop=True, inplace=True)
    cr_data_cat_treated = person_emp_rmvd.copy()
      
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
    cr_data_cat_treated['loan_grade_encoded'] = cr_data_cat_treated['loan_grade'].map(grade_mapping)
    cr_data_cat_treated.drop(columns=['loan_grade'], inplace=True)
    person_home_ownership = pd.get_dummies(cr_data_cat_treated['person_home_ownership'],drop_first=True).astype(int)
    loan_intent = pd.get_dummies(cr_data_cat_treated['loan_intent'],drop_first=True).astype(int)
    cr_data_cat_treated['cb_person_default_on_file_encoded'] = np.where(cr_data_cat_treated['cb_person_default_on_file']=='Y',1,0)
    cr_data_cat_treated.drop(columns=['cb_person_default_on_file'], inplace=True)
    data_to_scale = cr_data_cat_treated.drop(['person_home_ownership','loan_intent','loan_status','cb_person_default_on_file_encoded','loan_grade_encoded'],axis=1)

# Scaling the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_scale)
    scaled_df = pd.DataFrame(scaled_data,columns=['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
           'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'])
    scaled_data_combined = pd.concat([scaled_df,person_home_ownership,loan_intent],axis=1)
    scaled_data_combined['cb_person_default_on_file'] = cr_data_cat_treated['cb_person_default_on_file_encoded']
    scaled_data_combined['loan_grade'] = cr_data_cat_treated['loan_grade_encoded']
    scaled_data_combined['loan_status'] = cr_data_cat_treated['loan_status']
    
# Features and Target Creation    
    target = scaled_data_combined['loan_status']
    features = scaled_data_combined.drop('loan_status',axis=1)

# SMOTE Balancing
    smote = SMOTE()
    balanced_features, balanced_target = smote.fit_resample(features,target)
    
# return the final datasets
    return data_to_scale, features, target, balanced_features, balanced_target