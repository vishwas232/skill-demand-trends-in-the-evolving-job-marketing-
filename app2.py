
####base libraries
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")




import joblib
import pickle

mlb=joblib.load("mlb.pkl")
dt = joblib.load('multi_output_dt.pkl')
ohe=joblib.load('ohe.pkl')
le_job_type = joblib.load("le_job_type.pkl")
le_working_type = joblib.load("le_working_type.pkl")
le_company_size = joblib.load("le_company_size.pkl")


##def load_model():
    ##return joblib.load("multi_output_dt.pkl")

##def predict(input_data, model):
    ##return model.predict(input_data)


import streamlit as st
# Streamlit UI
st.title("Job Market Prediction")
st.write("Enter job details to predict required skills.")


# User Inputs
job_title = st.selectbox("Job title", ['product manager','cyber security analyst','graphic designer','software engineer','data scientist'])
company_name = st.selectbox("company_name", ['google','hcl','ibm','amazon','deloitte','microsoft','wipro','tcs','accenture','infosys'])
job_type = st.selectbox("Job Type", ["full-time", 'internship',"part-time", "contract"])
working_type = st.selectbox("Working Type", ["remote", "onsite", "hybrid"])
number_of_applicants= st.number_input("Number of Applicants(10-500)", min_value=1, step=1)
company_size = st.selectbox("Company Size", ["small", "medium", "large"])
min_experience = st.number_input("Min Experience (0-5yrs)", min_value=0, step=1)
max_experience = st.number_input("Max Experience (2-10yrs)", min_value=0, step=1)
min_salary = st.number_input("Min Salary (3-20LPA)", min_value=0.0, step=0.1)
max_salary = st.number_input("Max Salary (5-20LPA)", min_value=0.0, step=0.1)

# Load Model
##model = load_model()

# Predict
if st.button("Predict Skills"):
    input_data = pd.DataFrame([[job_title, company_name, job_type, working_type, number_of_applicants,
                                company_size, min_experience, max_experience, min_salary, max_salary]],
                              columns=["Job Title", "Company Name", "Job Type", "Working Type", "Number of Applicants",
                                       "Company Size", "Min Experience in years", "Max Experience in years",
                                       "Min Salary in LPA", "Max Salary in LPA"])
    
    row=input_data.copy()


    for str in row["Job Title"]:
        if str=="cyber security analyst":
            row["Job Title"]=row["Job Title"].replace(str,11.44)
        elif str=="data scientist":
            row["Job Title"]=row["Job Title"].replace(str, 11.02)   
        elif str=="graphic designer":
            row["Job Title"]=row["Job Title"].replace(str, 11.25)  
        elif str=="product manager":
            row["Job Title"]=row["Job Title"].replace(str, 11.27) 
        elif str=="software engineer":
            row["Job Title"]=row["Job Title"].replace(str, 10.58) 
    # One-Hot Encoding
    row_ohe = ohe.transform(row[["Company Name"]]).toarray()
    row_ohe = pd.DataFrame(row_ohe, columns=ohe.get_feature_names_out())
    row = pd.concat([row.drop("Company Name", axis=1), row_ohe], axis=1)
    
    
    
    #label encoding
    row["Job Type"] = le_job_type.transform([row["Job Type"]])[0]  # Convert single value
    row["Working Type"] = le_working_type.transform([row["Working Type"]])[0]
    row["Company Size"] = le_company_size.transform([row["Company Size"]])[0]
    
   
    print("Processed Input for Prediction:")
    st.dataframe(row)

    # **Model Prediction (Assuming Model is Loaded)**
    multi_output_clf = joblib.load("multi_output_dt.pkl")  # Load trained model
    
    prediction = multi_output_clf.predict(row)
    prediction=prediction.tolist()

    st.write("\n Predicted Skills:")
    st.dataframe(pd.DataFrame(prediction, columns=mlb.classes_))
    required_skills=[]
    #st.write(prediction)
    for i in range(len(prediction[0])):
        if prediction[0][i]==1:
            required_skills.append(mlb.classes_[i])
    st.write(required_skills)
    
    #prediction = predict(row, model)
    #st.write("Predicted Skills:", prediction)