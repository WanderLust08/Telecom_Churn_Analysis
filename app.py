from collections import defaultdict
import pandas as pd
from flask import Flask,render_template,request,url_for
import joblib
import sklearn


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/submit",methods=['POST'])
def submit():

    #Gettting a ImmutableDict from request body
    # print(request.form)

    #Converting it into a normal Dictionary 
    data = request.form 
    result = defaultdict(list)
    for key, value in data.items(multi=True):
        result[key].append(value)

    res = {key: values[0] if len(values) == 1 else values for key, values in result.items()}
    print(res)





    # contract = res.pop('Contract')

    new_dict = {}
    for key, value in res.items():
        new_dict[key] = value
        if key == "Contract":
            if value == 'Month-to-Month':
                new_dict['Contract_One year'] = 0
                new_dict['Contract_Two year'] = 0
            elif value == 'One year':
                new_dict['Contract_One year'] = 1
                new_dict['Contract_Two year'] = 0
            elif value == 'Two year':
                new_dict['Contract_One year'] = 0
                new_dict['Contract_Two year'] = 1

            new_dict.pop('Contract')

        if key == "PaymentMethod":
            if value == 'Mailed check':
                new_dict['PaymentMethod_Credit card (automatic)'] = 0   
                new_dict['PaymentMethod_Electronic check'] = 0
                new_dict['PaymentMethod_Mailed check'] = 1
            elif value == 'Electronic check':
                new_dict['PaymentMethod_Credit card (automatic)'] = 0
                new_dict['PaymentMethod_Electronic check'] = 1
                new_dict['PaymentMethod_Mailed check'] = 0
                
            elif value == 'Credit card (automatic)':
                new_dict['PaymentMethod_Credit card (automatic)'] = 1
                new_dict['PaymentMethod_Electronic check'] = 0
                new_dict['PaymentMethod_Mailed check'] = 0
                
                
            else:
                new_dict['PaymentMethod_Credit card (automatic)'] = 0
                new_dict['PaymentMethod_Electronic check'] = 0
                new_dict['PaymentMethod_Mailed check'] = 0
                
            new_dict.pop('PaymentMethod')


    print("NEW DICT : ")
    print(new_dict)

    typecast_value_in_dict(new_dict, 'SeniorCitizen', int)
    typecast_value_in_dict(new_dict, 'tenure', int)
    typecast_value_in_dict(new_dict, 'MonthlyCharges', float)
    typecast_value_in_dict(new_dict, 'TotalCharges', float)



    print("TYPE CASTED : ")
    print(new_dict)





    # Example usage:
    ip_tenure =  new_dict['tenure']
    ip_monthly_charges = new_dict['MonthlyCharges']
    ip_total_charges = new_dict['TotalCharges']

    scaled_tenure, scaled_monthly_charges, scaled_total_charges = manual_min_max_scaling(ip_tenure, ip_monthly_charges, ip_total_charges)


    new_dict['tenure'] = scaled_tenure
    new_dict['MonthlyCharges'] = scaled_monthly_charges
    new_dict['TotalCharges'] = scaled_total_charges


    # new_dict['tenure'] = 0.014085
    # new_dict['MonthlyCharges'] = 0.354229
    # new_dict['TotalCharges'] = 0.010310


    print("Scaled tenure:", scaled_tenure)
    print("Scaled MonthlyCharges:", scaled_monthly_charges)
    print("Scaled TotalCharges:", scaled_total_charges)


    print("\nPOST SCALING : \n")
    print(new_dict)
    
   

    mydata =  PrepareData(new_dict)

    print("FINAL DATA : ",mydata)
    print("INTERNET SERVICE :",mydata['InternetService'])
    # mydata['tenure'] = 0.014085
    # mydata['MonthlyCharges'] = 0.354229
    # mydata['TotalCharges'] = 0.010310




    # print("\nMYDATA FROM SAISH : \n")
    # print(mydata)


  
    print(sklearn.__version__)
    # model = joblib.load('Final_Model.joblib')


    model_path = 'Final_Model.joblib'

# Load the model from the file
    model = joblib.load(model_path)


    predictions = model.predict(mydata)
    print("PREDICTION : ",predictions[0])
    
    if predictions[0] == 0:
        value=False
        print("VALUE :",value)
    else:
        value=True
        print("VALUE :",value)


    
    return render_template('index.html',value=value)







def typecast_value_in_dict(data, key, target_type):
    if key in data:
        try:
            data[key] = target_type(data[key])
        except ValueError:
            print(f"Value for key '{key}' cannot be converted to {target_type}")



def manual_min_max_scaling(ip_tenure, ip_monthly_charges, ip_total_charges):
    # Define the minimum and maximum values for each feature
    min_tenure = 1
    max_tenure = 72  # Maximum tenure in the dataset

    min_monthly_charges = 18.25
    max_monthly_charges = 118.75  # Maximum monthly charges in the dataset

    min_total_charges = 18.8  # Minimum total charges in the dataset
    max_total_charges = 8684.8  # Maximum total charges in the dataset

    # Manually scale the 'tenure' feature
    scaled_tenure = (ip_tenure - min_tenure) / (max_tenure - min_tenure)

    # Manually scale the 'MonthlyCharges' feature
    scaled_monthly_charges = (ip_monthly_charges - min_monthly_charges) / (max_monthly_charges - min_monthly_charges)

    # Manually scale the 'TotalCharges' feature
    scaled_total_charges = (ip_total_charges - min_total_charges) / (max_total_charges - min_total_charges)

    # Return the scaled values
    return scaled_tenure, scaled_monthly_charges, scaled_total_charges



def map_to_target(df, cols, target_mapping):
    print("\nHELLO MTP\n")
    for col in cols:
        df[col] = df[col].map(target_mapping)
    return df


def PrepareData(inputDict):

  data = pd.DataFrame(inputDict,index=[0])
#   if 'customerID' in data.columns:
#     data.drop(columns=['customerID'], inplace=True)

  target_mapping = {
    "No internet service": 0,
    "No phone service": 0,
    "No": 0,
    "Yes": 1,
    "Female": 0,
    "Male": 1,
    "DSL": 1,
    "Fiber optic": 2
  }

  cols  = ['gender','Partner','Dependents','PhoneService','PaperlessBilling','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
           'StreamingTV','StreamingMovies','MultipleLines','InternetService']


  return map_to_target(data, cols, target_mapping)



if __name__ == '__main__':
    app.run(debug=True)