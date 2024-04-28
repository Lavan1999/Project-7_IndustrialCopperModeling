from datetime import date
import numpy as np
import pickle, joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
from joblib import dump, load


st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: left; color: blue;'>Copper Modeling Web Page</h1>",
    unsafe_allow_html=True)

st.header(' :black[Fill the below details to find Predicted Selling Price]')


st.markdown(f""" <style>.stApp {{
                    background: url('https://ruthbroadbent.com/wp-content/uploads/2020/08/1RuthBroadbentCopperWireImaginedLines2017DetailInExhibition1Website-min.jpg');   
                    background-size: cover}}
                 </style> """,unsafe_allow_html=True)

st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                                                        background-color: #367F89;
                                                        color: white;
                                                        width: 70%}
                    </style>
                """, unsafe_allow_html=True)

def model():
    df1 = pd.read_csv('C:/Users/DELL/Desktop/Project5-Copper/CopperRegression.csv')
    x1=df1.drop(["selling_price", 'item_date', 'delivery date'],axis=1)
    y1=df1["selling_price"]
    x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.2)
    model = RandomForestRegressor(max_depth=20, max_features=None, min_samples_leaf=1, min_samples_split=2).fit(x_train, y_train)
    dump(model, 'regression.joblib')
    predict = model.predict(x_test)[0]
    return 'regression.joblib'

# Define dictionaries for mapping strings to integers
status_dict = {'Lost': 0, 'Won': 1, 'Draft': 7, 'To be approved': 6, 'Not lost for AM': 2, 'Wonderful': 8, 'Revised': 5, 'Offered': 4, 'Offerable': 3}
item_type_dict = {'W': 5, 'WI': 6, 'S': 3, 'Others': 1, 'PL': 2, 'IPL': 0, 'SLAWR': 4}


# custom style for prediction result text - color and position


country_values = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 
                78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]

status_values = ['Won', 'Lost', 'Draft', 'To be approved', 'Not lost for AM',
                'Wonderful', 'Revised', 'Offered', 'Offerable']

item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']

application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 
                    27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 
                    59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]

product_ref_values = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 
                    640405, 640665, 164141591, 164336407, 164337175, 929423819, 
                    1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 
                    1665584642, 1665584662, 1668701376, 1668701698, 1668701718, 
                    1668701725, 1670798778, 1671863738, 1671876026, 1690738206, 
                    1690738219, 1693867550, 1693867563, 1721130331, 1722207579]
    
user_data = pd.DataFrame(columns=[['quantity tons', 'customer', 'country', 'status', 'item_type', 'application', 'thickness', 'width', 'product_ref']])
# Get input from users
with st.form('Regression'):

    col1,col2,col3 = st.columns([0.5,0.1,0.5])

    with col1:

        quantity = st.number_input(label='Quantity Tons' ,min_value= 0.1, max_value= 6.9248)
        country = st.selectbox(label='Country', options=country_values)
        item_type = st.selectbox(label='Item Type', options=item_type_values)
        thickness = st.number_input(label='Thickness', min_value=0.1, max_value=3.28154)
        product_ref = st.selectbox(label='Product Ref', options=product_ref_values)


    with col3:
        
        customer = st.number_input(label='Customer ID', min_value= 30147616 , max_value=30408185)
        status = st.selectbox(label='Status', options=status_values)
        application = st.selectbox(label='Application', options=application_values)
        width = st.number_input(label='Width', min_value=700, max_value=1500)
        st.write('')
        st.write('')
        button = st.form_submit_button(label='SUBMIT')
        
col1,col2 = st.columns([0.65,0.35])
with col2:
    st.caption(body='*Min and Max values are reference only')

# Process user input and make predictions
# Adjusted form submission check
if button:
    # Convert quantity to string       
    # Load the model 
    file_path = 'C:/Users/DELL/Desktop/Project5-Copper/regression.joblib'
    loaded_model = load(file_path)
    
    # Convert status and item_type to their corresponding integer representations
    status_int = status_dict.get(status)
    item_type_int = item_type_dict.get(item_type)
    
    # Make predictions
    user_data.loc[0] = [quantity, customer, country, status_int, item_type_int, application, thickness, width, product_ref]

    user_series = pd.Series([quantity, customer, country, status_int, item_type_int, application, thickness, width, product_ref])
    user_data = user_series.values.reshape(1,-1)
    y_pred = loaded_model.predict(user_data)[0]
    st.write(f"Predicted Selling Price: {y_pred[0]}")
    
    
    if button:
            
            # load the regression pickle model
            with open(r'models\regression_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # make array for all user input values in required order for model prediction
            user_datas = np.array([[quantity, customer, country, status_int, item_type_int, 
                                   application, thickness, width, product_ref]])
            
            # model predict the selling price based on user input
            y_pred = model.predict(user_datas)

            # inverse transformation for log transformation data
            selling_price = np.exp(y_pred[0])

            # round the value with 2 decimal point (Eg: 1.35678 to 1.36)
            selling_price = round(selling_price, 2)

            st.write(selling_price)


    