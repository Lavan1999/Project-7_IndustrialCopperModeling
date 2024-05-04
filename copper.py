import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

def selling_price():
    
    st.write(' :white[Fill the below details to find Predicted Selling Price]')

    
    # Define dictionaries for mapping strings to integers
    status_dict = {'Lost': 0, 'Won': 1, 'Draft': 7, 'To be approved': 6, 'Not lost for AM': 2, 'Wonderful': 8, 'Revised': 5, 'Offered': 4, 'Offerable': 3}
    item_type_dict = {'W': 5, 'WI': 6, 'S': 3, 'Others': 1, 'PL': 2, 'IPL': 0, 'SLAWR': 4}

    # custom style for prediction result text - color and position
    country_values = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 
                    78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]

    status_values = ['Won', 'Lost', 'Draft', 'To be approved', 'Not lost for AM',
                    'Wonderful', 'Revised', 'Offered', 'Offerable']

    item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']

    application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 39.0, 
                          40.0, 41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]

    product_ref_values = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407,
                          164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642, 
                        1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026, 
                        1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]
        
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
        file_path = 'regression1.joblib'  # Adjust the file path as per your saved model

        # Load the saved model
        loaded_model = load(file_path)
        
        # Convert status and item_type to their corresponding integer representations
        status_int = status_dict.get(status)
        item_type_int = item_type_dict.get(item_type)

        # Make predictions
        y_pred = loaded_model.predict(np.array([[quantity, customer, country, status_int, item_type_int, 
                                application, thickness, width, product_ref]]))
        selling_price = np.exp(y_pred[0])  # Apply exponential to get back original scale
        
        selling_price = round(selling_price,2)
        
        st.header(f'Predicted Selling Price is: {selling_price}')


def status():
    
    st.write(' :white[Fill the below details to find Predicted Status]')
    
    # Define dictionaries for mapping strings to integers
    item_type_dict = {'W': 5, 'WI': 6, 'S': 3, 'Others': 1, 'PL': 2, 'IPL': 0, 'SLAWR': 4}

    # custom style for prediction result text - color and position
    country_values = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]

    item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']

    application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 39.0, 
                          40.0, 41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]

    product_ref_values = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407,
                          164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642, 
                        1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026, 
                        1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]
    
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
            selling_price = st.number_input(label='Selling_price', min_value=0 , max_value=2500)
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
        file_path = 'classfication.joblib'  # Adjust the file path as per your saved model

        # Load the saved model
        loaded_model = load(file_path)
        
        # Convert status and item_type to their corresponding integer representations
        item_type_int = item_type_dict.get(item_type)

        # Make predictions
        y_pred = loaded_model.predict(np.array([[quantity, customer, country, item_type_int, application, thickness, width, product_ref, selling_price]]))
        
        if y_pred[0] == 1:
            st.header('Prediction status is "Won"')

        else:
            st.header('Predicted status is "Lose"')
        
        
def home():
    
    st.markdown(
        "<h1 style='text-align: center; color: pink;'>Copper Modeling Web Page</h1>",
        unsafe_allow_html=True)

    #st.markdown(f""" <style>.stApp {{
     #                   background: url('https://ruthbroadbent.com/wp-content/uploads/2020/08/1RuthBroadbentCopperWireImaginedLines2017DetailInExhibition1Website-min.jpg');   
      ##             </style> """,unsafe_allow_html=True)

    st.markdown("""
                        <style>
                        div.stButton > button:first-child {
                                                            background-color: #367F89;
                                                            color: white;
                                                            width: 70%}
                        </style>
                    """, unsafe_allow_html=True)
    
    
    st.header(' :green[Importance of Machine Learning in Copper Pricing Prediction]')
    st.write('### :violet[Introduction:]')
    st.write('Welcome to the Copper Modeling Web Page! This platform utilizes machine learning algorithms') 
    st.write('to provide valuable insights into copper pricing and transaction status.')
    
    st.write('### :violet[Importance of Machine Learning:]')
    st.write('In the manufacturing domain, accurately predicting prices and transaction outcomes is paramount.') 
    st.write('Traditional methods often struggle to account for the myriad of factors influencing copper prices and')
    st.write('transaction statuses. This is where machine learning shines. By leveraging vast amounts of historical data and') 
    st.write('sophisticated algorithms, machine learning can uncover hidden patterns and make highly accurate predictions.')
    
        
    st.header(' :green[Use Case of This Project:]')
    st.write('Our project focuses on two key aspects:')

    st.write('### :violet[Selling Price Prediction:]')
    st.write('Users can input various parameters related to copper transactions, and')
    st.write('our machine learning model predicts the selling price. This empowers stakeholders to make informed')
    st.write('decisions about pricing strategies and negotiations.')
    st.write('### :violet[Status Prediction:]') 
    st.write('Users provide transaction details, and our model predicts whether the transaction is likely to be') 
    st.write('successful ("Won") or unsuccessful ("Lost"). This aids in risk assessment and resource allocation.')
        
        
option = option_menu("", ['Home','Selling Price Prediction','Status Prediction'],default_index=0,orientation="horizontal")

if option == 'Home':
    home()
elif option == 'Selling Price Prediction':

    selling_price()
elif option == 'Status Prediction':

    status()