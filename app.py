import numpy as np
import pickle
import pandas as pd
import streamlit as st


path = 'F:/Ineuron_Internship/Project_CCDP/BestModel/'
pickle_in = open(path + "Random_forest.pkl","rb")
classifier = pickle.load(pickle_in)
scaler = pickle.load(open('standard_scalar.pkl', 'rb'))


def predict_defaulter(limitbal, sex, education, marriage, age, pay1, pay2, pay3, totalbillamt, totalpayamt):

    data = [limitbal, sex, education, marriage, age, pay1, pay2, pay3, totalbillamt, totalpayamt ]
    feature_value = [np.array(data)]
    features_names = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE','AGE' , 'PAY_1', 'PAY_2', 'PAY_3', 'TOTAL_BILL_AMT', 'TOTAL_PAY_AMT']

    df = pd.DataFrame(feature_value, columns=features_names)

    std_data = scaler.transform(df)

    my_predict = classifier.predict(std_data)

    if my_predict == 1:
        return f"Customer is going to be a Defaulter!!!!"

    if my_predict == 0:
        return f"Customer will pay the credit card payment on time."



def main():
    st.title("Credit Card Default Prediction")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Defaulter Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    limitbal = st.number_input("Enter the Balance Limit", min_value=10000)
    if limitbal == 0 or limitbal >= 10000000:
        return st.error('Provide valid Balance Limit')

    sex = st.selectbox("Gender", ['Male', 'Female'])
    if sex == 'Male':
        sex = 1
    else:
        sex = 2

    education = st.selectbox("Education", ['Graduate School', 'University', 'High School', 'Other'])
    if education == 'Graduate School':
        education = 1
    elif education == 'University':
        education = 2
    elif education == 'High School':
        education = 3
    else:
        education = 4


    marriage = st.selectbox("Marriage", ['Married', 'Single'])
    if marriage == 'Married':
        marriage = 1
    elif marriage == 'Single':
        marriage = 2

    age = st.number_input("Age", min_value=21, max_value=100)
    if age == 0 or age>100:
        return st.error('Please Provide Valid Age')

    pay1 = st.selectbox('Pay1',['No Due','One Month Due','Two Month Due', 'Three Month Due', 'Four Month Due', 'Five Month Due', 'Six Month Due', 'Seven Month Due', 'Eight Month Due'])
    if pay1 == 'No Due':
        pay1 = 0
    elif pay1 == 'One Month Due':
        pay1 = 1
    elif pay1 == 'Two Month Due':
        pay1 = 2
    elif pay1 == 'Three Month Due':
        pay1 = 3
    elif pay1 == 'Four Month Due':
        pay1 = 4
    elif pay1 == 'Five Month Due':
        pay1 = 5
    elif pay1 == 'Six Month Due':
        pay1 = 6
    elif pay1 == 'Seven Month Due':
        pay1 = 7
    elif pay1 == 'Eight Month Due':
        pay1 = 8

    pay2 = st.selectbox('Pay2', ['No Due', 'One Month Due', 'Two Month Due', 'Three Month Due', 'Four Month Due',
                                 'Five Month Due', 'Six Month Due', 'Seven Month Due', 'Eight Month Due'])
    if pay2 == 'No Due':
        pay2 = 0
    elif pay2 == 'One Month Due':
        pay2 = 1
    elif pay2 == 'Two Month Due':
        pay2 = 2
    elif pay2 == 'Three Month Due':
        pay2 = 3
    elif pay2 == 'Four Month Due':
        pay2 = 4
    elif pay2 == 'Five Month Due':
        pay2 = 5
    elif pay2 == 'Six Month Due':
        pay2 = 6
    elif pay2 == 'Seven Month Due':
        pay2 = 7
    elif pay2 == 'Eight Month Due':
        pay2 = 8

    pay3 = st.selectbox('Pay3', ['No Due', 'One Month Due', 'Two Month Due', 'Three Month Due', 'Four Month Due',
                                 'Five Month Due', 'Six Month Due', 'Seven Month Due', 'Eight Month Due'])
    if pay3 == 'No Due':
        pay3 = 0
    elif pay3 == 'One Month Due':
        pay3 = 1
    elif pay3 == 'Two Month Due':
        pay3 = 2
    elif pay3 == 'Three Month Due':
        pay3 = 3
    elif pay3 == 'Four Month Due':
        pay3 = 4
    elif pay3 == 'Five Month Due':
        pay3 = 5
    elif pay3 == 'Six Month Due':
        pay3 = 6
    elif pay3 == 'Seven Month Due':
        pay3 = 7
    elif pay3 == 'Eight Month Due':
        pay3 = 8

    totalbillamt = st.number_input('Total Bill Amount Till 6 Month')
    totalpayamt = st.number_input('Total Pay Amount Till 6 Month')

    result = ""
    if st.button("Predict"):

        result = predict_defaulter(limitbal, sex, education, marriage, age, pay1, pay2, pay3, totalbillamt, totalpayamt)
    st.success(result)
    # if st.button("About the Project"):
    #     st.write('This is a Credit Card Default Project, in this we predict the defaulter customer with the help of input data \
    #              and for more, you can visit this [CCDP Project](https://github.com/PrajjwalSule21) link')
    #     st.write("Project was Build by [Prajjwal Sule](https://www.linkedin.com/in/prajjwal-sule/)")



if __name__ == '__main__':
    main()



