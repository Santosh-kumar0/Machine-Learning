import numpy as np
import pickle as pkl
import streamlit as st

model=pkl.load(open('model.pkl','rb'))
scaler=pkl.load(open('scaler.pkl','rb'))


def predict(user):
    scaled=scaler.transform(np.array(user).reshape(-1,4))
    res=model.predict(scaled)
    return res[0]

if __name__=='__main__':
    st.title("Iris Classification")

    st.divider()

    sepalLength=st.text_input("Enter Sepal Length in cm: ")
    sepalWidth=st.text_input("Enter Sepal width in cm: ")
    petalLength=st.text_input("Enter Petal length in cm: ")
    petalWidth=st.text_input("Enter petal width in cm: ")

    if sepalLength.isalpha() or sepalWidth.isalpha() or petalLength.isalpha() or petalWidth.isalpha():
        st.error("Input must be numeric")
        st.error("The given input is not numeric")
    
    if st.button('Predict'):
        user=[sepalLength,sepalWidth,petalLength,petalWidth]
        res=predict(user)
        st.success(res)

