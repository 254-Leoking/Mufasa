import streamlit as st
import pickle
import numpy as np

# Load the saved model and encoders


def load_model():
    with open(r'C:\Users\user\PycharmProjects\pythonProject1\pythonProject\ml\saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()
regressor = data['model']
le_country = data['le_country']
le_education = data['le_education']

# Show the prediction page


def clean_education(x):
    if "Bachelor’s degree" in x or "Bachelor's degree" in x:
        return "Bachelor's degree"
    if "Master’s degree" in x or "Master's degree" in x:
        return "Master's degree"
    if "Professional degree" in x or "Other doctoral degree" in x:
        return "Post grad"
    return "Less than a Bachelors"


def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    # Country selection
    countries = (
        "United States", "India", "Germany", "Canada", "United Kingdom", "Brazil", "France",
        "Spain", "Australia", "Netherlands", "Sweden", "Italy", "Poland", "Russian Federation"
    )

    # Education Level selection
    education = (
        "Less than a Bachelor's Degree", "Bachelor’s Degree", "Master’s Degree", "Post grad"
    )

    # Input fields for the web form
    country = st.selectbox("Country", countries)
    education_level = st.selectbox("Education Level", education)
    education_level = clean_education(education_level)
    experience = st.slider("Years of Experience", 0, 50, 3)

    # Button for prediction
    ok = st.button("Calculate Salary")

    if ok:
        x = np.array([[country, education_level, experience]])

        # Apply the label encoders
        x[:, 0] = le_country.transform(x[:, 0])
        x[:, 1] = le_education.transform(x[:, 1])
        x = x.astype(float)

        # Predict salary
        salary = regressor.predict(x)
        st.subheader(f"The estimated salary is ${salary[0]:,.2f} per year")

# Run the web app


if __name__ == "__main__":
    show_predict_page()



