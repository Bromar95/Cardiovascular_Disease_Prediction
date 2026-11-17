import pickle
import numpy as np
import streamlit as st

# Load the trained model (LGBM_model.pkl must be in the same folder as this file)
model = pickle.load(open('LGBM_model.pkl', 'rb'))


def main():
    st.set_page_config(page_title="Cardiovascular Disease Prediction App")

    st.title("Cardiovascular Disease Prediction App")
    st.caption("For educational purposes only. This is not a medical diagnosis.")

    form = st.form('CVD')

    # input variables
    Age = form.number_input('Age (years)', min_value=30, max_value=65, value=50)
    Height = form.number_input('Height (cm)', min_value=120, max_value=220, value=165)
    Weight = form.number_input('Weight (kg)', min_value=30.0, max_value=200.0, value=74.0)
    Systolic = form.slider('Systolic blood pressure (mmHg)', min_value=60, max_value=200, value=120)
    Diastolic = form.slider('Diastolic blood pressure (mmHg)', min_value=50, max_value=180, value=80)
    Gender = form.selectbox('Sex', ['Female', 'Male'])
    Cholesterol = form.selectbox('Cholesterol', ['normal', 'high', 'very high'])
    Glucose = form.selectbox('Glucose', ['normal', 'high', 'very high'])
    Smoke = form.selectbox('Smoker', ['No', 'Yes'])
    Alcohol = form.selectbox('Drinks alcohol', ['No', 'Yes'])
    Physical_Activity = form.selectbox('Physically active', ['Yes', 'No'])

    submit = form.form_submit_button('Predict')

    if submit:
        if Diastolic >= Systolic:
            st.error('Diastolic pressure must be lower than systolic pressure. Please check the readings.')
            return

        # Features must be in the same order the model was trained on
        features = np.array([[
            Age,
            Height,
            Weight,
            Systolic,
            Diastolic,
            1 if Gender == 'Male' else 0,               # Gender_Male
            1 if Cholesterol == 'normal' else 0,        # Cholesterol_normal
            1 if Cholesterol == 'very high' else 0,     # Cholesterol_very high
            1 if Glucose == 'normal' else 0,            # Glucose_normal
            1 if Glucose == 'very high' else 0,         # Glucose_very high
            1 if Smoke == 'Yes' else 0,                 # Smoke_Yes
            1 if Alcohol == 'Yes' else 0,               # Alcohol_yes
            1 if Physical_Activity == 'Yes' else 0,     # Physical_Activity_yes
        ]], dtype=float)

        probability = model.predict_proba(features)[0][1]

        if probability >= 0.5:
            st.error(f'Higher risk of cardiovascular disease ({probability:.0%} predicted probability).')
        else:
            st.success(f'Lower risk of cardiovascular disease ({probability:.0%} predicted probability).')

        st.write('Please talk with a healthcare provider about your heart health.')


if __name__ == '__main__':
    main()
