import streamlit as st
import joblib
import numpy as np

# Define user credentials
credentials = {
    "ingeniero1": "p1",
    "ingeniero2": "p1",
    "b": "b"
}

def login_page():
    st.title("Well Production Predictor")

    st.image("https://cdn1.iconfinder.com/data/icons/oil-and-gas-3/512/Gas_and_Oil-02-512.png", width=150)  # Add your logo here

    username = st.text_input("User")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in credentials and credentials[username] == password:
            st.success(f"Login successful. Welcome, {username}!")
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.error("Invalid username or password")

def model_app():
    st.title("Well Production Predictor")
    st.write(f"Welcome to the predictor, {st.session_state.username}!")
    st.write("Let's help you with today's final oil production prediction.")

    # Load the model
    model = joblib.load('linear_regression_model.pkl')

    # Use a form to group inputs
    with st.form(key='prediction_form'):
        st.subheader("Input features for prediction:")

        pozo_options = [f"Pozo {chr(65 + i)}" for i in range(20)]
        selected_pozo = st.selectbox("Select Pozo", pozo_options)
        pozo_features = [1 if selected_pozo == f"Pozo {chr(65 + i)}" else 0 for i in range(20)]

        llenado_bomba = st.number_input("Llenado Bomba (%)", value=98.00)
        diametro_bomba = st.number_input("Diámetro Bomba (in)", value=2.25)
        longitud_stroke = st.number_input("Longitud Stroke (ft)", value=190.0)
        velocidad_bomba = st.number_input("Velocidad Bomba (SPM)", value=3.0)
        run_time_hoy = st.number_input("Run Time Hoy (rate of day)", value=90.0)

        # Combine all features
        features = np.array(pozo_features + [llenado_bomba, diametro_bomba, longitud_stroke, velocidad_bomba, run_time_hoy]).reshape(1, -1)

        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        prediction = model.predict(features)
        st.success(f"The oil production of this well for today should be around: {prediction[0]} bbl/D")

def main():
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Use this sidebar to navigate through the app.")
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        model_app()
    else:
        login_page()

if __name__ == "__main__":
    main()
