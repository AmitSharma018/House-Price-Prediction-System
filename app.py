import streamlit as st
import pandas as pd
import joblib
import time

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.image("images/banner.png", use_container_width=True)


st.title("🏠 House Price Prediction System")

st.markdown("""
### Predict House Prices using Machine Learning

This application predicts the estimated price of a house using a trained **Random Forest Regressor**.

Enter the property details below and click **Predict House Price**.
""")



model = joblib.load("models/model.pkl")



st.sidebar.title("📌 Project Information")

st.sidebar.success("Model Used: Random Forest")

st.sidebar.write("### Features")

st.sidebar.write("""
- Bedrooms
- Bathrooms
- Living Area
- Lot Area
- Floors
- Waterfront
- View
- Condition
- Basement Area
- Year Built
- Year Renovated
- City
- State ZIP
""")
st.sidebar.subheader("📈 Model Performance")

st.sidebar.write("Algorithm : Random Forest")
st.sidebar.write("MAE : 165,856")
st.sidebar.write("RMSE : 988,081")
st.sidebar.write("R² Score : 0.043")

st.sidebar.divider()

st.sidebar.info("Developed by Amit Sharma")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 2.0)
    sqft_living = st.number_input("Living Area (sqft)", value=1500)
    sqft_lot = st.number_input("Lot Area (sqft)", value=5000)
    floors = st.number_input("Floors", 1.0, 5.0, 1.0)
    waterfront = st.selectbox("Waterfront", [0, 1])
    view = st.slider("View", 0, 4, 2)

with col2:
    condition = st.slider("Condition", 1, 5, 3)
    sqft_above = st.number_input("Above Ground Area", value=1200)
    sqft_basement = st.number_input("Basement Area", value=300)
    yr_built = st.number_input("Year Built", value=2000)
    yr_renovated = st.number_input("Year Renovated", value=0)


    city_name = st.selectbox(
    "City",
    ["Seattle", "Kent", "Bellevue", "Redmond", "Shoreline"]
)

zip_name = st.selectbox(
    "State ZIP",
    ["WA 98119", "WA 98042", "WA 98008", "WA 98052", "WA 98133"]
)

city_mapping = {
    "Seattle": 35,
    "Kent": 18,
    "Bellevue": 3,
    "Redmond": 31,
    "Shoreline": 36
}

zip_mapping = {
    "WA 98119": 58,
    "WA 98042": 26,
    "WA 98008": 7,
    "WA 98052": 31,
    "WA 98133": 62
}

city = city_mapping[city_name]
statezip = zip_mapping[zip_name]


st.markdown("---")

predict = st.button(
    "🏠 Predict House Price",
    use_container_width=True
)

if predict:
    input_data = pd.DataFrame([
        [
            bedrooms,
            bathrooms,
            sqft_living,
            sqft_lot,
            floors,
            waterfront,
            view,
            condition,
            sqft_above,
            sqft_basement,
            yr_built,
            yr_renovated,
            city,
            statezip
        ]
    ], columns=[
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "city",
        "statezip"
    ])
    start = time.time()

    prediction = model.predict(input_data)

    end = time.time()

    st.success("Prediction Completed Successfully!")

    st.metric(
        label="🏠 Estimated House Price",
        value=f"${prediction[0]:,.2f}"
    )

    st.info(f"Prediction Time: {(end-start):.4f} seconds")

    prediction = model.predict(input_data)

    st.balloons()

    st.success("Prediction Completed Successfully!")

    st.metric(
        label="🏡 Estimated House Price",
        value=f"${prediction[0]:,.2f}"
    )
    st.divider()

st.markdown("""
<center>

### Developed by Amit Sharma

AI & ML Internship Project

Rajkiya Engineering College Mainpuri

</center>
""", unsafe_allow_html=True)