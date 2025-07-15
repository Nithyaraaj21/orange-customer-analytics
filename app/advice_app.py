import joblib
import numpy as np
import os
import pandas as pd
import streamlit as st

# Set page configuration
PAGE_TITLE = "Orange Advice app"
PAGE_ICON = ":orange_heart:"
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# Set thin divider line
horizontal_bar = "<hr style='margin-top: 0; margin-bottom: 0; height: 2px; \
                        border: 1px solid #949494;'><br>"

# Set the style of the markdown text
st.markdown("""
    <style>
        .stMarkdown {
            margin-bottom: -40px;  /* Reduce the bottom margin */
            margin-top: 10px;  /* Increase the top margin */
        }
    </style>
""", unsafe_allow_html=True)

# Set the style of buttons
st.write('<style>div.stButton > button:first-child {background-color: white;color: black;}</style>', unsafe_allow_html=True)

# Set the style of the container
st.markdown("""
    <style>
        .custom-container {
            border: 2px solid #ff9900;  /* Orange border */
            font-size: 18px;
            text-align:center;
            margin-top:-10px;
        }
    </style>
""", unsafe_allow_html=True)


# Set the path to the dataset
data_link = os.path.abspath("../data/engineered/with_outliers/FULL_test_eng_with_outlers.csv")
# Load test dataset
@st.cache_data # This function will be cached
def load_data(link: str):
    return pd.read_csv(link)

test_data = load_data(data_link)

# Set the description of the columns
columns = test_data.columns
description = [
    "Current tariff plan:",
    "New tariff plan:",
    "Price of current tariff plan:",
    "Price of previous tariff plan:",
    "Average monthly data usage over the last 3 months (Gb):",
    "Average monthly duration of calls over the last 3 months (min):",
    "Data usage in the current month (Gb):",
    "Call duration in the current month (min):",
    "Total spent two months ago:",
    "Number of days since the last change in the tariff plan:",
    "Total spent three months ago:",
    "Revenue from calls in the current month:",
    "Total spent in the current month:",
    "Ratio of out-of-bundle to out-of-pocket expenditures:",
    "Average revenue from calls over the last 3 months:",
    "Number of tariff plan changes in the last 12 months:",
    "A metric for evaluating customer behavior:",
    "Months since the last upsell activity:",
    "Months since the last downsell activity:",
    "Ratio of data usage compared to plan limits (specific interpretation needed).",
    "Ratio of used to available minutes.",
    "If the customer uses a promotion:",
    "Total number of plan changes by the customer:",
    "If the customer has changed plans in the last year:",
    "Number of plan upgrades from 9 to 12 months ago:",
    "Number of plan upgrades from 6 to 9 months ago:",
    "Number of plan upgrades from 3 to 6 months ago:",
    "Number of plan upgrades in the last 3 months:",
    "Number of plan downgrades from 9 to 12 months ago:",
    "Number of plan downgrades from 6 to 9 months ago:",
    "Number of plan downgrades from 3 to 6 months ago:",
    "Number of plan downgrades in the last 3 months:",
    "If the customer has changed plans recently (checkbox):"
]
columns_description = dict(zip(columns, description))

# Define columns for user input
user_columns = ['old_tp', 'tp_price_cur', 'tp_price_prev',
       'data_usage_avg_3m', 'calls_dur_avg_3m', 'data_usage_cur',
       'calls_dur_cur', 'change_n_d', 'spent_cur', 'tp_n_12m',
       'move_in_LY']

other_columns = [col for col in test_data.columns if col not in user_columns and col not in ['new_tp', 'moved']]

# Create a dictionary to store user inputs
user_inputs = {}

# Define the global variable for the tariff plans
tps = {
    "Go Light": 0,
    "Go Plus": 1,
    "Go Intense": 2,
    "Go Extreme": 3
}

# Function to predict the probability of a customer moving to a new tariff plan
def predict(user_data: pd.DataFrame, index: int, old_tp: int):    
    # First prediction
    model_link = os.path.abspath("../models/")
    first_model = joblib.load(os.path.join(model_link, "L1_model.joblib"))
    probabilities_1 = first_model.predict_proba(user_data)

    st.subheader("Probabilities:")
    st.markdown("")
    # container = st.container(border=True)
    c1,c2,c3,c4 = st.columns([1, 1, 1, 2])
    with c1:
        c1.container(border=False, height=60).markdown(
            f'<div class="custom-container">Not move: {probabilities_1[0, 0] * 100:.2f}%</div>', 
            unsafe_allow_html=True
        )
    with c2:
        c2.container(border=False, height=60).markdown(
            f'<div class="custom-container">Move: {probabilities_1[0, 1] * 100:.2f}%</div>', 
            unsafe_allow_html=True
        )       
    
    probabilities_1 = probabilities_1[:, 1]
    
    # Show recommendation based on the probability
    st.subheader("Recommendation:")
    if probabilities_1[0] <= 0.761:            
        st.write("Current tariff plan is best suited for the customer. We do not recommend to change it for now.")
        st.markdown("")
        st.markdown("")
        # Show the image of the current tariff plan
        for key, value in tps.items():
            if value == user_inputs['old_tp']:
                img = "imgs/" + str(key).lower().replace(" ", "-") + ".png"
                st.image(img, width=250)
        st.markdown("")
        st.markdown("")
    else:
        st.write("It is not the optimal tariff plan for the customer. We recommend to consider moving to a new tariff plan.")
        st.markdown("")
        st.markdown("")

        output = user_data.copy()

        output['moved'] = np.where(probabilities_1 > 0.761, 1, 0)
        output['old_tp'] = int(tps[old_tp])
        output['new_tp'] = test_data.loc[index, 'new_tp']

        output_cols = [col for col in output.columns if col != 'price_per_gb']
        output = output[output_cols]

        probabilities_2 = None

        if output['moved'].any() == 1:
            if output['old_tp'].any() == 0:
                second_model = joblib.load(os.path.join(model_link, "tp0_lgb.pkl"))
                probabilities_2 = second_model.predict_proba(output)                    
            elif output['old_tp'].any() == 1:
                second_model = joblib.load(os.path.join(model_link, "tp1_lgb.pkl"))
                probabilities_2 = second_model.predict_proba(output)
            elif output['old_tp'].any() == 2:
                second_model = joblib.load(os.path.join(model_link, "tp2_lgb.pkl"))
                probabilities_2 = second_model.predict_proba(output)
            elif output['old_tp'].any() == 3:
                second_model = joblib.load(os.path.join(model_link, "tp3_lgb.pkl"))
                probabilities_2 = second_model.predict_proba(output)
        
        # Create list of possible tariff plans (excluding the current one)
        tps_2 = {key: value for key, value in tps.items() if value != output['old_tp'].values[0]}

        # Display the probabilities for the new tariff plan
        if probabilities_2.all():
            st.subheader("Probabilities for the new tariff plan:")
            st.markdown("")
            c1,c2,c3,c4 = st.columns([1, 1, 1, 2])
            with c1:
                c1.container(border=False, height=60).markdown(
                    f'<div class="custom-container">Move to {list(tps_2.keys())[0]}: {probabilities_2[0, 0] * 100:.2f}%</div>', 
                    unsafe_allow_html=True
                )
            with c2:
                c2.container(border=False, height=60).markdown(
                    f'<div class="custom-container">Move to {list(tps_2.keys())[1]}: {probabilities_2[0, 1] * 100:.2f}%</div>', 
                    unsafe_allow_html=True
                )
            with c3:
                c3.container(border=False, height=60).markdown(
                    f'<div class="custom-container">Move to {list(tps_2.keys())[2]}: {probabilities_2[0, 2] * 100:.2f}%</div>', 
                    unsafe_allow_html=True
                )

            # Find the index of the biggest probability
            new_tp = np.where(probabilities_2[0] == max(probabilities_2[0]))[0][0]
            # Find the name of the tariff plan with the biggest probability
            new_tp_name = list(tps_2.keys())[new_tp]
            # Show the image of the tariff plan with the biggest probability
            img = "imgs/" + str(new_tp_name).lower().replace(" ", "-") + ".png"
            st.image(img, width=250)
            st.markdown("")
            st.markdown("")

        

# Function to show the content of the page for Customer Advice app
def show_content_customer(DEFAULTS: dict, columns_description: dict, test_data: pd.DataFrame, index: int):
    with st.form("form_1"):
        col1, spacer, col2, spacer, col3 = st.columns([1.5, 0.05, 1.5, 0.05, 1])        

        with col1:            
            old_tp = st.radio(
                columns_description["old_tp"], 
                options=tps.keys(),
                index=int(DEFAULTS["old_tp"]),
                horizontal=True
            )
            user_inputs["old_tp"] = tps[old_tp]

            st.markdown("")
            tp_price_cur = st.number_input(
                columns_description["tp_price_cur"], 
                min_value=8, 
                max_value=80, 
                value=int(DEFAULTS["tp_price_cur"])
            )
            user_inputs["tp_price_cur"] = tp_price_cur
            
            st.markdown("")
            move_in_LY = st.checkbox(
                columns_description["move_in_LY"], 
                value=DEFAULTS["move_in_LY"]
            )

            if move_in_LY:
                move_in_LY = 1
                tp_n_12m = st.slider(
                    columns_description["tp_n_12m"], 
                    value=int(DEFAULTS["tp_n_12m"]), 
                    min_value=1, 
                    max_value=12
                )
                
                change_n_d = st.number_input(
                    columns_description["change_n_d"], 
                    min_value=0, 
                    max_value=1500, 
                    value=int(DEFAULTS["change_n_d"])
                )

                tp_price_prev = st.number_input(
                    columns_description["tp_price_prev"], 
                    min_value=8, 
                    max_value=80, 
                    value=int(DEFAULTS["tp_price_prev"])
                )                
                
            else:
                move_in_LY = 0
                tp_n_12m = 0
                tp_price_prev = tp_price_cur
                change_n_d = 0

            user_inputs["move_in_LY"] = move_in_LY
            user_inputs["tp_price_prev"] = tp_price_prev
            user_inputs["change_n_d"] = change_n_d
            user_inputs["tp_n_12m"] = tp_n_12m
            
        with col2:
            spent_cur = st.number_input(
                columns_description["spent_cur"], 
                min_value=0.0, 
                max_value=600.0, 
                value=DEFAULTS["spent_cur"]
            )
            user_inputs["spent_cur"] = spent_cur

            calls_dur_cur = st.number_input(
                columns_description["calls_dur_cur"], 
                min_value=0, 
                max_value=15000, 
                value=int(DEFAULTS["calls_dur_cur"])
            )
            user_inputs["calls_dur_cur"] = calls_dur_cur
            
            calls_dur_avg_3m = st.number_input(
                columns_description["calls_dur_avg_3m"], 
                min_value=0, 
                max_value=15000, 
                value=int(DEFAULTS["calls_dur_avg_3m"])
            )
            user_inputs["calls_dur_avg_3m"] = calls_dur_avg_3m

            data_usage_cur = st.number_input(
                columns_description["data_usage_cur"], 
                min_value=0, 
                max_value=600, 
                value=int(DEFAULTS["data_usage_cur"]/1000),
            ) * 1000
            user_inputs["data_usage_cur"] = data_usage_cur
            
            data_usage_avg_3m = st.number_input(
                columns_description["data_usage_avg_3m"], 
                min_value=0, 
                max_value=600, 
                value=int(DEFAULTS["data_usage_avg_3m"]/1000)
            ) * 1000
            user_inputs["data_usage_avg_3m"] = data_usage_avg_3m

        with col3:    
            st.markdown("")
            st.image("imgs/orange_tariff_social.png", use_column_width=True)

        start_prediction = st.form_submit_button("Predict") 

    if start_prediction:
        # Convert the user inputs to a DataFrame
        user_input_df = pd.DataFrame(user_inputs, index=[index])
        user_data = test_data.loc[index, other_columns].to_frame().T
        # Concatenate the user input DataFrame with the user data
        user_data = pd.concat([user_data, user_input_df], axis=1)
        user_data['price_per_gb'] = user_data['tp_price_cur'] / user_data['data_usage_avg_3m'] + 1
        # Reorder the columns in user_data to match the expected order
        user_data = user_data[['tp_price_cur', 'tp_price_prev', 'data_usage_avg_3m', 'calls_dur_avg_3m', 'data_usage_cur', 'calls_dur_cur', 'spent_m2', 'change_n_d', 'spent_m3', 'calls_rev_cur', 'spent_cur', 'ratio_oob_oop', 'calls_rev_avg_3m', 'tp_n_12m', 'distance_metric', 'm_n_since_last_upsell', 'm_n_since_last_downsell', 'tp_data_ratio', 'minutes_ratio', 'promotion', 'total_moves', 'move_in_LY', 'move_up_9to12m.abs', 'move_up_6t9m.abs', 'move_up_3to6m.abs', 'move_up_0to3m.abs', 'move_down_9to12m.abs', 'move_down_6to9m.abs', 'move_down_3to6m_abs', 'move_down_0to3m_abs', 'price_per_gb']]
        with st.form("form_2"):
            predict(pd.DataFrame(user_data, index=[index]), index, old_tp)
            rerun = st.form_submit_button("Rerun")

        if rerun:
            st.experimental_rerun()
        

# Function to show the content of the page for Orange app
def show_content_orange(DEFAULTS: dict, columns_description: dict, test_data: pd.DataFrame, index: int):
    with st.form("form_orange"):
        col1, spacer, col2, spacer, col3 = st.columns([1.5, 0.05, 1.5, 0.05, 1])        

        with col1:            
            old_tp = st.radio(
                columns_description["old_tp"], 
                options=tps.keys(),
                index=int(DEFAULTS["old_tp"]),
                horizontal=True
            )
            user_inputs["old_tp"] = tps[old_tp]

            st.markdown("")
            tp_price_cur = st.number_input(
                columns_description["tp_price_cur"], 
                min_value=8, 
                max_value=80, 
                value=int(DEFAULTS["tp_price_cur"])
            )
            user_inputs["tp_price_cur"] = tp_price_cur
            
            st.markdown("")
            move_in_LY = st.checkbox(
                columns_description["move_in_LY"], 
                value=DEFAULTS["move_in_LY"]
            )

            if move_in_LY:
                move_in_LY = 1
                tp_n_12m = st.slider(
                    columns_description["tp_n_12m"], 
                    value=int(DEFAULTS["tp_n_12m"]), 
                    min_value=1, 
                    max_value=12
                )
                
                change_n_d = st.number_input(
                    columns_description["change_n_d"], 
                    min_value=0, 
                    max_value=1500, 
                    value=int(DEFAULTS["change_n_d"])
                )

                tp_price_prev = st.number_input(
                    columns_description["tp_price_prev"], 
                    min_value=8, 
                    max_value=80, 
                    value=int(DEFAULTS["tp_price_prev"])
                )                
                
            else:
                move_in_LY = 0
                tp_n_12m = 0
                tp_price_prev = tp_price_cur
                change_n_d = 0

            user_inputs["move_in_LY"] = move_in_LY
            user_inputs["tp_price_prev"] = tp_price_prev
            user_inputs["change_n_d"] = change_n_d
            user_inputs["tp_n_12m"] = tp_n_12m
            
        with col2:
            spent_cur = st.number_input(
                columns_description["spent_cur"], 
                min_value=0.0, 
                max_value=600.0, 
                value=DEFAULTS["spent_cur"]
            )
            user_inputs["spent_cur"] = spent_cur

            calls_dur_cur = st.number_input(
                columns_description["calls_dur_cur"], 
                min_value=0, 
                max_value=15000, 
                value=int(DEFAULTS["calls_dur_cur"])
            )
            user_inputs["calls_dur_cur"] = calls_dur_cur
            
            calls_dur_avg_3m = st.number_input(
                columns_description["calls_dur_avg_3m"], 
                min_value=0, 
                max_value=15000, 
                value=int(DEFAULTS["calls_dur_avg_3m"])
            )
            user_inputs["calls_dur_avg_3m"] = calls_dur_avg_3m

            data_usage_cur = st.number_input(
                columns_description["data_usage_cur"], 
                min_value=0, 
                max_value=600, 
                value=int(DEFAULTS["data_usage_cur"]/1000),
            ) * 1000
            user_inputs["data_usage_cur"] = data_usage_cur
            
            data_usage_avg_3m = st.number_input(
                columns_description["data_usage_avg_3m"], 
                min_value=0, 
                max_value=600, 
                value=int(DEFAULTS["data_usage_avg_3m"]/1000)
            ) * 1000
            user_inputs["data_usage_avg_3m"] = data_usage_avg_3m

        with col3:    
            st.markdown("")
            st.image("imgs/orange_tariff_social.png", use_column_width=True)

        # Convert the user inputs to a DataFrame
        user_input_df = pd.DataFrame(user_inputs, index=[index])
        user_data = test_data.loc[index, other_columns].to_frame().T
        # Concatenate the user input DataFrame with the user data
        user_data = pd.concat([user_data, user_input_df], axis=1)
        user_data['price_per_gb'] = user_data['tp_price_cur'] / user_data['data_usage_avg_3m'] + 1
        # Reorder the columns in user_data to match the expected order
        user_data = user_data[['tp_price_cur', 'tp_price_prev', 'data_usage_avg_3m', 'calls_dur_avg_3m', 'data_usage_cur', 'calls_dur_cur', 'spent_m2', 'change_n_d', 'spent_m3', 'calls_rev_cur', 'spent_cur', 'ratio_oob_oop', 'calls_rev_avg_3m', 'tp_n_12m', 'distance_metric', 'm_n_since_last_upsell', 'm_n_since_last_downsell', 'tp_data_ratio', 'minutes_ratio', 'promotion', 'total_moves', 'move_in_LY', 'move_up_9to12m.abs', 'move_up_6t9m.abs', 'move_up_3to6m.abs', 'move_up_0to3m.abs', 'move_down_9to12m.abs', 'move_down_6to9m.abs', 'move_down_3to6m_abs', 'move_down_0to3m_abs', 'price_per_gb']]
        
        st.divider()
        
        predict(pd.DataFrame(user_data, index=[index]), index, old_tp)
        rerun = st.form_submit_button("Rerun")

        if rerun:
            st.experimental_rerun()

# Main part of the app

# Display logo in sidebar
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            color: black;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }        
    </style>
    """, unsafe_allow_html=True
)
with st.sidebar:
    st.image("imgs/orange_mobile.png", width=150)


# Display title and base content
st.title("Orange Advice app :orange_heart:")
st.markdown(horizontal_bar, True)

st.markdown(
    """
    Select which app to use:
"""
)
selected_app = st.radio(
    "Select the app to use:",
    ["Orange app", "Customer Advice app"],
    label_visibility="hidden",
    horizontal=True
)
st.markdown("")
if selected_app == "Orange app":
    # Select an index from the dataset
    index = int(st.number_input(
        "Select an id of the customer", 
        min_value=0, 
        max_value=test_data.shape[0]-1, 
        value=0,
        on_change=None
    ))
    # Submit button
    submitted = st.button("Search Dataset")

    # Check if the index has been submitted
    if submitted:
        # Generate number input fields for each column and collect user inputs
        DEFAULTS = dict(zip(columns, test_data.loc[index]))
        show_content_orange(DEFAULTS, columns_description, test_data, index)


elif selected_app == "Customer Advice app":
    index = 5
    DEFAULTS = dict(zip(columns, test_data.loc[index]))

    # Define binary and numerical columns
    # binary_columns = ['promotion', 'moved']
    # numerical_columns = [col for col in other_columns if col not in binary_columns]
    # Calculate the mode of each binary column and the median of each numerical column
    # mode_values = test_data[binary_columns].mode().iloc[0]
    # median_values = test_data[numerical_columns].median()
    # Fill in DEFAULTS with the mode for binary columns and the median for numerical columns
    # for column in binary_columns:
        # DEFAULTS[column] = mode_values[column]
    # for column in numerical_columns:
        # DEFAULTS[column] = median_values[column]
    
    show_content_customer(DEFAULTS, columns_description, test_data, index)
    

# Display footer
footer = """
    <style>    
    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #FF6600;
    color: white;
    text-align: center;
    padding: 10px;
    }
    </style>
    <div class="footer">
        Developed with  ❤
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)