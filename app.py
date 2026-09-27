import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CollegeMatch AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("college_match_model.pkl")


model = load_model()


# =========================================================
# CHECK DATA
# =========================================================

required_columns = [
    "INSTNM",
    "CITY",
    "STABBR",
    "CONTROL",
    "SAT_AVG",
    "ACTCMMID",
    "TUITIONFEE_OUT",
    "UGDS"
]



if missing_columns:
    st.error(
        f"Missing columns in college_match.csv: {missing_columns}"
    )
    st.stop()


# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna(
    subset=required_columns
).copy()


# =========================================================
# APP BACKGROUND
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4F7FB;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.title("🎓 CollegeMatch AI")

st.subheader(
    "Find colleges that fit your preferences."
)

st.write(
    "Choose your academic, financial, location, and "
    "campus preferences to create a personalized "
    "college shortlist."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 CollegeMatch AI")

st.sidebar.write(
    "Set your college preferences."
)

st.sidebar.divider()

st.sidebar.header("Your Preferences")


# =========================================================
# STATE
# =========================================================

states = sorted(
    df["STABBR"].dropna().unique()
)

selected_state = st.sidebar.selectbox(
    "Preferred State",
    ["Any"] + states
)


# =========================================================
# COLLEGE TYPE
# =========================================================

college_type = st.sidebar.selectbox(
    "College Type",
    [
        "Any",
        "Public",
        "Private"
    ]
)


# =========================================================
# MAXIMUM TUITION
# =========================================================

max_tuition = st.sidebar.number_input(
    "Maximum Tuition ($)",
    min_value=0,
    max_value=100000,
    value=20000,
    step=1000
)


# =========================================================
# MINIMUM SAT
# =========================================================

min_sat = st.sidebar.number_input(
    "Minimum SAT",
    min_value=400,
    max_value=1600,
    value=1100,
    step=10
)


# =========================================================
# MINIMUM ACT
# =========================================================

min_act = st.sidebar.number_input(
    "Minimum ACT",
    min_value=1,
    max_value=36,
    value=22,
    step=1
)


# =========================================================
# COLLEGE SIZE
# =========================================================

college_size = st.sidebar.selectbox(
    "Preferred College Size",
    [
        "Any",
        "Small",
        "Medium",
        "Large"
    ]
)


# =========================================================
# NUMBER OF RESULTS
# =========================================================

number_of_results = st.sidebar.slider(
    "Number of Colleges",
    min_value=5,
    max_value=30,
    value=10
)


st.sidebar.divider()


find_colleges = st.sidebar.button(
    "🔎 Find My Colleges",
    use_container_width=True
)


# =========================================================
# MODEL PREDICTION FUNCTION
# =========================================================

def get_model_predictions(data):

    # Copy college data
    model_data = data.copy()

    # -----------------------------------------------------
    # Get features expected by the trained model
    # -----------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        model_features = list(model.feature_names_in_)

    else:

        st.error(
            "The trained model does not expose its feature names. "
            "The model training features must match the data supplied "
            "to the model."
        )

        st.stop()


    # -----------------------------------------------------
    # Check that required model features exist
    # -----------------------------------------------------

    missing_features = [
        col for col in model_features
        if col not in model_data.columns
    ]

    if missing_features:

        st.error(
            "The trained model expects these features, but they are "
            f"not available in college_match.csv: {missing_features}"
        )

        st.stop()


    # -----------------------------------------------------
    # Create model input
    # -----------------------------------------------------

    X = model_data[model_features].copy()


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    try:

        # Classification model
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X)

            # Probability of positive class
            if probabilities.shape[1] == 2:

                predictions = probabilities[:, 1]

            else:

                predictions = probabilities.max(axis=1)

        else:

            # Regression model
            predictions = model.predict(X)


    except Exception as e:

        st.error(
            "The trained model could not process the college data."
        )

        st.exception(e)

        st.stop()


    return predictions


# =========================================================
# RUN MATCHING
# =========================================================

if find_colleges:

    results = df.copy()


    # =====================================================
    # APPLY USER FILTERS
    # =====================================================

    if selected_state != "Any":

        results = results[
            results["STABBR"] == selected_state
        ]


    if college_type == "Public":

        results = results[
            results["CONTROL"] == 1
        ]

    elif college_type == "Private":

        results = results[
            results["CONTROL"] != 1
        ]


    # =====================================================
    # COLLEGE SIZE FILTER
    # =====================================================

    if college_size == "Small":

        results = results[
            results["UGDS"] < 5000
        ]

    elif college_size == "Medium":

        results = results[
            (results["UGDS"] >= 5000)
            &
            (results["UGDS"] <= 15000)
        ]

    elif college_size == "Large":

        results = results[
            results["UGDS"] > 15000
        ]


    # =====================================================
    # TUITION FILTER
    # =====================================================

    results = results[
        results["TUITIONFEE_OUT"] <= max_tuition
    ]


    # =====================================================
    # SAT FILTER
    # =====================================================

    results = results[
        results["SAT_AVG"] >= min_sat
    ]


    # =====================================================
    # ACT FILTER
    # =====================================================

    results = results[
        results["ACTCMMID"] >= min_act
    ]


    # =====================================================
    # CHECK RESULTS
    # =====================================================

    if results.empty:

        st.warning(
            "No colleges were found with these preferences. "
            "Try changing your filters."
        )

        st.stop()


    # =====================================================
    # USE TRAINED MODEL
    # =====================================================

    predictions = get_model_predictions(results)


    results["Match Score"] = predictions


    # =====================================================
    # CONVERT SCORE TO PERCENTAGE
    # =====================================================

    min_score = results["Match Score"].min()
    max_score = results["Match Score"].max()


    if max_score != min_score:

        results["Match Score"] = (
            (results["Match Score"] - min_score)
            /
            (max_score - min_score)
            * 100
        )

    else:

        results["Match Score"] = 100


    # =====================================================
    # SORT
    # =====================================================

    results = results.sort_values(
        "Match Score",
        ascending=False
    )


    # =====================================================
    # TOP RESULTS
    # =====================================================

    results = results.head(
        number_of_results
    )


    # =====================================================
    # RESULTS
    # =====================================================

    st.header("🎓 Your College Matches")

    st.write(
        f"We found {len(results)} colleges "
        "based on your preferences."
    )


    # =====================================================
    # RESULTS TABLE
    # =====================================================

    display_df = results[
        [
            "INSTNM",
            "CITY",
            "STABBR",
            "CONTROL",
            "SAT_AVG",
            "ACTCMMID",
            "TUITIONFEE_OUT",
            "UGDS",
            "Match Score"
        ]
    ].copy()


    # College type

    display_df["CONTROL"] = display_df[
        "CONTROL"
    ].map({
        1: "Public",
        2: "Private nonprofit",
        3: "Private for-profit"
    })


    # SAT

    display_df["SAT_AVG"] = (
        display_df["SAT_AVG"]
        .round(0)
        .astype(int)
    )


    # ACT

    display_df["ACTCMMID"] = (
        display_df["ACTCMMID"]
        .round(0)
        .astype(int)
    )


    # Tuition

    display_df["TUITIONFEE_OUT"] = (
        display_df["TUITIONFEE_OUT"]
        .round(0)
        .apply(
            lambda x: f"${x:,.0f}"
        )
    )


    # Students

    display_df["UGDS"] = (
        display_df["UGDS"]
        .round(0)
        .apply(
            lambda x: f"{x:,.0f}"
        )
    )


    # Match score

    display_df["Match Score"] = (
        display_df["Match Score"]
        .round(1)
        .astype(str)
        + "%"
    )


    # Rename columns

    display_df.columns = [
        "College",
        "City",
        "State",
        "Type",
        "Average SAT",
        "Average ACT",
        "Tuition",
        "Students",
        "Match Score"
    ]


    # =====================================================
    # SHOW TABLE
    # =====================================================

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=450
    )


    # =====================================================
    # TOP MATCH
    # =====================================================

    st.divider()

    st.header("⭐ Top College Match")

    st.write(
        "The highest-ranked college according to "
        "the trained machine-learning model."
    )


    top = results.iloc[0]


    # =====================================================
    # TOP MATCH METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "College",
            top["INSTNM"]
        )


    with col2:

        st.metric(
            "Match Score",
            f"{top['Match Score']:.1f}%"
        )


    with col3:

        st.metric(
            "Average SAT",
            f"{top['SAT_AVG']:.0f}"
        )


    with col4:

        st.metric(
            "Tuition",
            f"${top['TUITIONFEE_OUT']:,.0f}"
        )


    # =====================================================
    # TOP MATCH DETAILS
    # =====================================================

    st.write(
        f"📍 **Location:** "
        f"{top['CITY']}, {top['STABBR']}"
    )

    st.write(
        f"🎓 **Average ACT:** "
        f"{top['ACTCMMID']:.0f}"
    )

    st.write(
        f"👥 **Undergraduate Students:** "
        f"{top['UGDS']:,.0f}"
    )


# =========================================================
# WELCOME SCREEN
# =========================================================

else:

    st.header("Find Your College Fit")

    st.write(
        "Use the preference panel on the left to "
        "find colleges that match your requirements."
    )


    st.subheader("How It Works")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.info(
            """
            **1. Choose Preferences**

            Select your state, college type,
            tuition budget, SAT, ACT,
            and college size.
            """
        )


    with col2:

        st.info(
            """
            **2. Machine Learning Model**

            The trained CollegeMatch AI model
            analyzes the available college data.
            """
        )


    with col3:

        st.info(
            """
            **3. Get Your Shortlist**

            Colleges are ranked according
            to the model's predictions.
            """
        )


    st.divider()

    st.write(
        "👈 Set your preferences in the sidebar "
        "and click **Find My Colleges**."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Developed By Habibulie 🔴 alias Leda🟢 | Data Science Student."
)
