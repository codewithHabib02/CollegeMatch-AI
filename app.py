
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
# BACKGROUND
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
# LOAD COLLEGE DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("college_match.csv")


try:
    df = load_data()
except Exception as e:
    st.error("Could not load college_match.csv")
    st.exception(e)
    st.stop()


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("college_match_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error("Could not load college_match_model.pkl")
    st.exception(e)
    st.stop()


# =========================================================
# REQUIRED COLLEGE DATA COLUMNS
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

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"The following required columns are missing from "
        f"college_match.csv: {missing_columns}"
    )
    st.stop()


# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna(
    subset=[
        "INSTNM",
        "CITY",
        "STABBR",
        "CONTROL",
        "SAT_AVG",
        "ACTCMMID",
        "TUITIONFEE_OUT",
        "UGDS"
    ]
).copy()


# =========================================================
# TITLE
# =========================================================

st.title("🎓 CollegeMatch AI")

st.subheader(
    "Find colleges that match your preferences using Machine Learning."
)

st.write(
    "Enter your college preferences in the sidebar and CollegeMatch AI "
    "will analyze the available colleges and provide a ranked shortlist."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🎯 Your College Preferences")


# State
states = sorted(
    df["STABBR"].astype(str).dropna().unique()
)

selected_state = st.sidebar.selectbox(
    "Preferred State",
    ["Any"] + states
)


# College type
college_type = st.sidebar.selectbox(
    "College Type",
    ["Any", "Public", "Private"]
)


# Tuition
max_tuition = st.sidebar.number_input(
    "Maximum Tuition ($)",
    min_value=0,
    value=30000,
    step=1000
)


# SAT
min_sat = st.sidebar.number_input(
    "Minimum SAT Score",
    min_value=0,
    max_value=1600,
    value=1000,
    step=10
)


# ACT
min_act = st.sidebar.number_input(
    "Minimum ACT Score",
    min_value=0,
    max_value=36,
    value=20,
    step=1
)


# College size
college_size = st.sidebar.selectbox(
    "Preferred College Size",
    ["Any", "Small", "Medium", "Large"]
)


# Number of results
number_of_colleges = st.sidebar.slider(
    "Number of Colleges",
    min_value=5,
    max_value=30,
    value=10
)


# =========================================================
# CREATE MODEL FEATURES
# =========================================================

def prepare_model_data(data):
    """
    Create the exact features expected by the trained model.
    """

    model_data = data.copy()

    # -----------------------------------------------------
    # TUITION MATCH
    # -----------------------------------------------------

    model_data["tuition_match"] = (
        model_data["TUITIONFEE_OUT"] <= max_tuition
    ).astype(int)


    # -----------------------------------------------------
    # STATE MATCH
    # -----------------------------------------------------

    if selected_state == "Any":
        model_data["state_match"] = 1

    else:
        model_data["state_match"] = (
            model_data["STABBR"] == selected_state
        ).astype(int)


    # -----------------------------------------------------
    # TYPE MATCH
    # -----------------------------------------------------

    if college_type == "Any":

        model_data["type_match"] = 1

    elif college_type == "Public":

        model_data["type_match"] = (
            model_data["CONTROL"] == 1
        ).astype(int)

    else:

        model_data["type_match"] = (
            model_data["CONTROL"] != 1
        ).astype(int)


    # -----------------------------------------------------
    # SAT MATCH
    # -----------------------------------------------------

    model_data["sat_match"] = (
        model_data["SAT_AVG"] >= min_sat
    ).astype(int)


    # -----------------------------------------------------
    # ACT MATCH
    # -----------------------------------------------------

    model_data["act_match"] = (
        model_data["ACTCMMID"] >= min_act
    ).astype(int)


    # -----------------------------------------------------
    # SIZE MATCH
    # -----------------------------------------------------

    if college_size == "Any":

        model_data["size_match"] = 1

    elif college_size == "Small":

        model_data["size_match"] = (
            model_data["UGDS"] < 5000
        ).astype(int)

    elif college_size == "Medium":

        model_data["size_match"] = (
            (model_data["UGDS"] >= 5000) &
            (model_data["UGDS"] <= 15000)
        ).astype(int)

    else:

        model_data["size_match"] = (
            model_data["UGDS"] > 15000
        ).astype(int)


    # -----------------------------------------------------
    # EXACT MODEL COLUMNS
    # -----------------------------------------------------

    model_columns = [
        "INSTNM",
        "CITY",
        "tuition_match",
        "state_match",
        "type_match",
        "sat_match",
        "act_match",
        "size_match"
    ]

    return model_data[model_columns]


# =========================================================
# FIND COLLEGES
# =========================================================

if st.button("🔎 Find My Colleges", type="primary"):

    results = df.copy()


    # -----------------------------------------------------
    # FILTER BY STATE
    # -----------------------------------------------------

    if selected_state != "Any":

        results = results[
            results["STABBR"] == selected_state
        ]


    # -----------------------------------------------------
    # FILTER BY COLLEGE TYPE
    # -----------------------------------------------------

    if college_type == "Public":

        results = results[
            results["CONTROL"] == 1
        ]

    elif college_type == "Private":

        results = results[
            results["CONTROL"] != 1
        ]


    # -----------------------------------------------------
    # CHECK RESULTS
    # -----------------------------------------------------

    if results.empty:

        st.warning(
            "No colleges were found with these preferences. "
            "Try changing your filters."
        )

        st.stop()


    # -----------------------------------------------------
    # CREATE MODEL INPUT
    # -----------------------------------------------------

    X_model = prepare_model_data(results)


    # -----------------------------------------------------
    # VERIFY MODEL FEATURES
    # -----------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        expected_features = list(
            model.feature_names_in_
        )

        missing_features = [
            column
            for column in expected_features
            if column not in X_model.columns
        ]

        if missing_features:

            st.error(
                f"The model still expects these missing columns: "
                f"{missing_features}"
            )

            st.write(
                "Model expects:"
            )

            st.write(expected_features)

            st.write(
                "App is providing:"
            )

            st.write(list(X_model.columns))

            st.stop()


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    try:

        # Classification model
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X_model)

            if probabilities.shape[1] == 2:

                predictions = probabilities[:, 1]

            else:

                predictions = probabilities.max(axis=1)

        else:

            predictions = model.predict(X_model)

    except Exception as e:

        st.error(
            "The trained model could not process these inputs."
        )

        st.code(str(e))

        st.write(
            "Model expects:"
        )

        if hasattr(model, "feature_names_in_"):
            st.write(
                list(model.feature_names_in_)
            )

        st.write(
            "App provided:"
        )

        st.write(
            list(X_model.columns)
        )

        st.stop()


    # -----------------------------------------------------
    # ADD PREDICTIONS
    # -----------------------------------------------------

    results["Match Score"] = predictions


    # -----------------------------------------------------
    # CONVERT TO DISPLAY SCORE
    # -----------------------------------------------------

    if pd.api.types.is_numeric_dtype(
        results["Match Score"]
    ):

        min_prediction = results["Match Score"].min()

        max_prediction = results["Match Score"].max()

        if max_prediction != min_prediction:

            results["Match Score"] = (
                (
                    results["Match Score"]
                    - min_prediction
                )
                /
                (
                    max_prediction
                    - min_prediction
                )
                * 100
            )

        else:

            results["Match Score"] = 100


    # -----------------------------------------------------
    # SORT RESULTS
    # -----------------------------------------------------

    results = results.sort_values(
        by="Match Score",
        ascending=False
    )


    # -----------------------------------------------------
    # LIMIT RESULTS
    # -----------------------------------------------------

    results = results.head(
        number_of_colleges
    )


    # =====================================================
    # RESULTS
    # =====================================================

    st.success(
        f"Found {len(results)} colleges matching your preferences."
    )

    st.subheader("🏆 Your College Matches")


    # -----------------------------------------------------
    # TOP MATCH
    # -----------------------------------------------------

    top_college = results.iloc[0]

    st.markdown(
        f"""
        ### 🎓 {top_college["INSTNM"]}

        **Location:** {top_college["CITY"]}, {top_college["STABBR"]}

        **Match Score:** {top_college["Match Score"]:.1f}%
        """
    )


    # -----------------------------------------------------
    # RESULTS TABLE
    # -----------------------------------------------------

    display_results = results[
        [
            "INSTNM",
            "CITY",
            "STABBR",
            "SAT_AVG",
            "ACTCMMID",
            "TUITIONFEE_OUT",
            "UGDS",
            "Match Score"
        ]
    ].copy()


    display_results = display_results.rename(
        columns={
            "INSTNM": "College",
            "CITY": "City",
            "STABBR": "State",
            "SAT_AVG": "SAT",
            "ACTCMMID": "ACT",
            "TUITIONFEE_OUT": "Tuition",
            "UGDS": "Enrollment"
        }
    )


    display_results["Match Score"] = (
        display_results["Match Score"]
        .round(1)
    )


    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # COLLEGE DETAILS
    # =====================================================

    st.subheader("📊 Top College Details")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "SAT",
            int(top_college["SAT_AVG"])
        )


    with col2:

        st.metric(
            "ACT",
            int(top_college["ACTCMMID"])
        )


    with col3:

        st.metric(
            "Tuition",
            f"${int(top_college['TUITIONFEE_OUT']):,}"
        )


    with col4:

        st.metric(
            "Enrollment",
            f"{int(top_college['UGDS']):,}"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Developed By Habibulie 🔴 alias Leda🟢 | Data Science Student."
)
