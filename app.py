
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
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return joblib.load("college_match_model.pkl")


df = load_data()


# =========================================================
# REQUIRED COLUMNS
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
    col
    for col in required_columns
    if col not in df.columns
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

st.subheader("Find colleges that fit your preferences.")

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

st.sidebar.write("Set your college preferences.")

st.sidebar.divider()

st.sidebar.header("Your Preferences")


# =========================================================
# STATE
# =========================================================

states = sorted(
    df["STABBR"].unique()
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
# MATCHING FUNCTION
# =========================================================

def calculate_match_score(row):

    scores = []

    # -----------------------------------------------------
    # STATE
    # -----------------------------------------------------

    if selected_state == "Any":

        state_score = 100

    else:

        state_score = (
            100
            if row["STABBR"] == selected_state
            else 0
        )

    scores.append(state_score)


    # -----------------------------------------------------
    # COLLEGE TYPE
    # -----------------------------------------------------

    if college_type == "Any":

        type_score = 100

    elif college_type == "Public":

        type_score = (
            100
            if row["CONTROL"] == 1
            else 0
        )

    else:

        type_score = (
            100
            if row["CONTROL"] != 1
            else 0
        )

    scores.append(type_score)


    # -----------------------------------------------------
    # TUITION
    # -----------------------------------------------------

    if row["TUITIONFEE_OUT"] <= max_tuition:

        tuition_score = 100

    else:

        difference = (
            row["TUITIONFEE_OUT"]
            - max_tuition
        )

        tuition_score = max(
            0,
            100
            - (
                difference
                / max_tuition
                * 100
            )
        )

    scores.append(tuition_score)


    # -----------------------------------------------------
    # SAT
    # -----------------------------------------------------

    if row["SAT_AVG"] >= min_sat:

        sat_score = 100

    else:

        difference = (
            min_sat
            - row["SAT_AVG"]
        )

        sat_score = max(
            0,
            100
            - (
                difference
                / min_sat
                * 100
            )
        )

    scores.append(sat_score)


    # -----------------------------------------------------
    # ACT
    # -----------------------------------------------------

    if row["ACTCMMID"] >= min_act:

        act_score = 100

    else:

        difference = (
            min_act
            - row["ACTCMMID"]
        )

        act_score = max(
            0,
            100
            - (
                difference
                / min_act
                * 100
            )
        )

    scores.append(act_score)


    # -----------------------------------------------------
    # COLLEGE SIZE
    # -----------------------------------------------------

    if college_size == "Any":

        size_score = 100

    elif college_size == "Small":

        if row["UGDS"] < 5000:
            size_score = 100
        else:
            size_score = 0

    elif college_size == "Medium":

        if 5000 <= row["UGDS"] <= 15000:
            size_score = 100
        else:
            size_score = 0

    else:

        if row["UGDS"] > 15000:
            size_score = 100
        else:
            size_score = 0

    scores.append(size_score)


    # -----------------------------------------------------
    # FINAL SCORE
    # -----------------------------------------------------

    return sum(scores) / len(scores)


# =========================================================
# RUN MATCHING
# =========================================================

if find_colleges:

    results = df.copy()


    # -----------------------------------------------------
    # CALCULATE SCORE
    # -----------------------------------------------------

    results["Match Score"] = results.apply(
        calculate_match_score,
        axis=1
    )


    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    results = results.sort_values(
        "Match Score",
        ascending=False
    )


    # -----------------------------------------------------
    # TOP RESULTS
    # -----------------------------------------------------

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
        "your selected preferences."
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


    # =====================================================
    # HOW IT WORKS
    # =====================================================

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
            **2. Compare Colleges**

            The app compares each college
            with your selected preferences.
            """
        )


    with col3:

        st.info(
            """
            **3. Get Your Shortlist**

            The colleges are sorted by
            their match score.
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

