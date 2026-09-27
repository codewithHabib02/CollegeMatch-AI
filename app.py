import streamlit as st 
import pandas as pd 
import joblib 
 
 
# ========================================================= 
# PAGE CONFIG 
# ========================================================= 
 
st.set_page_config( 
    page_title="CollegeMatch AI", 
    page_icon="🎓", 
    layout="wide", 
    initial_sidebar_state="expanded" 
) 
 
 
# ========================================================= 
# DARK AI THEME 
# ========================================================= 
 
st.markdown(""" 
<style> 
 
    /* Main application background */ 
    .stApp { 
        background: 
            radial-gradient( 
                circle at 10% 10%, 
                rgba(37, 99, 235, 0.20), 
                transparent 30% 
            ), 
            radial-gradient( 
                circle at 90% 15%, 
                rgba(139, 92, 246, 0.18), 
                transparent 30% 
            ), 
            radial-gradient( 
                circle at 50% 90%, 
                rgba(6, 182, 212, 0.10), 
                transparent 35% 
            ), 
            linear-gradient( 
                135deg, 
                #050B18 0%, 
                #081426 50%, 
                #0A1024 100% 
            ); 
 
        color: #F8FAFC; 
    } 
 
 
    /* ===================================================== 
       SIDEBAR 
       ===================================================== */ 
 
    section[data-testid="stSidebar"] { 
        background: 
            linear-gradient( 
                180deg, 
                #071225 0%, 
                #0A1830 100% 
            ); 
 
        border-right: 1px solid rgba(96, 165, 250, 0.15); 
    } 
 
 
    section[data-testid="stSidebar"] * { 
        color: #E2E8F0; 
    } 
 
 
    /* Sidebar headings */ 
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 { 
        color: #F8FAFC !important; 
    } 
 
 
    /* Sidebar selectboxes */ 
    section[data-testid="stSidebar"] 
    div[data-baseweb="select"] > div { 
 
        background-color: #14243D !important; 
 
        border: 1px solid #334E68 !important; 
 
        border-radius: 10px !important; 
 
        color: #F8FAFC !important; 
    } 
 
 
    /* Selected value inside selectbox */ 
    section[data-testid="stSidebar"] 
    div[data-baseweb="select"] span { 
 
        color: #F8FAFC !important; 
    } 
 
 
    /* Selectbox arrow */ 
    section[data-testid="stSidebar"] 
    div[data-baseweb="select"] svg { 
 
        fill: #67E8F9 !important; 
    } 
 
 
    /* Dropdown menu */ 
    div[data-baseweb="popover"] { 
 
        background-color: #10233D !important; 
    } 
 
 
    div[data-baseweb="popover"] ul { 
 
        background-color: #10233D !important; 
    } 
 
 
    div[data-baseweb="popover"] li { 
 
        color: #F8FAFC !important; 
        background-color: #10233D !important; 
    } 
 
 
    div[data-baseweb="popover"] li:hover { 
 
        background-color: #1E3A5F !important; 
        color: #FFFFFF !important; 
    } 
 
 
    /* Sidebar number inputs */ 
    section[data-testid="stSidebar"] 
    div[data-baseweb="input"] { 
 
        background-color: #14243D !important; 
 
        border: 1px solid #334E68 !important; 
 
        border-radius: 10px !important; 
    } 
 
 
    section[data-testid="stSidebar"] 
    input { 
 
        color: #F8FAFC !important; 
 
        background-color: #14243D !important; 
    } 
 
 
    /* Number input buttons */ 
    section[data-testid="stSidebar"] 
    button { 
 
        color: #67E8F9 !important; 
    } 
 
 
    /* Sidebar slider */ 
    section[data-testid="stSidebar"] 
    div[data-testid="stSlider"] { 
 
        color: #F8FAFC; 
    } 
 
 
    /* ===================================================== 
       MAIN TITLE 
       ===================================================== */ 
 
    .main-title { 
 
        font-size: 52px; 
 
        font-weight: 800; 
 
        text-align: center; 
 
        margin-top: 20px; 
 
        margin-bottom: 8px; 
 
        background: 
            linear-gradient( 
                90deg, 
                #60A5FA, 
                #22D3EE, 
                #A78BFA 
            ); 
 
        -webkit-background-clip: text; 
 
        -webkit-text-fill-color: transparent; 
    } 
 
 
    .subtitle { 
 
        text-align: center; 
 
        color: #94A3B8; 
 
        font-size: 18px; 
 
        margin-bottom: 35px; 
    } 
 
 
    /* ===================================================== 
       NORMAL STREAMLIT HEADINGS 
       ===================================================== */ 
 
    h1, h2, h3 { 
 
        color: #F8FAFC !important; 
    } 
 
 
    /* ===================================================== 
       RECOMMENDED COLLEGES TABLE
       ===================================================== */ 
 
    div[data-testid="stDataFrame"] { 
 
        background-color: #0F1F33 !important; 
 
        border: 1px solid #2B4C6F !important; 
 
        border-radius: 12px !important; 
 
        overflow: hidden !important; 
    } 
 
 
    div[data-testid="stDataFrame"] [role="gridcell"] { 
 
        background-color: #0F1F33 !important; 
 
        color: #E6EDF5 !important; 
    } 
 
 
    div[data-testid="stDataFrame"] [role="columnheader"] { 
 
        background-color: #173B5E !important; 
 
        color: #FFFFFF !important; 
 
        font-weight: 700 !important; 
    } 
 
 
    /* ===================================================== 
       TOP COLLEGE MATCH 
       ===================================================== */ 
 
    div[data-testid="stMetric"] { 
 
        background: linear-gradient( 
            135deg, 
            #172554, 
            #312E81 
        ) !important; 
 
        border: 1px solid #22D3EE !important; 
 
        border-radius: 15px !important; 
 
        padding: 18px !important; 
 
        box-shadow: 
            0 4px 15px rgba(34, 211, 238, 0.15); 
    } 
 
 
    /* Metric label */ 
    div[data-testid="stMetric"] label { 
 
        color: #BAE6FD !important; 
    } 
 
 
    /* Metric value */ 
    div[data-testid="stMetric"] 
    div[data-testid="stMetricValue"] { 
 
        color: #FFFFFF !important; 
    } 
 
 
    /* ===================================================== 
       PREDICTION BUTTON 
       ===================================================== */ 
 
    div.stButton > button { 
 
        background-color: #DC2626 !important; 
 
        color: white !important; 
 
        border: none !important; 
 
        border-radius: 10px !important; 
 
        font-size: 18px !important; 
 
        font-weight: 700 !important; 
 
        padding: 12px 24px !important; 
 
        width: 100% !important; 
    } 
 
 
    div.stButton > button:hover { 
 
        background-color: #B91C1C !important; 
 
        color: white !important; 
    } 
 
 
</style> 
""", unsafe_allow_html=True) 
 
 
# ========================================================= 
# LOAD DATA 
# ========================================================= 
 
@st.cache_data 
def load_data(): 
 
    return pd.read_csv("college_match.csv") 
 
 
# ========================================================= 
# LOAD MODEL 
# ========================================================= 
 
@st.cache_resource 
def load_model(): 
 
    return joblib.load("college_match_model.pkl") 
 
 
try: 
 
    df = load_data() 
 
    model = load_model() 
 
except Exception as e: 
 
    st.error("Could not load the college data or model.") 
 
    st.exception(e) 
 
    st.stop() 
 
 
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
    col for col in required_columns 
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
    subset=[ 
        "INSTNM", 
        "CITY", 
        "STABBR", 
        "SAT_AVG", 
        "ACTCMMID", 
        "TUITIONFEE_OUT", 
        "UGDS" 
    ] 
).copy() 
 
 
# ========================================================= 
# APP TITLE 
# ========================================================= 
 
st.markdown( 
    '<div class="main-title">🎓 CollegeMatch AI</div>', 
    unsafe_allow_html=True 
) 
 
st.markdown( 
    '<div class="subtitle">' 
    'Find colleges that match your academic and personal preferences ' 
    'using Machine Learning.' 
    '</div>', 
    unsafe_allow_html=True 
) 
 
 
# ========================================================= 
# SIDEBAR 
# ========================================================= 
 
st.sidebar.title("🎓 College Preferences") 
 
st.sidebar.write( 
    "Enter your requirements below to find matching colleges." 
) 
 
 
# --------------------------------------------------------- 
# Preferred State 
# --------------------------------------------------------- 
 
states = sorted( 
    df["STABBR"] 
    .dropna() 
    .astype(str) 
    .unique() 
    .tolist() 
) 
 
selected_state = st.sidebar.selectbox( 
    "Preferred State", 
    ["Any"] + states 
) 
 
 
# --------------------------------------------------------- 
# College Type 
# --------------------------------------------------------- 
 
selected_type = st.sidebar.selectbox( 
    "College Type", 
    ["Any", "Public", "Private"] 
) 
 
 
# --------------------------------------------------------- 
# Maximum Tuition 
# --------------------------------------------------------- 
 
max_tuition = st.sidebar.number_input( 
    "Maximum Tuition ($)", 
    min_value=0, 
    max_value=100000, 
    value=30000, 
    step=1000 
) 
 
 
# --------------------------------------------------------- 
# Minimum SAT 
# --------------------------------------------------------- 
 
min_sat = st.sidebar.number_input( 
    "Minimum SAT Score", 
    min_value=0, 
    max_value=1600, 
    value=1000, 
    step=10 
) 
 
 
# --------------------------------------------------------- 
# Minimum ACT 
# --------------------------------------------------------- 
 
min_act = st.sidebar.number_input( 
    "Minimum ACT Score", 
    min_value=0, 
    max_value=36, 
    value=20, 
    step=1 
) 
 
 
# --------------------------------------------------------- 
# College Size 
# --------------------------------------------------------- 
 
selected_size = st.sidebar.selectbox( 
    "Preferred College Size", 
    ["Any", "Small", "Medium", "Large"] 
) 
 
 
# --------------------------------------------------------- 
# Number of Colleges 
# --------------------------------------------------------- 
 
num_colleges = st.sidebar.slider( 
    "Number of Colleges", 
    min_value=5, 
    max_value=30, 
    value=10 
) 
 
 
# ========================================================= 
# CREATE MODEL FEATURES 
# ========================================================= 
 
model_df = df.copy() 
 
 
# --------------------------------------------------------- 
# Tuition Match 
# --------------------------------------------------------- 
 
model_df["tuition_match"] = ( 
    model_df["TUITIONFEE_OUT"] <= max_tuition 
).astype(int) 
 
 
# --------------------------------------------------------- 
# State Match 
# --------------------------------------------------------- 
 
if selected_state == "Any": 
 
    model_df["state_match"] = 1 
 
else: 
 
    model_df["state_match"] = ( 
        model_df["STABBR"] == selected_state 
    ).astype(int) 
 
 
# --------------------------------------------------------- 
# College Type Match 
# --------------------------------------------------------- 
 
if selected_type == "Any": 
 
    model_df["type_match"] = 1 
 
elif selected_type == "Public": 
 
    model_df["type_match"] = ( 
        model_df["CONTROL"] == 1 
    ).astype(int) 
 
else: 
 
    model_df["type_match"] = ( 
        model_df["CONTROL"] != 1 
    ).astype(int) 
 
 
# --------------------------------------------------------- 
# SAT Match 
# --------------------------------------------------------- 
 
model_df["sat_match"] = ( 
    model_df["SAT_AVG"] >= min_sat 
).astype(int) 
 
 
# --------------------------------------------------------- 
# ACT Match 
# --------------------------------------------------------- 
 
model_df["act_match"] = ( 
    model_df["ACTCMMID"] >= min_act 
).astype(int) 
 
 
# --------------------------------------------------------- 
# College Size Match 
# --------------------------------------------------------- 
 
if selected_size == "Any": 
 
    model_df["size_match"] = 1 
 
elif selected_size == "Small": 
 
    model_df["size_match"] = ( 
        model_df["UGDS"] < 5000 
    ).astype(int) 
 
elif selected_size == "Medium": 
 
    model_df["size_match"] = ( 
        (model_df["UGDS"] >= 5000) & 
        (model_df["UGDS"] <= 15000) 
    ).astype(int) 
 
else: 
 
    model_df["size_match"] = ( 
        model_df["UGDS"] > 15000 
    ).astype(int) 
 
 
# ========================================================= 
# MODEL INPUT 
# ========================================================= 
 
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
 
 
X_model = model_df[model_columns] 
 
 
# ========================================================= 
# PREDICTION BUTTON 
# ========================================================= 
 
st.markdown("### 🎯 Ready to Find Your College Match?") 
 
predict_button = st.button( 
    "🔍 Predict College Matches", 
    use_container_width=True 
) 
 
 
# ========================================================= 
# PREDICTION 
# ========================================================= 
 
try: 
 
    if hasattr(model, "predict_proba"): 
 
        probabilities = model.predict_proba(X_model) 
 
        if probabilities.shape[1] == 2: 
 
            predictions = probabilities[:, 1] 
 
        else: 
 
            predictions = probabilities.max(axis=1) 
 
    else: 
 
        predictions = model.predict(X_model) 
 
except Exception as e: 
 
    st.error("The model could not make predictions.") 
 
    st.exception(e) 
 
    st.stop() 
 
 
# ========================================================= 
# MATCH SCORE 
# ========================================================= 
 
model_df["prediction"] = predictions 
 
 
prediction_min = model_df["prediction"].min() 
 
prediction_max = model_df["prediction"].max() 
 
 
if prediction_max != prediction_min: 
 
    model_df["Match Score"] = ( 
 
        (model_df["prediction"] - prediction_min) 
 
        / 
 
        (prediction_max - prediction_min) 
 
        * 100 
 
    ) 
 
else: 
 
    model_df["Match Score"] = 100 
 
 
# ========================================================= 
# SORT RESULTS 
# ========================================================= 
 
results = model_df.sort_values( 
    by="Match Score", 
    ascending=False 
).head(num_colleges) 
 
 
# ========================================================= 
# RESULTS 
# ========================================================= 
 
st.header("🔎 Recommended Colleges") 
 
 
display_columns = [ 
    "INSTNM", 
    "CITY", 
    "STABBR", 
    "SAT_AVG", 
    "ACTCMMID", 
    "TUITIONFEE_OUT", 
    "UGDS", 
    "Match Score" 
] 
 
 
display_df = results[display_columns].copy() 
 
 
display_df["Match Score"] = ( 
    display_df["Match Score"].round(1) 
) 
 
 
display_df = display_df.rename( 
    columns={ 
        "INSTNM": "College", 
        "CITY": "City", 
        "STABBR": "State", 
        "SAT_AVG": "SAT", 
        "ACTCMMID": "ACT", 
        "TUITIONFEE_OUT": "Tuition", 
        "UGDS": "Students" 
    } 
) 
 
 
st.dataframe( 
    display_df, 
    use_container_width=True, 
    hide_index=True 
) 
 
 
# ========================================================= 
# TOP MATCH 
# ========================================================= 
 
if len(results) > 0: 
 
    top = results.iloc[0] 
 
    st.subheader("🏆 Top College Match") 
 
    col1, col2 = st.columns(2) 
 
    with col1: 
 
        st.metric( 
            "College", 
            top["INSTNM"] 
        ) 
 
        st.write( 
            f"📍 {top['CITY']}, {top['STABBR']}" 
        ) 
 
    with col2: 
 
        st.metric( 
            "AI Match Score", 
            f"{top['Match Score']:.1f}%" 
        ) 
 
 
# ========================================================= 
# MATCH SUMMARY 
# ========================================================= 
 
st.subheader("📊 Match Summary") 
 
col1, col2, col3, col4 = st.columns(4) 
 
 
with col1: 
 
    st.metric( 
        "Colleges Shown", 
        len(results) 
    ) 
 
 
with col2: 
 
    st.metric( 
        "Average Match", 
        f"{results['Match Score'].mean():.1f}%" 
    ) 
 
 
with col3: 
 
    st.metric( 
        "Tuition Limit", 
        f"${max_tuition:,}" 
    ) 
 
 
with col4: 
 
    st.metric( 
        "Minimum SAT", 
        min_sat 
    ) 
 
 
# ========================================================= 
# FOOTER 
# ========================================================= 
 
st.caption( 
    "🎓 CollegeMatch AI •Developed By • Habibulie • alias Leda• Data Science Student•" 
    
)
