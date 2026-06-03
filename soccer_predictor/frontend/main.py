import streamlit as st
import requests

# Page UI configurations
st.set_page_config(page_title="AI Match Predictor", page_icon="⚽", layout="centered")

BACKEND_URL = "http://127.0.0.1:8000"

st.title("⚽ AI Soccer Score Predictor")
st.write("Powered by a mathematical prediction matrix analyzing 20 years of comprehensive historical match data.")
st.markdown("---")

# Fetch active teams from storage pool
@st.cache_data
def load_teams():
    try:
        response = requests.get(f"{BACKEND_URL}/teams")
        if response.status_code == 200:
            return response.json()["teams"]
    except:
        return []

teams = load_teams()

if not teams:
    st.error("Unable to establish communication with the AI prediction core. Verify your backend process is live on port 8000.")
else:
    # Build columns for layout design choice
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("Select Home Team", options=teams, index=0)
        
    with col2:
        # Default index set cleanly to secondary options to prevent matching selections instantly
        away_team = st.selectbox("Select Away Team", options=teams, index=min(1, len(teams)-1))

    if home_team == away_team:
        st.warning("Please choose distinct teams for the opposing home and away slots.")
    else:
        if st.button("Run Prediction Analysis", type="primary", use_container_width=True):
            with st.spinner("Processing tactical models against 20 years of history..."):
                payload = {"home_team": home_team, "away_team": away_team}
                res = requests.post(f"{BACKEND_URL}/predict", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
                    
                    # Display results visually inside clean dashboard modules
                    st.success(f"### Predicted Scoreline: {data['predicted_score']}")
                    
                    st.subheader("Match Outcome Distributions")
                    c1, c2, c3 = st.columns(3)
                    c1.metric(label=f"{home_team} Win", value=f"{data['probabilities']['home_win']}%")
                    c2.metric(label="Draw Chance", value=f"{data['probabilities']['draw']}%")
                    c3.metric(label=f"{away_team} Win", value=f"{data['probabilities']['away_win']}%")
                    
                    # Expected Metric values broken out explicitly
                    with st.expander("View Underlying Tactical Analytics"):
                        st.write(f"**Expected {home_team} Home Goals:** {data['expected_goals']['home']}")
                        st.write(f"**Expected {away_team} Away Goals:** {data['expected_goals']['away']}")
                else:
                    st.error("Error computing match trends. Please retry.")