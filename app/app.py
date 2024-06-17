import streamlit as st
import joblib
import json

def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)

model = joblib.load('trained_model.pkl')

champion_data:list = load_json('c:/Users/fares/Desktop/Ai_Goru/trainIt/league-of-legends-predector/data/champion_info.json')['data']

st.title('League of Legends Predictor')
st.write('This is a simple web app to predict the winner of a game of League of Legends')

team1 = []
team2 = []
champion_names = [champion_data[str(champion_id)]['name'] for champion_id in champion_data]

col1,col2 = st.columns(2, gap='large')

with col1:
    st.header('Team 1')
    for i in range(1,6):
        team1.append(
            st.selectbox(
                f'Champion {i}',
                options=champion_names,
                key=f'team1_champion_{i}'
                )
        )

with col2:
    st.header('Team 2')
    for i in range(1,6):
        team2.append(
            st.selectbox(
                f'Champion {i}',
                options=champion_names,
                key=f'team2_champion_{i}'
                )   
            )

button_text = 'Predict'

if  st.markdown(f"""<span id="custom-button">{st.button(button_text)}</span>""", unsafe_allow_html=True):
    X = [ ]
    for i in range(5):
        X.append(int(list(champion_data.keys())[list(champion_data.values()).index(team1[i])]))
    for i in range(5):
        X.append(int(list(champion_data.keys())[list(champion_data.values()).index(team2[i])]))
    prob = model.predict_proba([X])
    st.write(f'Team1: {prob[0][0]*100:.2f}% | Team2: {prob[0][1]*100:.2f}%')