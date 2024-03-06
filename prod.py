import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)
data = pd.read_csv(
    'c:/Users/fares/Desktop/Ai_Goru/trainIt/league-of-legends-predector/data/games.csv'
)
champion_data:list = load_json('c:/Users/fares/Desktop/Ai_Goru/trainIt/league-of-legends-predector/data/champion_info.json')['data']


df  = data[[
    't1_champ1id', 
    't1_champ2id',
    't1_champ3id',
    't1_champ4id',
    't1_champ5id',
    't2_champ1id', 
    't2_champ2id',
    't2_champ3id',
    't2_champ4id',
    't2_champ5id',
    'winner'
]] 

# for k in range(50):
#     team1 =[]
#     team2 =[]
#     for i in range(1,6):
#         access1 = f't1_champ{i}id'
#         access2 = f't2_champ{i}id'
#         team1.append(champion_data[str(data.iloc[k][access1])]['name'])
#         team2.append(champion_data[str(data.iloc[k][access2])]['name'])
#     print(f'Game[{k}]: {data.iloc[k]["gameId"]}')
#     print(f'Team1: {team1}')
#     print(f'Team2: {team2}')






x = df.drop('winner', axis=1)
y = df['winner']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = RandomForestClassifier()
model.fit(X_train, y_train)



st.title('League of Legends Predictor')

st.write('This is a simple web app to predict the winner of a game of League of Legends')


team1 = []
team2 = []
champion_names = [champion_data[str(champion_id)]['name'] for champion_id in champion_data]


col1,col2 = st.columns(2,
                       gap='large')


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
        
if st.button('Predict'):
    X = [ ]
    for i in range(5):
        X.append(int(list(champion_data.keys())[list(champion_data.values()).index(team1[i])]))
    for i in range(5):
        X.append(int(list(champion_data.keys())[list(champion_data.values()).index(team2[i])]))
    prob = model.predict_proba([X])
    st.write(f'Team1: {prob[0][0]*100:.2f}% | Team2: {prob[0][1]*100:.2f}%')



