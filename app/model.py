import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)

data = pd.read_csv('c:/Users/fares/Desktop/Ai_Goru/trainIt/league-of-legends-predector/data/games.csv')
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

x = df.drop('winner', axis=1)
y = df['winner']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'trained_model.pkl')