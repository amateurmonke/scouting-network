import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

df = pd.read_csv("data.csv")

if "Player" not in df.columns:
    st.error("Error: 'Player' column not found in dataset!")
    st.stop()

df = df.drop(columns=["Unnamed: 0_level_0", "Unnamed: 36_level_0"], errors="ignore")

positions = set()
for pos in df["Pos"].dropna():
    positions.update(pos.split(","))
unique_positions = sorted(positions)  

for i in positions:
    df[i] = df["Pos"].apply(lambda x: 1 if pos in str(x).split(",") else 0)

features = ['Gls', 'Ast', 'G+A', 'PK', 'CrdY', 'CrdR', 'xG',
       'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR', 'Touches',
       'Def Pen', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Cmp_P', 'Att_P', 'SCA',
       'SCA90', 'GCA', 'Err', 'Tkl', 'TklW', 'Blocks', 'Int', 'Tkl+Int',
       'Clr'] + list(positions)
df = df.dropna(subset=features)

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 15
kmeans = KMeans(n_clusters=k, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

model = Sequential([
    Dense(64, activation="relu", input_shape=(len(features),)),
    Dense(32, activation="relu"),
    Dense(k, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

y = df["Cluster"].astype(int)

model.fit(X_scaled, y, epochs=10, batch_size=8, verbose=1)


player_cluster_map = df.set_index("Player")["Cluster"].to_dict()

st.title("Scouting Network")

player_name = st.text_input("Enter Player Name:")

if player_name in player_cluster_map:
    cluster_id = player_cluster_map[player_name]
    similar_players = df[df["Cluster"] == cluster_id][["Player", "Pos", "Squad", "Born"]]

    if len(similar_players) >= 5:
        similar_players = similar_players.sample(5)
    
    st.write(f"🔹 Here are 5 similar players to **{player_name}**:")
    st.table(similar_players)
else:
    st.warning("Player not found! Try another name.")
