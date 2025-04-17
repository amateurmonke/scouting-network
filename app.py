import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
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

# one-hot encoding for positions
for pos in positions:
    df[pos] = df["Pos"].apply(lambda x: 1 if pos in str(x).split(",") else 0)

features = ['Gls', 'Ast', 'G+A', 'PK', 'CrdY', 'CrdR', 'xG',
       'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR', 'Touches',
       'Def Pen', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Cmp_P', 'Att_P', 'SCA',
       'SCA90', 'GCA', 'Err', 'Tkl', 'TklW', 'Blocks', 'Int', 'Tkl+Int',
       'Clr'] + list(positions)

df = df.dropna(subset=features)

# Scale features
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# apply k-means clustering
k = 15
kmeans = KMeans(n_clusters=k, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# create a dictionary for player to cluster lookup
player_cluster_map = df.set_index("Player")["Cluster"].to_dict()

# tensorflow models for each cluster
cluster_models = {}
cluster_encoders = {}

for cluster_id in range(k):
    # Get players from this cluster
    cluster_mask = df["Cluster"] == cluster_id
    cluster_features = X_scaled[cluster_mask]
    
    if len(cluster_features) < 10: 
        continue
        
    # Create an autoencoder for this cluster
    input_dim = len(features)
    encoding_dim = 16 
    
    # Build the autoencoder
    inputs = tf.keras.Input(shape=(input_dim,))
    x = Dense(32, activation='relu')(inputs)
    bottleneck = Dense(encoding_dim, activation='relu')(x)
    x = Dense(32, activation='relu')(bottleneck)
    outputs = Dense(input_dim, activation='linear')(x)
    
    autoencoder = tf.keras.Model(inputs, outputs)
    encoder = tf.keras.Model(inputs, bottleneck)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(
        cluster_features, 
        cluster_features, 
        epochs=30, 
        batch_size=8, 
        verbose=0
    )
    
    cluster_encoders[cluster_id] = encoder

# finding similar players using neural embeddings
def similar_players_nn(player_name, top_n=5):
    if player_name not in player_cluster_map:
        return None
    
    cluster_id = player_cluster_map[player_name]
    
    cluster_mask = df["Cluster"] == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    cluster_players = df.iloc[cluster_indices]
    cluster_features = X_scaled[cluster_mask]
    
    if cluster_id not in cluster_encoders: # fallback if cluster has too few players
        similar_players = cluster_players[["Player", "Pos", "Squad", "Born"]]
        if len(similar_players) > top_n:
            similar_players = similar_players.sample(top_n)
        return similar_players
    
    # get player index
    player_idx_df = df[df["Player"] == player_name].index[0]
    player_idx_cluster = np.where(cluster_indices == player_idx_df)[0][0]
    
    # get embeddings for cluster players
    encoder = cluster_encoders[cluster_id]
    embeddings = encoder.predict(cluster_features)
    
    # calculate distances in embedding space
    player_embedding = embeddings[player_idx_cluster].reshape(1, -1)
    distances = np.linalg.norm(embeddings - player_embedding, axis=1)
    
    # sort by distance
    similar_indices = np.argsort(distances)
    # Exclude the player himself at index 0
    similar_indices = similar_indices[1:top_n+1]

    similar_players = cluster_players.iloc[similar_indices][["Player", "Pos", "Squad", "Born"]]
    return similar_players


st.title("Scouting Network")

player_name = st.text_input("Enter Player Name:")

if player_name:
    if player_name in player_cluster_map:
        player_info = df[df["Player"] == player_name][["Player", "Pos", "Squad", "Born"]].iloc[0]
    
        st.write(f"### Selected Player: {player_name}")
        st.write(f"Position: {player_info['Pos']} | Team: {player_info['Squad']} | Born: {player_info['Born'].astype(int)}")
        
        similar_players = similar_players_nn(player_name, top_n=5)
        
        st.write("### Similar Players")
        st.write(f"🔹 Here are 5 similar players to **{player_name}**:")
        
        if similar_players is not None and len(similar_players) > 0:
            similar_players["Born"] = similar_players["Born"].astype(int)
            st.table(similar_players)
        else:
            st.write("No similar players found in the database.")
    else:
        st.warning("Player not found! Ensure the spellings and accents are correct or try another player.")
