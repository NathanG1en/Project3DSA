import networkx as nx 
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt

st.write("Welcome to The Olympic's Analyzer!")

# Specify the number of rows to read as a sample
sample_size = 1000  # Adjust the sample size as needed

olympic_df = pd.read_csv("Project3/Data/archive/athlete_events.csv", nrows=sample_size)

G = nx.DiGraph()

G.add_nodes_from(olympic_df['ID'])

for event, group in olympic_df.groupby('Event'):
    participants = group['ID'].tolist()
    for i in range(len(participants)):
        for j in range(i + 1, len(participants)):
            G.add_edge(participants[i], participants[j], event=event)

# Plot the graph using NetworkX
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_size=10, alpha=0.5, node_color='blue', edge_color='gray')
plt.title('Olympic Participants Network')
st.pyplot(plt)



# G = nx.Graph() # Create an empty undirected graph (or nx.DiGraph() for a directed graph)
# # Add nodes from the 'source' and 'target' columns
# G.add_nodes_from(olympic_dfdf['source'])
# G.add_nodes_from(olympic_dfdf['target'])
# # Add edges from the DataFrame
# edges = [(row['source'], row['target']) for index, row in df.iterrows()]
# G.add_edges_from(edges)




