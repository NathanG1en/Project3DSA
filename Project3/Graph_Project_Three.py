import networkx as nx 
import pandas as pd 
import matplotlib as plt


olympic_df = pd.read_csv("Data/archive/athlete_events.csv")

# print(olympic_df)

G = nx.DiGraph()

# Add nodes from DataFrame columns
G.add_nodes_from(olympic_df['ID'])

# Add edges based on relationships (e.g., shared event)
for event, group in olympic_df.groupby('Event'):
    participants = group['ID'].tolist()
    for i in range(len(participants)):
        for j in range(i + 1, len(participants)):
            G.add_edge(participants[i], participants[j], event=event)


# Print basic information about the graph
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Print sample nodes and edges
print("\nSample nodes:", list(G.nodes)[:5])
print("Sample edges:", list(G.edges)[:5])

# plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_size=10, alpha=0.5, node_color='blue', edge_color='gray')
plt.title('Olympic Participants Network')
plt.show()






# G = nx.Graph() # Create an empty undirected graph (or nx.DiGraph() for a directed graph)
# # Add nodes from the 'source' and 'target' columns
# G.add_nodes_from(olympic_dfdf['source'])
# G.add_nodes_from(olympic_dfdf['target'])
# # Add edges from the DataFrame
# edges = [(row['source'], row['target']) for index, row in df.iterrows()]
# G.add_edges_from(edges)




