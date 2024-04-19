import networkx as nx
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.backends.backend_agg as agg
import time

def Weight_to_Sport_Graph():
    olympic_df = pd.read_csv("Project3/Data/archive/athlete_events.csv")
    sorted_df =  olympic_df.sort_values(by='Year', ascending=False) # sorting by most recent year
    cleaned_df = sorted_df.drop_duplicates(subset='Name', keep='first') # cleaning up df by removing duplicate names 

    # creating a function that creates a weight range 
    def weight_range(weight, range_size=5, min_weight=0):
        lower_bound = min_weight
        upper_bound = min_weight + range_size
        
        while weight >= upper_bound:
            lower_bound = upper_bound
            upper_bound += range_size
        
        return f"{lower_bound}-{upper_bound}"
    
    # partitioning the data to make the graph 
    Weight_df = cleaned_df
    Weight_df['Weight_Range'] = cleaned_df['Weight'].apply(weight_range, range_size=5, min_weight=cleaned_df['Weight'].min())

    columns_to_keep = ['Name', 'Sex', 'Weight', 'Weight_Range', 'Sport']
    Weight_df = Weight_df[columns_to_keep]  
    Weight_df = Weight_df.dropna()


    Weight_df_male = Weight_df[Weight_df["Sex"] == "M"]
    Weight_df_female = Weight_df[Weight_df["Sex"] == "F"]

    # Filter out rows with NaN values in 'Weight_Range' and 'Sport' columns
    Weight_df_male = Weight_df_male.dropna(subset=['Weight_Range', 'Sport'])

    # Create an undirected graph
    G = nx.Graph()

    # Add nodes based on Weight_Range


    weight_ranges = Weight_df_male['Weight_Range'].unique()
    G.add_nodes_from(weight_ranges)


    # Iterate over a set of unique sports
    for sport in Weight_df_male['Sport'].unique():
        # create a filtered DataFrame for rows with the current sport
        sport_df = Weight_df_male[Weight_df_male['Sport'] == sport]
        #   unique Weight_Ranges for the current sport
        sport_weight_ranges = sport_df['Weight_Range'].unique()
        # Add edges between nodes with the same sport (basically just add edges between all the weight ranges present)
        for i in range(len(sport_weight_ranges)):
            for j in range(i+1, len(sport_weight_ranges)):
                G.add_edge(sport_weight_ranges[i], sport_weight_ranges[j])
    
    return G




def test1():
    # Specify the number of rows to read as a sample
    sample_size = 1000  # Adjust the sample size as needed

    # olympic_df = pd.read_csv("C:\\Users\\Eric Brown\\PycharmProjects\\Project3DSA\\Project3\\Data\\archive\\athlete_events.csv", nrows=sample_size)
    olympic_df = pd.read_csv("./Project3/Data/archive/cleaned_data.csv")

    G = nx.DiGraph()

    G.add_nodes_from(olympic_df['ID'])

    for event, group in olympic_df.groupby('Event'):
        participants = group['ID'].tolist()

        # print(event)
        # print("group")
        # print(group)
        # print("participants")
        # print(participants)
        # print(len(participants))
        # print("\n")

        for i in range(len(participants)):
            for j in range(i + 1, len(participants)):
                G.add_edge(participants[i], participants[j], event=event)

    # Plot the graph using NetworkX
    plt.figure(figsize=(40, 20))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False, node_size=10, alpha=0.5, node_color='blue', edge_color='gray')
    plt.title('Olympic Participants Network')
    st.pyplot(plt)


def breadth_first_search(graph, start_node):
    visited = set() # this is just a bfs 
    queue = [start_node]
    visited_order = []
    
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            visited_order.append(node)
            queue.extend(graph.neighbors(node))
    
    return visited_order

def visualize_bfs(graph, start_node):
    visited_order = breadth_first_search(graph, start_node) # Calls the BFS algorithm
    pos = nx.spring_layout(graph)  # Sets how the nodes look 
    visted = []
    
    step_paceholder = st.empty()
    plot_placeholder = st.empty()  # Create a placeholder for the plot

    for i in range(len(visited_order)):
        current_node = visited_order[i]
        visted.append(current_node)

        step_paceholder.write(f"Step {i+1}: Visiting node {visited_order[i]}") # Step by step for now
        node_colors = ['maroon' if node == current_node else 'darkblue' if node in visted  else 'skyblue' for node in graph.nodes()]
        edge_colors = ['red' if edge in graph.edges(visited_order[i]) else 'gray' for edge in graph.edges()] # Red if visited, gray otherwise
        
        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw(graph, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=10, font_weight='bold', edge_color=edge_colors, width=0.5)
        
        # Display the plot
        plot_placeholder.pyplot(fig)
        time.sleep(1)
        # Add a divider
        #st.write("---")

def test_bfs(): 
    # G = nx.Graph()
    # G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)])
    G = Weight_to_Sport_Graph()
    visualize_bfs(G, "25.0-30.0")


def depth_first_search(graph, start_node): 
    visited = set() # dfs 
    stack = [start_node]
    visited_order = []
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            visited_order.append(node)
            stack.extend(graph.neighbors(node))
    
    return visited_order

def visualize_dfs(graph, start_node):
    visited_order = depth_first_search(graph, start_node) # Calls the BFS algorithm
    pos = nx.spring_layout(graph)  # Sets how the nodes look 
    visted = []
    
    step_paceholder = st.empty()
    plot_placeholder = st.empty()  # Create a placeholder for the plot

    for i in range(len(visited_order)):
        current_node = visited_order[i]
        visted.append(current_node)

        step_paceholder.write(f"Step {i+1}: Visiting node {visited_order[i]}") # Step by step for now
        node_colors = ['maroon' if node == current_node else 'darkblue' if node in visted  else 'skyblue' for node in graph.nodes()]
        edge_colors = ['red' if edge in graph.edges(visited_order[i]) else 'gray' for edge in graph.edges()] # Red if visited, gray otherwise
        
        # Create the figure and axis objects
        fig, ax = plt.subplots()
        nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=700, edge_color=edge_colors, width=2.0, edge_cmap=plt.cm.Blues, ax=ax)
        
        # Display the plot
        plot_placeholder.pyplot(fig)
        time.sleep(1)
        # Add a divider
        #st.write("---")

def test_dfs(): 
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)])
    visualize_dfs(G, 0)

# st.title("Welcome to The Olympic's Analyzer!")

st.title("Welcome to The Olympics Analyzer!") # Title for the webapp

options1 = ['Weight Graph', 'Height Graph', 'Age Graph'] # options to choose from for graph(s)
options2 = ['Sport', 'Medal']
selected_option1 = st.selectbox('Select an option:', options1, index=None) # if nothing is selected, we don't want to waste resources on making a graph
selected_option2 = st.selectbox('Select an option:', options2, index=None)
result = st.button("Search", type="secondary")
cancel = st.button("Cancel", type="primary")

if selected_option1 == 'Weight Graph':
    if selected_option2 == 'Sport':
        if result == True and cancel == False:
            st.write('Making a Weight-Sport Graph... Please be patient...')
            # create a graph that shows the connectedness of weight vs the sport, separate men/women?
            test_bfs()



    if selected_option2 == 'Medal':
        if result == True and cancel == False:
            st.write('Making a Weight-Medal Graph... Please be patient...')
            # create a graph that shows the connectedness of weight vs the medals, separate men/women?
            test_dfs()

elif selected_option1 == 'Height Graph':
    if selected_option2 == 'Sport':
        if result == True and cancel == False:
            st.write('Making a Height-Sport Graph... Please be patient...')
            # create a graph that shows the connectedness of height vs the sport, separate men/women?
            pass
    if selected_option2 == 'Medal':
        if result == True and cancel == False:
            st.write('Making a Height-Medal Graph... Please be patient...')
            # create a graph that shows the connectedness of height vs the medal, separate men/women?
            pass

elif selected_option1 == 'Age Graph':
    if selected_option2 == 'Sport':
        if result == True and cancel == False:
            st.write('Making an Age-Sport Graph... Please be patient...')
            # create a graph that shows the connectedness of age vs the sport
            pass
    if selected_option2 == 'Medal':
        if result == True and cancel == False:
            st.write('Making an Age-Medal Graph... Please be patient...')
            # create a graph that shows the connectedness of age vs the medal
            pass




# G = nx.Graph() # Create an empty undirected graph (or nx.DiGraph() for a directed graph)
# # Add nodes from the 'source' and 'target' columns
# G.add_nodes_from(olympic_dfdf['source'])
# G.add_nodes_from(olympic_dfdf['target'])
# # Add edges from the DataFrame
# edges = [(row['source'], row['target']) for index, row in df.iterrows()]
# G.add_edges_from(edges)




