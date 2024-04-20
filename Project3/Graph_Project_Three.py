import networkx as nx
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.backends.backend_agg as agg
import time

def Age_to_Sport_Graph(gender):
    olympic_df = pd.read_csv("Project3/Data/archive/athlete_events.csv")
    sorted_df =  olympic_df.sort_values(by='Year', ascending=False) # sorting by most recent year
    cleaned_df = sorted_df.drop_duplicates(subset='Name', keep='first') # cleaning up df by removing duplicate names 

    # creating a function that creates an age range 
    def age_range(age, range_size=5, min_age=0):
        lower_bound = min_age
        upper_bound = min_age + range_size
        
        while age >= upper_bound:
            lower_bound = upper_bound
            upper_bound += range_size
        
        return f"{lower_bound}-{upper_bound}"
    
    # partitioning the data to make the graph 
    Age_df = cleaned_df
    Age_df['Age_Range'] = cleaned_df['Age'].apply(age_range, range_size=5, min_age=cleaned_df['Age'].min())

    columns_to_keep = ['Name', 'Sex', 'Age', 'Age_Range', 'Sport']
    Age_df = Age_df[columns_to_keep]  
    Age_df = Age_df.dropna()

    Age_df_gender = Age_df[Age_df["Sex"] == gender]

    # Filter out rows with NaN values in 'Age_Range' and 'Sport' columns
    Age_df_gender = Age_df_gender.dropna(subset=['Age_Range', 'Sport'])

    # Create an undirected graph
    G = nx.Graph()

    # Add nodes based on Age_Range
    age_ranges = Age_df_gender['Age_Range'].unique()
    G.add_nodes_from(age_ranges)

    # Iterate over a set of unique sports
    for sport in Age_df_gender['Sport'].unique():
        # create a filtered DataFrame for rows with the current sport
        sport_df = Age_df_gender[Age_df_gender['Sport'] == sport]
        #   unique Age_Ranges for the current sport
        sport_age_ranges = sport_df['Age_Range'].unique()
        # Add edges between nodes with the same sport (basically just add edges between all the age ranges present)
        for i in range(len(sport_age_ranges)):
            for j in range(i+1, len(sport_age_ranges)):
                G.add_edge(sport_age_ranges[i], sport_age_ranges[j])
    
    return G


def Height_to_Sport_Graph(gender):
    olympic_df = pd.read_csv("Project3/Data/archive/athlete_events.csv")
    sorted_df =  olympic_df.sort_values(by='Year', ascending=False) # sorting by most recent year
    cleaned_df = sorted_df.drop_duplicates(subset='Name', keep='first') # cleaning up df by removing duplicate names 

    # creating a function that creates a height range 
    def height_range(height, range_size=5, min_height=0):
        lower_bound = min_height
        upper_bound = min_height + range_size
        
        while height >= upper_bound:
            lower_bound = upper_bound
            upper_bound += range_size
        
        return f"{lower_bound}-{upper_bound}"
    
    # partitioning the data to make the graph 
    Height_df = cleaned_df
    Height_df['Height_Range'] = cleaned_df['Height'].apply(height_range, range_size=5, min_height=cleaned_df['Height'].min())

    columns_to_keep = ['Name', 'Sex', 'Height', 'Height_Range', 'Sport']
    Height_df = Height_df[columns_to_keep]  
    Height_df = Height_df.dropna()

    Height_df_gender = Height_df[Height_df["Sex"] == gender]

    # Filter out rows with NaN values in 'Height_Range' and 'Sport' columns
    Height_df_gender = Height_df_gender.dropna(subset=['Height_Range', 'Sport'])

    # Create an undirected graph
    G = nx.Graph()

    # Add nodes based on Height_Range
    height_ranges = Height_df_gender['Height_Range'].unique()
    G.add_nodes_from(height_ranges)

    # Iterate over a set of unique sports
    for sport in Height_df_gender['Sport'].unique():
        # create a filtered DataFrame for rows with the current sport
        sport_df = Height_df_gender[Height_df_gender['Sport'] == sport]
        #   unique Height_Ranges for the current sport
        sport_height_ranges = sport_df['Height_Range'].unique()
        # Add edges between nodes with the same sport (basically just add edges between all the height ranges present)
        for i in range(len(sport_height_ranges)):
            for j in range(i+1, len(sport_height_ranges)):
                G.add_edge(sport_height_ranges[i], sport_height_ranges[j])
    
    return G

def Weight_to_Sport_Graph(gender):
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


    Weight_df_gender = Weight_df[Weight_df["Sex"] == gender]

    # Filter out rows with NaN values in 'Weight_Range' and 'Sport' columns
    Weight_df_gender = Weight_df_gender.dropna(subset=['Weight_Range', 'Sport'])

    # Create an undirected graph
    G = nx.Graph()

    # Add nodes based on Weight_Range


    weight_ranges = Weight_df_gender['Weight_Range'].unique()
    G.add_nodes_from(weight_ranges)


    # Iterate over a set of unique sports
    for sport in Weight_df_gender['Sport'].unique():
        # create a filtered DataFrame for rows with the current sport
        sport_df = Weight_df_gender[Weight_df_gender['Sport'] == sport]
        #   unique Weight_Ranges for the current sport
        sport_weight_ranges = sport_df['Weight_Range'].unique()
        # Add edges between nodes with the same sport (basically just add edges between all the weight ranges present)
        for i in range(len(sport_weight_ranges)):
            for j in range(i+1, len(sport_weight_ranges)):
                G.add_edge(sport_weight_ranges[i], sport_weight_ranges[j])
    
    return G

def Weight_to_Medal_Graph(gender):
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

    columns_to_keep = ['Name', 'Sex', 'Weight', 'Weight_Range', 'Medal']
    Weight_df = Weight_df[columns_to_keep]  
    Weight_df = Weight_df.dropna()


    Weight_df_gender = Weight_df[Weight_df["Sex"] == gender]

    # Filter out rows with NaN values in 'Weight_Range' and 'Sport' columns
    Weight_df_gender = Weight_df_gender.dropna(subset=['Weight_Range', 'Medal'])

    # Create an undirected graph
    G = nx.Graph()

    # Add nodes based on Weight_Range


    weight_ranges = Weight_df_gender['Weight_Range'].unique()
    G.add_nodes_from(weight_ranges)


    # Iterate over a set of unique medal
    for medal in Weight_df_gender['Medal'].unique():
        # create a filtered DataFrame for rows with a medal
        medal_df = Weight_df_gender[Weight_df_gender['Medal'] == medal]
        #   unique Weight_Ranges for the current sport
        medal_weight_ranges = medal_df['Weight_Range'].unique()
        # Add edges between nodes with medals (basically just add edges between all the weight ranges present)
        for i in range(len(medal_weight_ranges)):
            for j in range(i+1, len(medal_weight_ranges)):
                G.add_edge(medal_weight_ranges[i], medal_weight_ranges[j])
    
    return G


def Age_to_Medal_Graph(gender):
    olympic_df = pd.read_csv("Project3/Data/archive/athlete_events.csv")
    sorted_df =  olympic_df.sort_values(by='Year', ascending=False) # sorting by most recent year
    cleaned_df = sorted_df.drop_duplicates(subset='Name', keep='first') # cleaning up df by removing duplicate names 

    # creating a function that creates an age range 
    def age_range(age, range_size=5, min_age=0):
        lower_bound = min_age
        upper_bound = min_age + range_size
        
        while age >= upper_bound:
            lower_bound = upper_bound
            upper_bound += range_size
        
        return f"{lower_bound}-{upper_bound}"
    
    # partitioning the data to make the graph 
    Age_df = cleaned_df
    Age_df['Age_Range'] = cleaned_df['Age'].apply(age_range, range_size=5, min_age=cleaned_df['Age'].min())

    columns_to_keep = ['Name', 'Sex', 'Age', 'Age_Range', 'Medal']
    Age_df = Age_df[columns_to_keep]  
    Age_df = Age_df.dropna()

    Age_df_gender = Age_df[Age_df["Sex"] == gender]

    # Filter out rows with NaN values in 'Age_Range' and 'Medal' columns
    Age_df_gender = Age_df_gender.dropna(subset=['Age_Range', 'Medal'])

    # Create an undirected graph
    G = nx.Graph()

    # Add nodes based on Age_Range
    age_ranges = Age_df_gender['Age_Range'].unique()
    G.add_nodes_from(age_ranges)

    # Iterate over a set of unique medals
    for medal in Age_df_gender['Medal'].unique():
        # create a filtered DataFrame for rows with a medal
        medal_df = Age_df_gender[Age_df_gender['Medal'] == medal]
        # unique Age_Ranges for the current medal
        medal_age_ranges = medal_df['Age_Range'].unique()
        # Add edges between nodes with medals (basically just add edges between all the age ranges present)
        for i in range(len(medal_age_ranges)):
            for j in range(i+1, len(medal_age_ranges)):
                G.add_edge(medal_age_ranges[i], medal_age_ranges[j])
    
    return G


def Height_to_Medal_Graph(gender):
    olympic_df = pd.read_csv("Project3/Data/archive/athlete_events.csv")
    sorted_df =  olympic_df.sort_values(by='Year', ascending=False) # sorting by most recent year
    cleaned_df = sorted_df.drop_duplicates(subset='Name', keep='first') # cleaning up df by removing duplicate names 

    # creating a function that creates a height range 
    def height_range(height, range_size=5, min_height=0):
        lower_bound = min_height
        upper_bound = min_height + range_size
        
        while height >= upper_bound:
            lower_bound = upper_bound
            upper_bound += range_size
        
        return f"{lower_bound}-{upper_bound}"
    
    # partitioning the data to make the graph 

    Height_df = cleaned_df
    Height_df['Height_Range'] = cleaned_df['Height'].apply(height_range, range_size=5, min_height=cleaned_df['Height'].min())

    columns_to_keep = ['Name', 'Sex', 'Height', 'Height_Range', 'Medal']
    Height_df = Height_df[columns_to_keep]  
    Height_df = Height_df.dropna()

    Height_df_gender = Height_df[Height_df["Sex"] == gender]

    # Filter out rows with NaN values in 'Height_Range' and 'Sport' columns
    Height_df_gender = Height_df_gender.dropna(subset=['Height_Range', 'Medal'])

    # Create an undirected graph
    G = nx.Graph()

    # Add nodes based on Height_Range
    height_ranges = Height_df_gender['Height_Range'].unique()
    G.add_nodes_from(height_ranges)

    # Iterate over a set of unique medal
    for medal in Height_df_gender['Medal'].unique():
        # create a filtered DataFrame for rows with a medal
        medal_df = Height_df_gender[Height_df_gender['Medal'] == medal]
        #   unique Height_Ranges for the current sport
        medal_height_ranges = medal_df['Height_Range'].unique()
        # Add edges between nodes with medals (basically just add edges between all the height ranges present)
        for i in range(len(medal_height_ranges)):
            for j in range(i+1, len(medal_height_ranges)):
                G.add_edge(medal_height_ranges[i], medal_height_ranges[j])
    
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

def visualize_bfs(graph):
    start_node = list(graph.nodes())[0]
    visited_order = breadth_first_search(graph, start_node) # Calls the BFS algorithm
    pos = nx.spring_layout(graph)  # Sets how the nodes look 
    visted = []
    
    step_paceholder = st.empty()
    plot_placeholder = st.empty()  # Create a placeholder for the plot

    for i in range(len(visited_order)):
        current_node = visited_order[i]
        visted.append(current_node)

        step_paceholder.write(f"BFS Step {i+1}: Visiting node {visited_order[i]}") # Step by step for now
        node_colors = ['maroon' if node == current_node else 'darkblue' if node in visted  else 'skyblue' for node in graph.nodes()]
        edge_colors = ['red' if edge in graph.edges(visited_order[i]) else 'gray' for edge in graph.edges()] # Red if visited, gray otherwise
        
        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw(graph, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=10, font_weight='bold', edge_color=edge_colors, width=0.5)
        
        # Display the plot
        plot_placeholder.pyplot(fig)
        time.sleep(.5)
        # Add a divider
        #st.write("---")

def test_bfs(): 
    # G = nx.Graph()
    # G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)])
    G = Weight_to_Sport_Graph(selected_gender)
    visualize_bfs(G)


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

def visualize_dfs(graph):
    start_node = list(graph.nodes())[0]
    visited_order = depth_first_search(graph, start_node) # Calls the BFS algorithm
    pos = nx.spring_layout(graph)  # Sets how the nodes look 
    visted = []
    
    step_paceholder = st.empty()
    plot_placeholder = st.empty()  # Create a placeholder for the plot

    for i in range(len(visited_order)):
        current_node = visited_order[i]
        visted.append(current_node)

        step_paceholder.write(f"DFS Step {i+1}: Visiting node {visited_order[i]}") # Step by step for now
        node_colors = ['maroon' if node == current_node else 'darkblue' if node in visted  else 'skyblue' for node in graph.nodes()]
        edge_colors = ['red' if edge in graph.edges(visited_order[i]) else 'gray' for edge in graph.edges()] # Red if visited, gray otherwise
        
        # Create the figure and axis objects
        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw(graph, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=10, font_weight='bold', edge_color=edge_colors, width=0.5)
        
        # Display the plot
        plot_placeholder.pyplot(fig)
        time.sleep(.5)
        # Add a divider
        #st.write("---")

def test_dfs(): 
    G = Weight_to_Medal_Graph(selected_gender)
    visualize_dfs(G)

# st.title("Welcome to The Olympic's Analyzer!")

st.title("Welcome to The Olympics Analyzer!") # Title for the webapp

options1 = ['Weight Graph', 'Height Graph', 'Age Graph'] # options to choose from for graph(s)
options2 = ['Sport', 'Medal']
options3 = ['Male', 'Female']
selected_option1 = st.selectbox('Select an option:', options1, index=None) # if nothing is selected, we don't want to waste resources on making a graph
selected_option2 = st.selectbox('Select an option:', options2, index=None)
selected_option3 = st.selectbox('Select an option:', options3, index=None)
result = st.button("Search", type="secondary")
cancel = st.button("Cancel", type="primary")

if selected_option3 == 'Male':
    selected_gender = 'M'
elif selected_option3 == 'Female':
    selected_gender = 'F'

if selected_option1 == 'Weight Graph':
    if selected_option2 == 'Sport':
        if result == True and cancel == False:
            st.write('Making a Weight-Sport Graph... Please be patient...')
            # create a graph that shows the connectedness of weight vs the sport, separate men/women?
            G = Weight_to_Sport_Graph(selected_gender)
            visualize_bfs(G)
            visualize_dfs(G)



    if selected_option2 == 'Medal':
        if result == True and cancel == False:
            st.write('Making a Weight-Medal Graph... Please be patient...')
            # create a graph that shows the connectedness of weight vs the medals, separate men/women?
            G = Weight_to_Medal_Graph(selected_gender)
            visualize_bfs(G)
            visualize_dfs(G)

elif selected_option1 == 'Height Graph':
    if selected_option2 == 'Sport':
        if result == True and cancel == False:
            st.write('Making a Height-Sport Graph... Please be patient...')
            # create a graph that shows the connectedness of height vs the sport, separate men/women?
            G = Height_to_Sport_Graph(selected_gender)
            visualize_bfs(G)
            visualize_dfs(G)
    if selected_option2 == 'Medal':
        if result == True and cancel == False:
            st.write('Making a Height-Medal Graph... Please be patient...')
            # create a graph that shows the connectedness of height vs the medal, separate men/women?
            G = Height_to_Medal_Graph(selected_gender)
            visualize_bfs(G)
            visualize_dfs(G)

elif selected_option1 == 'Age Graph':
    if selected_option2 == 'Sport':
        if result == True and cancel == False:
            st.write('Making an Age-Sport Graph... Please be patient...')
            # create a graph that shows the connectedness of age vs the sport
            G = Age_to_Sport_Graph(selected_gender)
            visualize_bfs(G)
            visualize_dfs(G)
    if selected_option2 == 'Medal':
        if result == True and cancel == False:
            st.write('Making an Age-Medal Graph... Please be patient...')
            # create a graph that shows the connectedness of age vs the medal
            G = Age_to_Medal_Graph(selected_gender)
            visualize_bfs(G)
            visualize_dfs(G)




# G = nx.Graph() # Create an empty undirected graph (or nx.DiGraph() for a directed graph)
# # Add nodes from the 'source' and 'target' columns
# G.add_nodes_from(olympic_dfdf['source'])
# G.add_nodes_from(olympic_dfdf['target'])
# # Add edges from the DataFrame
# edges = [(row['source'], row['target']) for index, row in df.iterrows()]
# G.add_edges_from(edges)




