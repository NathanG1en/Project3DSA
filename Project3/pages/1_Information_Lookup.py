import pandas as pd
import streamlit as st

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

data = pd.read_csv("Project3/Data/archive/athlete_events.csv")

def search_athlete_by_name(name, data):
    # F(x) didn't work before because athletes have middle names
    name_parts = name.lower().split()
    # take user input and get all individual words
    
    # set that checks if full name in df has all the name parts
    def name_in_full_name(full_name, name_parts):
        return all(part in full_name for part in name_parts)

    data['lower_name'] = data['Name'].str.lower()
    athlete = data[data['lower_name'].apply(lambda x: name_in_full_name(x, name_parts))]

    if athlete.empty:
        return None
    else:
        return athlete



def main():
    st.title("Search For Your Favorite Athlete!")


    # user input
    athlete_name = st.text_input("Enter the name of the athlete:")

    if st.button("Search"):
        # searching
        result = search_athlete_by_name(athlete_name, data)
        if result is None:
            st.write("Name does not exist.")
        else:
            st.write("Athlete Found:")
            st.write(result)

if __name__ == "__main__":
    main()


