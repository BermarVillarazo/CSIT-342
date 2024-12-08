import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go

# Load data
df = pd.read_csv("archive/data-2.csv")


def descriptive_stats(df):
    # TODO sayop guro ni? gikan ra sa midterm project nato sauna
    # Select numerical columns only
    numerical_data = df.select_dtypes(include=[np.number])

    # Create a dictionary to store statistics
    stats_dict = {}

    for col in numerical_data.columns:
        col_data = numerical_data[col].dropna()  # Drop NaN values

        if len(col_data) == 0:  # Skip empty columns
            continue

        # Calculate basic statistics using pandas methods for mean, median, std, min, max, percentiles
        mean = col_data.mean()
        median = col_data.median()
        mode = col_data.mode()[0] if not col_data.mode().empty else np.nan
        std_dev = col_data.std()
        variance = col_data.var()
        min_value = col_data.min()
        max_value = col_data.max()
        range_value = max_value - min_value
        percentiles = np.percentile(col_data, [25, 50, 75])

        # Store all statistics in a dictionary
        stats_dict[col] = {
            "Mean": mean,
            "Median": median,
            "Mode": mode,
            "Standard Deviation": std_dev,
            "Variance": variance,
            "Min": min_value,
            "Max": max_value,
            "Range": range_value,
            "25th Percentile": percentiles[0],
            "50th Percentile (Median)": percentiles[1],
            "75th Percentile": percentiles[2],
        }

    # Convert the dictionary to a pandas DataFrame for a clean output
    stats_df = pd.DataFrame(stats_dict).T
    return stats_df


# START STREAMLIT CODE HERE
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Ecommerce Dataset Analysis</h1>",
    unsafe_allow_html=True,
)

selected_nav = option_menu(
    None,
    ["Overview", "Data Exploration", "Analysis & Insights", "Conclusion"],
    icons=["house", "graph-down", "clipboard-data", "three-dots"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

if selected_nav == "Overview":
    st.write("### Overview TODO KUWANG/SAYOP PA GURO NI")
    st.write(
        "This Ecommerce dataset contains all the transactions occurring for a UK-based and registered non-store online retail between 01/12/2009 and 09/12/2011.The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers."
    )
    st.write("The dataset includes the following columns:")
    st.write(df.sample(5))

    st.write("### Descriptive Statistics")
    st.write(descriptive_stats(df))

    st.write("##### Presented by Teletubbies of CSIT342-G1")
    st.write(
        "- Aguinaldo, Rovelyn\n- Borres, Joshua\n- Tampus, Nathaniel\n- Villarazo, Bermar\n- Visbal, Andrhey"
    )

elif selected_nav == "Data Exploration":
    st.title("Data Exploration and Preparation TODO KUWANG PA")
    st.markdown("### Dataset Overview")
    st.write(df.describe())

    st.markdown("### Handling Missing Values")
    st.write("TODO DATA CLEANING ??")

    # lag kaayo
    # st.markdown("### Unit Price by Country")
    # fig_box = px.box(df, x="Country", y="UnitPrice", title="Unit Price by Country")
    # st.plotly_chart(fig_box)

elif selected_nav == "Analysis & Insights":
    st.title("Analysis & Insights")
    st.write("todo")

elif selected_nav == "Conclusion":
    st.title("Conclusion")
    st.write("todo")
    st.write("##### Presented by Teletubbies of CSIT342-G1")
    st.write(
        "- Aguinaldo, Rovelyn\n- Borres, Joshua\n- Tampus, Nathaniel\n- Villarazo, Bermar\n- Visbal, Andrhey"
    )
