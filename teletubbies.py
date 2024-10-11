import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go




# Load data
df = pd.read_csv("archive/StudentPerformanceFactors.csv")

# Function to convert categorical variables to numeric
def convert_categorical_to_numeric(df):
    # Specify the columns you want to convert
    categorical_columns = {
        'Parental_Involvement': {'Low': 1, 'Medium': 2, 'High': 3},
        'Access_to_Resources': {'Low': 1, 'Medium': 2, 'High': 3},
        'Extracurricular_Activities': {'Yes': 1, 'No': 0},
        'Motivation_Level': {'Low': 1, 'Medium': 2, 'High': 3},
        'Internet_Access': {'Yes': 1, 'No': 0},
        'Family_Income': {'Low': 1, 'Medium': 2, 'High': 3},
        'Teacher_Quality': {'Low': 1, 'Medium': 2, 'High': 3},
        'School_Type': {'Public': 1, 'Private': 2},
        'Peer_Influence': {'Negative': 0, 'Neutral': 1, 'Positive': 2},
        'Learning_Disabilities': {'Yes': 1, 'No': 0},
        'Parental_Education_Level': {'High School': 1, 'College': 2, 'Postgraduate': 3},
        'Distance_from_Home': {'Near': 1, 'Moderate': 2, 'Far': 3},
        'Gender': {'Male': 1, 'Female': 2}
    }

    for column, mapping in categorical_columns.items():
        df[column] = df[column].map(mapping)

    return df

df = convert_categorical_to_numeric(df)


def descriptive_stats(df):
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


def plot_histogram(df, column):
    st.write(f"### Histogram for {column}")
    fig = px.histogram(df, x=column, nbins=10)
    st.plotly_chart(fig)

def plot_boxplot(df, column):
    st.write(f"### Box Plot for {column}")
    fig = px.box(df, y=column)
    st.plotly_chart(fig)

def plot_correlation_matrix(df):
    st.write("### Correlation Matrix Heatmap")
    numerical_data = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_data.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis'
    ))
    fig.update_layout(width=800, height=600)
    st.plotly_chart(fig)

def plot_exam_score_by_parental_education(df):
    st.write("### Distribution of Exam Scores by Parental Education Level")
    fig = px.violin(df, x="Parental_Education_Level", y="Exam_Score", box=True, points="all")
    st.plotly_chart(fig)


# Interactive Streamlit App
st.title("Student Performance Analysis")

selected_nav = option_menu(None, ["Introduction", "Visualizations", "Conclusion"], icons=['house', 'graph-up', 'three-dots'], menu_icon="cast", default_index=0, orientation="horizontal")

if selected_nav == "Introduction":
    st.write("### Introduction")
    st.write("This dataset offers an extensive summary of different elements impacting student exam performance. It encompasses details on study routines, attendance, parental engagement, and other factors contributing to academic excellence.")
    st.write("The dataset includes the following columns:")
    st.write(df.sample(5))

    st.write("### Descriptive Statistics")
    st.write(descriptive_stats(df))

    st.write("##### Presented by Teletubbies of CSIT342-G1")
    st.write("- Aguinaldo, Rovelyn\n- Borres, Joshua\n- Tampus, Nathaniel\n- Villarazo, Bermar\n- Visbal, Andrhey")

elif selected_nav == "Visualizations":

    histogram, boxplot, correlation_matrix = st.tabs(["Histogram", "Boxplot", "Correlation Matrix"])

    with histogram:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_column = st.selectbox(
            "Select a column for the Histogram", numerical_columns
        )

        plot_histogram(df, selected_column)
    

    with boxplot:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_column = st.selectbox(
            "Select a column for the Boxplot", numerical_columns
        )
        plot_boxplot(df, selected_column)

    with correlation_matrix:
        plot_correlation_matrix(df)
    # with exam_score_by_parental_education:
    #     plot_exam_score_by_parental_education(df)

elif selected_nav == "Conclusion":
    st.write("### Conclusion")
    st.write("Based on the visualizations, we can see that the distribution of exam scores varies by parental education level.")
    st.write("There is a positive correlation between parental involvement and exam scores.")
    st.write("Further analysis can be done to explore the impact of other factors on student performance.")