import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import math
from streamlit_option_menu import option_menu



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


# Visualization functions
def plot_histogram(df, column):
    st.write(f"### Histogram for {column}")
    fig, ax = plt.subplots()
    df[column].hist(ax=ax, bins=10)
    st.pyplot(fig)


def plot_boxplot(df, column):
    st.write(f"### Box Plot for {column}")
    fig, ax = plt.subplots()
    sns.boxplot(df[column], ax=ax)
    st.pyplot(fig)


def plot_correlation_matrix(df):
    st.write("### Correlation Matrix Heatmap")
    numerical_data = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)


def plot_exam_score_by_parental_education(df):
    st.write("### Distribution of Exam Scores by Parental Education Level")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.violinplot(
        data=df, x="Parental_Education_Level", y="Exam_Score", palette="Set3", ax=ax
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)


# Interactive Streamlit App
st.title("Student Performance Analysis")

# # Show descriptive stats
# if st.sidebar.checkbox("Show Descriptive Statistics"):
#     stats_summary = descriptive_stats(df)
#     st.write("### Basic Descriptive Statistics")
#     st.write(stats_summary)

# # Plot Histogram
# if st.sidebar.checkbox("Plot Histogram"):
#     numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
#     selected_column = st.sidebar.selectbox(
#         "Select a column for Histogram", numerical_columns
#     )
#     plot_histogram(df, selected_column)

# # Plot Boxplot
# if st.sidebar.checkbox("Plot Boxplot"):
#     numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
#     selected_column = st.sidebar.selectbox(
#         "Select a column for Boxplot", numerical_columns
#     )
#     plot_boxplot(df, selected_column)

# # Plot Correlation Matrix
# if st.sidebar.checkbox("Plot Correlation Matrix"):
#     plot_correlation_matrix(df)

# # Plot Exam Score by Parental Education
# if st.sidebar.checkbox("Plot Exam Score by Parental Education Level"):
#     plot_exam_score_by_parental_education(df)


# intro, visualizations, conclusion = st.tabs(["Introduction", "Visualizations", "Conclusion"])
selected_nav = option_menu(None, ["Introduction", "Visualizations", "Conclusion"], icons=['house', 'graph-up', 'three-dots'], menu_icon="cast", default_index=0, orientation="horizontal")

if selected_nav == "Introduction":
    st.write("### Introduction")
    st.write("This dataset contains information about student performance factors.")
    st.write("The dataset includes the following columns:")
    st.write(descriptive_stats(df))

elif selected_nav == "Visualizations":

    histogram, boxplot, correlation_matrix, exam_score_by_parental_education = st.tabs(["Histogram", "Boxplot", "Correlation Matrix", "Exam Score by Parental Education"])

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
    with exam_score_by_parental_education:
        plot_exam_score_by_parental_education(df)

elif selected_nav == "Conclusion":
    st.write("### Conclusion")
    st.write("Based on the visualizations, we can see that the distribution of exam scores varies by parental education level.")
    st.write("There is a positive correlation between parental involvement and exam scores.")
    st.write("Further analysis can be done to explore the impact of other factors on student performance.")