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
    st.write(f"This histogram shows the distribution of values for the selected column.")
    
    categorical_columns = {
        'Parental_Involvement': {1: 'Low', 2: 'Medium', 3: 'High'},
        'Access_to_Resources': {1: 'Low', 2: 'Medium', 3: 'High'},
        'Extracurricular_Activities': {1: 'Yes', 0: 'No'},
        'Motivation_Level': {1: 'Low', 2: 'Medium', 3: 'High'},
        'Internet_Access': {1: 'Yes', 0: 'No'},
        'Family_Income': {1: 'Low', 2: 'Medium', 3: 'High'},
        'Teacher_Quality': {1: 'Low', 2: 'Medium', 3: 'High'},
        'School_Type': {1: 'Public', 2: 'Private'},
        'Peer_Influence': {0: 'Negative', 1: 'Neutral', 2: 'Positive'},
        'Learning_Disabilities': {1: 'Yes', 0: 'No'},
        'Parental_Education_Level': {1: 'High School', 2: 'College', 3: 'Postgraduate'},
        'Distance_from_Home': {1: 'Near', 2: 'Moderate', 3: 'Far'},
        'Gender': {1: 'Male', 2: 'Female'}
    }

    df_show = df.copy()

    if column in categorical_columns:
        df_show[column] = df_show[column].map(categorical_columns[column])
    
    fig = px.histogram(df_show, x=column, nbins=10)
    st.plotly_chart(fig)

def plot_boxplot(df, column):
    st.write(f"### Box Plot for {column}")
    st.write(f"This box plot visualizes the spread and outliers of the selected column.")
    fig = px.box(df, y=column)
    st.plotly_chart(fig)
    

def plot_correlation_matrix(df):
    st.write("### Correlation Matrix Heatmap")
    st.write("This heatmap displays the correlation coefficients between all columns.")
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
    st.write("""
    The dataset offers insights into factors affecting student performance. Key observations include:

    1. **Central Tendency & Variability**:
        - Students study an average of 20 hours per week, sleep 7 hours, and have 80% school attendance.
        - High variability in Previous Scores, Tutoring Sessions, and Hours Studied; low variability in Sleep Hours and Exam Scores.

    2. **Outliers & Skewness**:
        - Outliers in Hours Studied, Attendance, Previous Scores, and Tutoring Sessions indicate extreme behaviors.
        - Tutoring Sessions and Physical Activity are right-skewed, with most students engaging minimally.

    3. **Correlations**:
        - Strongest correlation between Exam Scores and Attendance (0.58).
        - Hours Studied show a modest positive correlation with Exam Scores.
        - Distance from Home has a weak negative correlation with Exam Scores.
    """)