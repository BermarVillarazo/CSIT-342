import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go

# Load data
df = pd.read_csv("archive/data-2.csv")


def descriptive_stats(df):
    numerical_data = df.select_dtypes(include=[np.number])
    stats_dict = {}

    for col in numerical_data.columns:
        col_data = numerical_data[col].dropna()

        if len(col_data) == 0:
            continue

        try:
            mean = col_data.mean()
            median = col_data.median()
            mode = col_data.mode()[0] if not col_data.mode().empty else np.nan
            std_dev = col_data.std()
            variance = col_data.var()
            min_value = col_data.min()
            max_value = col_data.max()
            range_value = max_value - min_value
            percentiles = np.percentile(col_data, [25, 50, 75])

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
        except Exception as e:
            stats_dict[col] = {"Error": str(e)}

    return pd.DataFrame(stats_dict).T

# Function to handle missing values
def handle_missing_values(df, action):
    if action == "Drop rows with missing values":
        return df.dropna(), "Drop rows with missing values"
    elif action == "Fill missing values with zero":
        return df.fillna(0), "Fill missing values with zero"
    else:
        return df, "No action taken"

# Function for missing values analysis
def missing_values_analysis(df):
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]
    return missing_columns

# Function to generate bar plot for unit price by country
def plot_unit_price_by_country(df, method="aggregate"):
    if method == "aggregate":
        avg_unit_price = df.groupby("Country")["UnitPrice"].median().reset_index()
        avg_unit_price = avg_unit_price.sort_values(by="UnitPrice", ascending=False)
        return px.bar(
            avg_unit_price,
            x="Country",
            y="UnitPrice",
            title="Median Unit Price by Country",
            labels={"UnitPrice": "Median Unit Price"},
            template="plotly_white"
        )
    elif method == "sample":
        sampled_df = df.sample(n=10000, random_state=42)
        median_prices = sampled_df.groupby("Country")["UnitPrice"].median().sort_values(ascending=False)
        sampled_df["Country"] = pd.Categorical(sampled_df["Country"], categories=median_prices.index, ordered=True)
        return px.box(
            sampled_df,
            x="Country",
            y="UnitPrice",
            title="Unit Price by Country (Sampled Data)",
            template="plotly_white"
        )
    return None

# START STREAMLIT CODE HERE
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Ecommerce Dataset Analysis</h1>",
    unsafe_allow_html=True,
)
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Overview", "Data Exploration", "Analysis & Insights", "Conclusion"],
        icons=["house", "graph-down", "clipboard-data", "three-dots"],
    )


if selected == "Overview":
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

# default cleaned data
if st.session_state.get("df_cleaned") is "fill":
    df_cleaned = df.fillna(0)
else:
    df_cleaned = df.dropna()

# Data Exploration
if selected == "Data Exploration":

    tabs = st.tabs(["Data Preparation", "Data Visualization"])
    
    with tabs[0]:
    # Section 1: Dataset Overview
        st.markdown("## 📊 Dataset Overview")
        st.write(
            """
            This section provides a general overview of the dataset, including its structure, dimensions, and a 
            sample of its contents. Understanding the dataset's format and key attributes is essential before 
            proceeding with any analysis.
            """
        )
        st.write(df.describe())
        st.write(f"**Dataset Dimensions:** {df.shape[0]} rows and {df.shape[1]} columns.")
        
        st.markdown("### ✏️ Sample of the Dataset")
        st.write(
            """
            Below is a preview of the dataset, showing a few rows of data. This helps us understand the columns
            available and the type of data we are working with.
            """
        )
        st.write(df.head())

        st.markdown("### 🔍 Missing Values Analysis")
        st.write(
            """
            Upon inspection, we found that the **Description** column has 1,454 missing values, while the **CustomerID** column has 135,080 missing values. The large number of missing values in the **CustomerID** column may affect the analysis of customer-related insights, as this column plays a key role in identifying individual transactions. 

            Depending on the context and analysis goals, missing values can be handled in various ways:
            - **Dropping rows** with missing values when the data is crucial.
            - **Filling missing values** with placeholders, like "No Description" for missing **Description** values, or using "Unknown" for missing **CustomerID** values to retain the dataset size and prevent loss of data.
            """
        )
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.warning(
                f"The dataset contains missing values in {missing_values[missing_values > 0].shape[0]} columns."
            )
            with st.expander("View Missing Values Details"):
                st.write(missing_values[missing_values > 0])
        else:
            st.success("No missing values found in the dataset. All columns are complete and ready for analysis.")

        # Section 2: Handling Missing Values
        st.markdown("## 🧹 Handling Missing Values")
        st.write(
            """
            To prepare the dataset for analysis, we need to address any missing values. Depending on the context, 
            missing values can either be removed entirely or replaced with an appropriate placeholder. Choose one 
            of the methods below to handle the missing values:
            """
        )
        action = st.radio(
            "Select an action for handling missing values:",
            ["Drop rows with missing values", "Fill missing values with zero"],
            index=1 if st.session_state.df_cleaned == "fill" else 0,
            horizontal=True
        )

        drop_rows_code = """
        # Drop rows with missing values
        df_cleaned = df.dropna()
        """

        fill_zero_code = """
        # Fill missing values with zero
        df_cleaned = df.fillna(0)
        """
        
        if action == "Drop rows with missing values":
            df_cleaned = df.dropna()
            st.session_state.df_cleaned = "drop"

            st.code(drop_rows_code, language='python')
            st.success(
                f"✅ Rows after dropping missing values: {df_cleaned.shape[0]} (from {df.shape[0]} rows originally)."
            )
        elif action == "Fill missing values with zero":
            df_cleaned = df.fillna(0)
            st.session_state.df_cleaned = "fill"

            st.code(fill_zero_code, language='python')
            st.success("✅ Missing values have been filled with zero.")

    with tabs[1]:
        # Section 3: Visualization
        st.markdown("## 📈 Unit Price by Country")
        st.write(
            """
            In this section, we visualize the distribution of unit prices across different countries. This allows 
            us to uncover trends, outliers, or variations in pricing based on geographical location. Choose a 
            visualization method to display this data.
            """
        )
        st.info("Visualization helps us gain insights into patterns and relationships in the data.")
        
        choice = st.radio("Select Visualization Method:", ["Aggregate Data", "Sample Data"], horizontal=True)
        if choice == "Aggregate Data":
            st.write(
                """
                **Aggregate Data:** This method calculates the median unit price for each country and displays it 
                as a bar chart. This helps us summarize and compare typical prices across different regions.
                """
            )
            avg_unit_price = df.groupby("Country")["UnitPrice"].median().reset_index()
            avg_unit_price = avg_unit_price.sort_values(by="UnitPrice", ascending=False)
            fig_bar = px.bar(
                avg_unit_price,
                x="Country",
                y="UnitPrice",
                title="Median Unit Price by Country",
                labels={"UnitPrice": "Median Unit Price"},
                template="plotly_white"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        elif choice == "Sample Data":
            st.write(
                """
                **Sample Data:** This method uses a random sample of 10,000 rows to create a box plot, which 
                visualizes the distribution of unit prices for each country. Sampling is used to improve 
                performance when dealing with large datasets.
                """
            )
            sampled_df = df.sample(n=10000, random_state=42)
            median_prices = sampled_df.groupby("Country")["UnitPrice"].median().sort_values(ascending=False)

            # Ensure the x-axis is ordered by median prices
            sampled_df["Country"] = pd.Categorical(
                sampled_df["Country"], categories=median_prices.index, ordered=True
            )
            fig_box = px.box(
                sampled_df,
                x="Country",
                y="UnitPrice",
                title="Unit Price by Country (Sampled Data)",
                template="plotly_white"
            )
            st.plotly_chart(fig_box, use_container_width=True)

            # Top 10 Most Popular Products
        st.markdown("## 🏆 Top 10 Most Popular Products")
        st.write("""
        This visualization highlights the 10 most frequently purchased products. 
        Understanding popular items can help identify key drivers of sales and frequent itemsets for association rule mining.
        """)

        # Aggregate data to count product purchases
        product_popularity = df.groupby("Description")["Quantity"].sum().reset_index()
        product_popularity = product_popularity.sort_values(by="Quantity", ascending=False).head(10)

        # Plotting the data
        fig_products = px.bar(
            product_popularity,
            x="Description",
            y="Quantity",
            title="Top 10 Most Popular Products",
            labels={"Description": "Product Description", "Quantity": "Total Quantity Sold"},
            template="plotly_white"
        )
        st.plotly_chart(fig_products, use_container_width=True)
        st.write(
            """
            The chart above highlights the best-selling items based on total quantity sold. The top product exceeds 50,000 units, while the rest range between 30,000 and 50,000 units. Popular items include the "JUMBO BAG RED RETROSPOT," "PACK OF 72 RETROSPOT CAKE CASES," and "WHITE HANGING HEART T-LIGHT HOLDER."

    The data suggests that customers prefer decorative or practical items in bulk. Retailers can use these insights to focus on stocking and promoting similar high-demand products.
            """
        )


        # Additional Notes
        st.markdown("### 📌 Notes:")
        st.write(
            """
            - Aggregate data uses the median as it is more robust against outliers compared to the mean.
            - Sample data provides a quick overview without processing the entire dataset, ensuring faster performance.
            """
        )

        # lag kaayo
        # st.markdown("### Unit Price by Country")
        # fig_box = px.box(df, x="Country", y="UnitPrice", title="Unit Price by Country")
        # st.plotly_chart(fig_box)


elif selected == "Analysis & Insights":
    st.title("Analysis & Insights")
    # write the count of df_cleaned rows
    st.write(f"rows: {df_cleaned.shape[0]}")
    st.write("todo")


elif selected == "Conclusion":
    st.title("Conclusion")
    st.write("todo")
    st.write("##### Presented by Teletubbies of CSIT342-G1")
    st.write(
        "- Aguinaldo, Rovelyn\n- Borres, Joshua\n- Tampus, Nathaniel\n- Villarazo, Bermar\n- Visbal, Andrhey"
    )
