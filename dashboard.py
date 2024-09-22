import streamlit as st
import plotly.express as px
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from matplotlib.ticker import FuncFormatter
from prophet import Prophet
import calendar 

# Page configuration
st.set_page_config(page_title="Superstore!!!", page_icon=":bar_chart:", layout="wide")

# Title and CSS
st.title(" :bar_chart: Tesco Superstore")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

# File uploader
fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding="ISO-8859-1")
else:
    os.chdir(r"C:\Users\arjun\Desktop\Streamlit")
    df = pd.read_csv("Superstore.csv", encoding="ISO-8859-1")
    
df_new=df.copy()

df["Order Date"] = pd.to_datetime(df["Order Date"])

# Getting the min and max date
startDate = pd.to_datetime(df["Order Date"]).min()
endDate = pd.to_datetime(df["Order Date"]).max()

col1, col2 = st.columns((2))

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["Order Date"] >= date1) & (df["Order Date"] <= date2)].copy()

# Calculate KPIs in millions
total_sales = df["Sales"].sum() / 1_000_000
total_profit = df["Profit"].sum() / 1_000_000
total_orders = df.shape[0]
total_loss = df[df["Profit"] < 0]["Profit"].sum() / 1_000_000



# Sidebar filters
st.sidebar.header("Choose your filter:")

region = st.sidebar.multiselect("Pick your Region", df["Region"].unique())
if not region:
    df2 = df.copy()
else:
    df2 = df[df["Region"].isin(region)]

state = st.sidebar.multiselect("Pick the State", df2["State"].unique())
if not state:
    df3 = df2.copy()
else:
    df3 = df2[df2["State"].isin(state)]

city = st.sidebar.multiselect("Pick the City", df3["City"].unique())

# Filter the data based on Region, State and City
if not region and not state and not city:
    filtered_df = df
elif not state and not city:
    filtered_df = df[df["Region"].isin(region)]
elif not region and not city:
    filtered_df = df[df["State"].isin(state)]
elif state and city:
    filtered_df = df3[df["State"].isin(state) & df3["City"].isin(city)]
elif region and city:
    filtered_df = df3[df["Region"].isin(region) & df3["City"].isin(city)]
elif region and state:
    filtered_df = df3[df["Region"].isin(region) & df3["State"].isin(state)]
elif city:
    filtered_df = df3[df3["City"].isin(city)]
else:
    filtered_df = df3[df3["Region"].isin(region) & df3["State"].isin(state) & df3["City"].isin(city)]

# Define the menu for navigation
with st.sidebar:
    selected = option_menu(
        "Main Menu", ["Home", "Actionable Insights", "Sales by Category", "Profit Margins by Region", "Time Series Analysis", "Treemap Visualization"],
        icons=['house', 'bar-chart-line', 'pie-chart', 'clock', 'tree'],
        menu_icon="cast", default_index=0
    )



# Actionable Insights Sub-tabs
if selected == "Actionable Insights":
    sub_tab = st.radio("Select a Team", ["Supply Chain Team", "Sales Team", "Marketing Team", "Product Owner"])

    if sub_tab == "Supply Chain Team":
        st.title("Supply Chain Team Insights")
        # Add your Supply Chain-related analysis and visualizations here
        st.subheader("Top Sub-Categories by Loss")
        
        # Calculate total sales and total loss per sub-category based on filtered data
        loss_by_subcategory_df = filtered_df.groupby(["Region", "State", "Category", "Sub-Category"]).agg({"Profit": "sum", "Sales": "sum"}).reset_index()
        loss_by_subcategory_df = loss_by_subcategory_df[loss_by_subcategory_df["Profit"] < 0]

        # Calculate loss percentage
        loss_by_subcategory_df["Loss Percentage"] = (-loss_by_subcategory_df["Profit"] / loss_by_subcategory_df["Sales"]) * 100

        # Get top 5 sub-categories with the highest loss percentages
        top_5_loss_subcategories_df = loss_by_subcategory_df.sort_values(by="Loss Percentage", ascending=False).head(5)

        # Create a donut chart with customized label formatting
        fig_top5_loss_donut = px.pie(
                top_5_loss_subcategories_df,
                names="Sub-Category",
                values="Loss Percentage",
                hole=0.4,
                labels={"Loss Percentage": "Loss Percentage"}
        )

        # Customize the labels to show percentage correctly
        fig_top5_loss_donut.update_traces(
                textinfo='label+percent',  # Display both label and percentage
                insidetextorientation='radial'  # Ensure text is readable and correctly placed
        )

        # Update hover info to match the displayed percentage
        fig_top5_loss_donut.update_traces(
                hovertemplate='%{label}: %{percent:.2f}%'  # Show percentage with 2 decimal places on hover
        )

        st.plotly_chart(fig_top5_loss_donut, use_container_width=True)


        st.subheader("Top Sub-Categories by Loss")
               
        # Detailed loss data for top sub-categories
        detailed_loss_df = filtered_df[filtered_df["Sub-Category"].isin(top_5_loss_subcategories_df["Sub-Category"])]
        detailed_loss_df = detailed_loss_df.groupby(["Sub-Category", "Product ID", "Product Name", "Ship Mode"]).agg({"Profit": "sum", "Sales": "sum"}).reset_index()

        # Calculate Loss Percentage as a negative value
        detailed_loss_df["Loss Percentage"] = -(detailed_loss_df["Profit"] / detailed_loss_df["Sales"].abs()) * 100

        # Ensure Loss Percentage is negative even if both Profit and y are negative
        detailed_loss_df["Loss Percentage"] = detailed_loss_df["Loss Percentage"].apply(lambda x: x if x < 0 else -x)

   
        
        # Define Recommendations based on Loss Percentage
        def supply_chain_recommendation(row):
          if row["Loss Percentage"] <= -30:  # High loss threshold (e.g., 30% or more)
            return "High Loss: Immediate analysis needed to identify and mitigate supply chain issues."
          elif -30 < row["Loss Percentage"] <= -15:  # Moderate loss threshold (e.g., 15% to 30%)
            return "Moderate Loss: Significant analysis recommended to address losses."
          elif -15 < row["Loss Percentage"] <= -5:  # Low loss threshold (e.g., 5% to 15%)
            return "Low Loss: Regular monitoring advised to prevent future losses."
          else:
            return "Minimal Loss: Losses are minor; maintain current strategies."

    # Apply the recommendation function to create the Recommendations column
        detailed_loss_df["Recommendations"] = detailed_loss_df.apply(supply_chain_recommendation, axis=1)

    # Select only relevant columns to display, including the new Recommendations column
        display_df = detailed_loss_df[["Sub-Category", "Product ID", "Product Name", "Ship Mode", "Loss Percentage", "Recommendations"]]

    # Display the DataFrame in Streamlit
        #st.write("Detailed Loss Data for Top Sub-Categories with Recommendations")
        st.dataframe(display_df, use_container_width=True)


        st.markdown('<br><br>', unsafe_allow_html=True)
        st.subheader("Shipping Mode Analysis Insights")
        
        
        
        #Assuming filtered_df is your DataFrame
    # Aggregate sales and profit data by shipping mode
        ship_mode_df = filtered_df.groupby("Ship Mode").agg({"Sales": "sum", "Profit": "sum"}).reset_index()

    # Rename columns for clarity
        ship_mode_df.columns = ['Ship Mode', 'Total Sales', 'Total Profit']

    # Convert Total Sales to millions for better readability
        ship_mode_df['Total Sales (Millions)'] = ship_mode_df['Total Sales'] / 1_000_000
        ship_mode_df['Total Profit (Millions)'] = ship_mode_df['Total Profit'] / 1_000_000

    # Create a bar chart for Total Sales by Ship Mode
        fig_sales_by_ship_mode = px.bar(
        ship_mode_df,
        x='Ship Mode',
        y='Total Sales (Millions)',
        title='Total Sales by Ship Mode',
        template='seaborn',
        color='Total Sales (Millions)',  # Color bars based on sales in millions
        labels={'Total Sales (Millions)': 'Total Sales (Millions)'},
        hover_data={
            'Ship Mode': True,
            'Total Sales (Millions)': ':.2f',  # Format sales in millions to 2 decimal places
            'Total Profit (Millions)': ':.2f'  # Format profit in millions to 2 decimal places
        }
    )

    # Update layout to format y-axis ticks to 2 decimal places
        fig_sales_by_ship_mode.update_layout(
        yaxis_tickformat='.2f'  # Format y-axis ticks to 2 decimal places
    )

   
    # Display the bar charts in the Streamlit app
        st.plotly_chart(fig_sales_by_ship_mode, use_container_width=True)

# Aggregate sales and profit data by shipping mode
        ship_mode_df = filtered_df.groupby("Ship Mode").agg({"Sales": "sum", "Profit": "sum"}).reset_index()

# Rename columns for clarity
        ship_mode_df.columns = ['Ship Mode', 'Total Sales', 'Total Profit']

# Convert Total Sales to millions for better readability
        ship_mode_df['Total Sales (Millions)'] = ship_mode_df['Total Sales'] / 1_000_000
        ship_mode_df['Total Profit (Millions)'] = ship_mode_df['Total Profit'] / 1_000_000

# Define recommendations based on the performance of each shipping mode
        def recommend_shipping_mode(row):
          if row['Total Sales (Millions)'] > 2 and row['Total Profit (Millions)'] > 0.5:
            return (
            f"High Performance: Focus on enhancing this shipping mode to maximize revenue. "
            f"Optimize logistics and increase marketing efforts."
        )
          elif row['Total Sales (Millions)'] > 1 and row['Total Profit (Millions)'] > 0:
           return (
            f"Moderate Performance: Consider optimizing operations to improve profitability. "
            f"Monitor closely and adjust strategies as needed."
        )
          else:
           return (
            f"Low Performance: Requires analysis to identify issues. "
            f"Consider restructuring or reallocating resources."
        )

# Add the Recommendations column
        ship_mode_df['Recommendations'] = ship_mode_df.apply(recommend_shipping_mode, axis=1)

# Display the updated DataFrame with recommendations
        st.write("The following table shows the sales, profit, and recommendations for each shipping mode:")
        st.dataframe(ship_mode_df[['Ship Mode', 'Total Sales (Millions)', 'Total Profit (Millions)', 'Recommendations']], use_container_width=True)

# Determine the best-performing shipping mode based on sales
        best_sales_mode = ship_mode_df.loc[ship_mode_df['Total Sales'].idxmax()]





    elif sub_tab == "Sales Team":
        st.title("Sales Team Insights")
        #st.write("Analyze data specific to the sales team performance.")
    
        

    # Aggregate sales and profit data by state
        state_sales_profit_df = filtered_df.groupby("State").agg({"Sales": "sum", "Profit": "sum"}).reset_index()

    # Rename columns for clarity
        state_sales_profit_df.columns = ['State', 'Total Sales', 'Total Profit']

    # Convert Total Sales and Profit to millions for better readability
        state_sales_profit_df['Total Sales (Millions)'] = state_sales_profit_df['Total Sales'] / 1_000_000
        state_sales_profit_df['Total Profit (Millions)'] = state_sales_profit_df['Total Profit'] / 1_000_000

    # Filter states to include only those with positive profit
        positive_profit_states_df = state_sales_profit_df[state_sales_profit_df["Total Profit (Millions)"] > 0]

    # Sort the DataFrame by 'Total Sales' in descending order and select the top 5 states
        top_5_states_df = positive_profit_states_df.sort_values(by='Total Sales', ascending=False).head(5)

    # Create a bar chart for the top 5 states
        fig_top_states_bar = px.bar(
        top_5_states_df,
        x='State',
        y='Total Sales (Millions)',
        title='Top 5 States by Sales',
        template='seaborn',
        color='Total Sales (Millions)',
        labels={'Total Sales (Millions)': 'Total Sales (Millions)'},
        hover_data={
            'State': True,
            'Total Sales (Millions)': ':.2f',
            'Total Profit (Millions)': ':.2f'
        }
    )

    # Update layout to format y-axis ticks to 2 decimal places
        fig_top_states_bar.update_layout(
        yaxis_tickformat='.2f'
    )

    # Display the bar chart in the Streamlit app
        st.plotly_chart(fig_top_states_bar, use_container_width=True)

    # Define Recommendations based on Total Sales and Profit
        def recommend(row):
            if row["Total Sales (Millions)"] > 2:  # Adjust threshold as needed
               return (
                f"High Possibility for Relocation: "
                f"Relocate or Increase Product Stock. "
                f"Total Profit: ${row['Total Profit (Millions)']:.2f} million. "
                f"Evaluate Distribution Channels. "
                f"Monitor Performance."
            )
            else:
               return (
                f"Moderate Possibility for Relocation: "
                f"Total Profit: ${row['Total Profit (Millions)']:.2f} million. "
                f"Consider reallocating resources. "
                f"Evaluate marketing and distribution strategies. "
                f"Monitor Performance."
            )

    # Add Recommendations column
        top_5_states_df["Recommendations"] = top_5_states_df.apply(recommend, axis=1)

    # Extract names, sales, profit, and recommendations of the top 5 states
        top_5_states_recommendations = top_5_states_df[['State', 'Total Sales (Millions)', 'Total Profit (Millions)', 'Recommendations']]

    # Display recommendations with horizontal scrolling enabled
        st.write("Based on the analysis, here are the top 5 states by sales along with their sales figures, profits, and recommendations:")
        st.dataframe(top_5_states_recommendations, use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

    # Aggregate sales and profit data by state
        st.write("### Sales Forecasting Analysis Insights")
    
    # Ensure Order Date is in datetime format
        filtered_df["Order Date"] = pd.to_datetime(filtered_df["Order Date"])

    # Prepare data for Prophet
        filtered_df = filtered_df.rename(columns={"Order Date": "ds", "Sales": "y"})

    
    # Ensure Order Date is in datetime format
        df_new["Order Date"] = pd.to_datetime(df_new["Order Date"])

    # Extract year and month for grouping
        df_new["month_year"] = df_new["Order Date"].dt.to_period("M")

    # Group by month_year and sum Sales
        linechart = pd.DataFrame(df_new.groupby(df_new["month_year"])["Sales"].sum()).reset_index()
        linechart.columns = ["Month_Year", "Sales"]

    # Convert Period to datetime for plotting
        linechart["Month_Year"] = linechart["Month_Year"].dt.to_timestamp()

    # Prepare data for Prophet
        prophet_df = linechart.rename(columns={"Month_Year": "ds", "Sales": "y"})

    # Initialize and fit the Prophet model
        model = Prophet()
        model.fit(prophet_df)

    # Define the future periods you want to predict
        future_periods = 12  # Forecast for 12 months into the future
        future = model.make_future_dataframe(periods=future_periods, freq='M')

    # Make predictions
        forecast = model.predict(future)

    # Ensure that forecast DataFrame has the expected columns
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Combine historical data and forecast
        combined_data = pd.concat([prophet_df.set_index('ds'), forecast.set_index('ds')], axis=0).reset_index()

    # Separate historical and forecasted data
        historical_data = combined_data[combined_data['ds'].dt.year < 2018]
        forecast_2018 = combined_data[combined_data['ds'].dt.year == 2018]

    # Plotting the forecast with historical data
        fig = go.Figure()

    # Add historical sales data
        fig.add_trace(go.Scatter(
        x=historical_data["ds"],
        y=historical_data["y"],
        mode='lines',
        name='Historical Sales',
        line=dict(color='blue', width=2)
    ))

    # Add forecasted sales data
        fig.add_trace(go.Scatter(
        x=forecast_2018["ds"],
        y=forecast_2018["yhat"],
        mode='lines',
        name='Forecasted Sales',
        line=dict(color='orange', dash='dash', width=2)
    ))

    # Add confidence interval
        fig.add_trace(go.Scatter(
        x=forecast_2018["ds"].tolist() + forecast_2018["ds"].tolist()[::-1],
        y=forecast_2018["yhat_upper"].tolist() + forecast_2018["yhat_lower"].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 165, 0, 0.2)',
        line=dict(color='rgba(255, 165, 0, 0)'),
        showlegend=False,
        name='Confidence Interval'
    ))

    # Update layout for a better look
        fig.update_layout(
        title="Sales Forecast with Prophet",
        xaxis_title="Date",
        yaxis_title="Sales Amount",
        template="gridon",
        height=500,
        width=1000,
        hovermode="x unified"
    )

    # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    
    # Calculate peak sales month in 2018
        peak_month = forecast_2018.loc[forecast_2018["yhat"].idxmax()]
        peak_month_sales = peak_month["yhat"] / 1_000_000  # Convert to millions
        peak_month_date = peak_month["ds"].strftime("%B %Y")

    # Recommendations based on the forecast
        st.subheader("Sales Forecast Recommendations")

        recommendations = f"""
    Based on the sales forecast for 2018, the peak sales month is {peak_month_date} with a forecasted sales amount of ${peak_month_sales:,.2f} million.

    **Recommendations:**
    - **🙂 Focus Marketing Efforts:** Increase marketing and promotional activities during {peak_month_date} to maximize sales.
    - **🙂 Inventory Planning:** Ensure adequate inventory levels for high-demand periods.
    - **🙂 Customer Engagement:** Engage with customers proactively during peak periods to enhance sales performance.
    """

        st.markdown(recommendations)
    
 
    

    elif sub_tab == "Marketing Team":
        #st.subheader("Marketing Team Insights")
        #st.write("Analyze data specific to marketing strategies.")
        # Add your Marketing Team-related analysis and visualizations here
        
        # Adding space using Markdown
        #st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Aggregate sales and profit data by state
        st.title("Marketing Team Insights")
        

        segment_agg_df = filtered_df.groupby("Segment").agg({"Sales": "sum", "Profit": "sum"}).reset_index()

        # Create the bar chart
        fig_segment_bar = px.bar(
            segment_agg_df, 
            x="Segment", 
            y=["Sales", "Profit"],
            title="Sales and Profit by Customer Segment",
            barmode="group",
            template="seaborn"
        )

        # Display the bar chart in the Streamlit app
        st.plotly_chart(fig_segment_bar, use_container_width=True)

        # Find the segment with the highest profit
        max_profit_segment = segment_agg_df.loc[segment_agg_df['Profit'].idxmax()]
        highest_profit_segment_name = max_profit_segment['Segment']
        highest_profit_value = max_profit_segment['Profit']
        highest_sales_value = max_profit_segment['Sales']

        # Display recommendations
        st.write("### Recommendations")

        # Display the recommendation
        st.write(f"The customer segment with the highest profit is **{highest_profit_segment_name}**.")
        st.write(f"**Profit**: ${highest_profit_value:,.2f}")
        st.write(f"**Total Sales**: ${highest_sales_value:,.2f}")

        # Suggested actions based on the segment with the highest profit
        st.write("#### Suggested Actions:")

        if highest_profit_segment_name == "Consumer":
            st.write("- **Increase Product Offerings**: Collaborate with the CRM team to tailor product offerings based on customer demand and preferences.")
            st.write("- **Optimize Customer Engagement**: Use insights from CRM data to enhance marketing strategies and improve customer satisfaction.")
        elif highest_profit_segment_name == "Corporate":
            st.write("- **Tailor Business Solutions**: Customize offerings to meet the specific needs of different corporate groups, such as providing specialized accounts or financing options for diverse business types.")
            st.write("- **Enhance Corporate Services**: Develop and market specialized products or services that cater to distinct corporate needs, such as advanced features for tech companies or bulk supplies for large enterprises.")
        elif highest_profit_segment_name == "Home Office":
            st.write("- **Offer Tailored Office Supplies**: Provide products that meet the specific needs of different office setups, such as cost-effective essentials for small businesses or high-quality tools for creative professionals.")
            st.write("- **Segmented Product Offerings**: Design product ranges that cater to the varying requirements of home office users, including durable materials for educational institutions and premium supplies for high-end offices.")
        


    elif sub_tab == "Product Owner":

        st.title("Product Owner Insights")
        #st.write("Analyze data specific to product performance and ownership.")
    
    # Adding space using Markdown
        #st.markdown("<br><br>", unsafe_allow_html=True)

        #st.title("Profit Analysis")

        st.subheader("Top Sub-Categories by Profit")
    
    # Calculate profit by sub-category
        profit_by_subcategory_df = filtered_df.groupby(["Region", "State", "Category", "Sub-Category"]).agg({"Profit": "sum"}).reset_index()
        profit_by_subcategory_df.sort_values(by="Profit", ascending=False, inplace=True)

    # Display top 5 sub-categories by profit
        top_5_subcategories_df = profit_by_subcategory_df.head(5)
        fig_top_subcategories = px.pie(
        top_5_subcategories_df, 
        names="Sub-Category", 
        values="Profit",
        title="Top 5 Sub-Categories by Profit",
        template="seaborn", 
        hole=0.4
    )
        #st.plotly_chart(fig_top_subcategories, use_container_width=True)

    # Calculate and display profit percentages
        total_profit_subcategories = top_5_subcategories_df["Profit"].sum()
        top_5_subcategories_df["Profit Percentage"] = (top_5_subcategories_df["Profit"] / total_profit_subcategories) * 100
    
        fig_profit_percentage = px.pie(
        top_5_subcategories_df, 
        names="Sub-Category", 
        values="Profit Percentage",
        template="seaborn", 
        hole=0.4,
        labels={'Profit Percentage': '% Profit'}  
    )
        fig_profit_percentage.update_traces(textinfo='label+percent')
        st.plotly_chart(fig_profit_percentage, use_container_width=True)

        st.subheader("Detailed Profit Data for Top Sub-Categories")
    
    # Detailed profit data for top sub-categories
        detailed_profit_df = filtered_df[filtered_df["Sub-Category"].isin(top_5_subcategories_df["Sub-Category"])]
        detailed_profit_df = detailed_profit_df.groupby(["Sub-Category", "Product ID", "Product Name"]).agg({"Profit": "sum", "Sales": "sum"}).reset_index()

    # Calculate Profit Percentage
        detailed_profit_df["Profit Percentage"] = (detailed_profit_df["Profit"] / detailed_profit_df["Sales"]) * 100

    # Filter out negative profits
        detailed_profit_df = detailed_profit_df[detailed_profit_df["Profit"] > 0]
    
    # Filter out negative or infinite profit percentages
        detailed_profit_df = detailed_profit_df[detailed_profit_df["Profit Percentage"] > 0]

    # Define Recommendations based on profit percentage
        def recommend(row):
            if row["Profit Percentage"] > 20:
              return "High investment recommended; Increase stock significantly."
            elif 10 < row["Profit Percentage"] <= 20:
               return "Moderate investment recommended; Increase stock moderately."
            else:
               return "Low investment; Consider minimal stock increase."

    # Add Recommendations column
        detailed_profit_df["Recommendations"] = detailed_profit_df.apply(recommend, axis=1)

    # Select only relevant columns to display
        display_df = detailed_profit_df[["Sub-Category", "Product ID", "Product Name", "Profit Percentage", "Recommendations"]]

    # Display the DataFrame in Streamlit
        #st.write("Detailed Profit Data for Top Sub-Categories (Positive Percentages Only)")
        st.dataframe(display_df, use_container_width=True)



   


if selected == "Home":
    #st.subheader("Overview KPIs")
    

    with col1:
        st.subheader("Total Sales")
        st.info(f"${total_sales:,.2f}M")

        st.subheader("Total Profit")
        st.success(f"${total_profit:,.2f}M")
        
  

    with col2:
        st.subheader("Total Orders")
        st.warning(total_orders)

        st.subheader("Total Loss")
        st.error(f"${total_loss:,.2f}M")
        
   
###########################################################################

elif selected == "Sales by Category":
    st.subheader("Category wise Sales")
    category_df = filtered_df.groupby(by=["Category"], as_index=False)["Sales"].sum()
    fig = px.bar(category_df, x="Category", y="Sales", text=['${:,.2f}'.format(x) for x in category_df["Sales"]],
                 template="seaborn")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Category_ViewData"):
        st.write(category_df.style.background_gradient(cmap="Blues"))
        csv = category_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="Category.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file')
        
        
    # Aggregate sales and profit by state
    state_summary = df.groupby('State').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()

    # Aggregate sales and profit by city
    city_summary = df.groupby('City').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()

    # Sort and get top 5 states and cities by sales and profit
    top_states_by_sales = state_summary.sort_values('Sales', ascending=False).head(5)
    top_states_by_profit = state_summary.sort_values('Profit', ascending=False).head(5)
    top_cities_by_sales = city_summary.sort_values('Sales', ascending=False).head(5)
    top_cities_by_profit = city_summary.sort_values('Profit', ascending=False).head(5)

    # Streamlit app
   

    # Formatter function to convert numbers to millions
    def millions(x, pos):
        'The two args are the value and tick position'
        return f'{x * 1e-6:.1f}M'

    # Create two columns
    col1, col2 = st.columns(2)

    # Top 5 States by Sales
    with col1:
        #st.markdown("### Top 5 States by Sales")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='State', y='Sales', data=top_states_by_sales, palette='viridis', ax=ax)
        ax.set_title('Top 5 States by Sales')
        ax.set_xlabel('State')
        ax.set_ylabel('Total Sales (in millions)')
        ax.yaxis.set_major_formatter(FuncFormatter(millions))  # Format y-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        #st.pyplot(fig)

 

    # Top 5 Cities by Sales
    with col2:
        #st.markdown("### Top 5 Cities by Sales")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='City', y='Sales', data=top_cities_by_sales, palette='viridis', ax=ax)
        ax.set_title('Top 5 Cities by Sales')
        ax.set_xlabel('City')
        ax.set_ylabel('Total Sales (in millions)')
        ax.yaxis.set_major_formatter(FuncFormatter(millions))  # Format y-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        #st.pyplot(fig)

   
        
    

    if state:
        st.subheader("Sales by City in Selected States")
        city_sales_df = filtered_df.groupby(by=["State", "City"], as_index=False)["Sales"].sum()
        fig_city_sales = px.bar(city_sales_df, x="City", y="Sales", color="State", text=['${:,.2f}'.format(x) for x in city_sales_df["Sales"]],
                                template="seaborn", barmode="group")
        st.plotly_chart(fig_city_sales, use_container_width=True)

        with st.expander("City Sales View Data"):
            st.write(city_sales_df.style.background_gradient(cmap="Blues"))
            csv = city_sales_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Data", data=csv, file_name="CitySales.csv", mime="text/csv",
                               help='Click here to download the data as a CSV file')

elif selected == "Profit Margins by Region":
    st.subheader("Profit by Region")
    
    # Calculate profit by region
    profit_by_region_df = filtered_df.groupby("Region").agg({"Profit": "sum"}).reset_index()
    fig_region_profit = px.bar(profit_by_region_df, x="Region", y="Profit", text=['${:,.2f}'.format(x) for x in profit_by_region_df["Profit"]],
                              template="seaborn", title="Total Profit by Region")
    st.plotly_chart(fig_region_profit, use_container_width=True)

    with st.expander("Profit by Region View Data"):
        st.write(profit_by_region_df.style.background_gradient(cmap="Oranges"))
        csv = profit_by_region_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="ProfitByRegion.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file')
        
      
    # st.subheader("Top Sub-Categories by Profit Percentage")
  
     # Calculate profit by sub-category
    profit_by_subcategory_df = filtered_df.groupby(["Region", "State", "Category", "Sub-Category"]).agg({"Profit": "sum"}).reset_index()
    profit_by_subcategory_df.sort_values(by="Profit", ascending=False, inplace=True)

    # Display top 5 sub-categories by profit
    top_5_subcategories_df = profit_by_subcategory_df.head(5)
    fig_top_subcategories = px.pie(top_5_subcategories_df, names="Sub-Category", values="Profit",
                                   title="Top 5 Sub-Categories by Profit",
                                   template="seaborn", hole=0.4)
    #st.plotly_chart(fig_top_subcategories, use_container_width=True)
    
    # Calculate and display profit percentages
    total_profit_subcategories = top_5_subcategories_df["Profit"].sum()
    top_5_subcategories_df["Profit Percentage"] = (top_5_subcategories_df["Profit"] / total_profit_subcategories) * 100
    fig_profit_percentage = px.pie(top_5_subcategories_df, names="Sub-Category", values="Profit Percentage",
                                   template="seaborn", hole=0.4,
                                   labels={'Profit Percentage': '% Profit'})  # Adjusted label for percentage
    fig_profit_percentage.update_traces(textinfo='label+percent')  # Update trace to show label and percent
    #st.plotly_chart(fig_profit_percentage, use_container_width=True)
    
    
    # Download the full profit by sub-category data
    # with st.expander("Profit by Sub-Category View Data"):
    #     st.write(profit_by_subcategory_df.style.background_gradient(cmap="Oranges"))
    #     csv = profit_by_subcategory_df.to_csv(index=False).encode('utf-8')
    #     st.download_button("Download Data", data=csv, file_name="ProfitBySubCategory.csv", mime="text/csv",
    #                     help='Click here to download the data as a CSV file')


    
    # st.subheader("Top Sub-Categories by Loss Percentage")

    # Calculate total sales and total loss per sub-category
    loss_by_subcategory_df = filtered_df.groupby("Sub-Category").agg({"Profit": "sum", "Sales": "sum"}).reset_index()
    loss_by_subcategory_df = loss_by_subcategory_df[loss_by_subcategory_df["Profit"] < 0]

    # Calculate loss percentage
    loss_by_subcategory_df["Loss Percentage"] = (-loss_by_subcategory_df["Profit"] / loss_by_subcategory_df["Sales"]) * 100

    # Get top 5 sub-categories with the highest loss percentages
    top5_loss_subcategories = loss_by_subcategory_df.sort_values(by="Loss Percentage", ascending=False).head(5)

    # Create a donut chart with customized label formatting
    fig_top5_loss_donut = px.pie(
        top5_loss_subcategories,
        names="Sub-Category",
        values="Loss Percentage",
        hole=0.4,
        labels={"Loss Percentage": "Loss Percentage"}
    )

    # Customize the labels to show percentage correctly
    fig_top5_loss_donut.update_traces(
        textinfo='label+percent',  # Display both label and percentage
        insidetextorientation='radial'  # Ensure text is readable and correctly placed
    )

    # Update hover info to match the displayed percentage
    fig_top5_loss_donut.update_traces(
        hovertemplate='%{label}: %{percent:.2f}%'  # Show percentage with 2 decimal places on hover
    )

    #st.plotly_chart(fig_top5_loss_donut, use_container_width=True)

    # with st.expander("Top 5 Sub-Categories by Loss Percentage View Data"):
    #     st.write(top5_loss_subcategories.style.background_gradient(cmap="Reds"))
    #     csv = top5_loss_subcategories.to_csv(index=False).encode('utf-8')
    #     st.download_button("Download Data", data=csv, file_name="Top5LossPercentageSubCategories.csv", mime="text/csv",
    #                     help='Click here to download the data as a CSV file')

    
    if region:
        st.subheader("Profit by State within Selected Regions")
        
        # Calculate profit by state within selected regions
        profit_by_state_df = filtered_df.groupby(["Region", "State"]).agg({"Profit": "sum"}).reset_index()
        fig_state_profit = px.bar(profit_by_state_df, x="State", y="Profit", color="Region", text=['${:,.2f}'.format(x) for x in profit_by_state_df["Profit"]],
                                 template="seaborn", title="Total Profit by State within Selected Regions", barmode="group")
        st.plotly_chart(fig_state_profit, use_container_width=True)

        with st.expander("Profit by State View Data"):
            st.write(profit_by_state_df.style.background_gradient(cmap="Oranges"))
            csv = profit_by_state_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Data", data=csv, file_name="ProfitByState.csv", mime="text/csv",
                               help='Click here to download the data as a CSV file')

    # Additional Visualization: Profit by Sub-Category
    st.subheader("Profit by Sub-Category within Selected Regions/States")
    
    # Calculate profit by sub-category
    profit_by_subcategory_df = filtered_df.groupby(["Region", "State", "Category", "Sub-Category"]).agg({"Profit": "sum"}).reset_index()
    profit_by_subcategory_df.sort_values(by="Profit", ascending=False, inplace=True)

    unique_states = filtered_df['State'].unique()
    for state in unique_states:
        state_df = profit_by_subcategory_df[profit_by_subcategory_df['State'] == state]
        fig_subcategory_profit = px.bar(state_df, x="Sub-Category", y="Profit", color="Category",
                                       text=['${:,.2f}'.format(x) for x in state_df["Profit"]],
                                       template="seaborn", title=f"Profit by Sub-Category for {state}")
        st.plotly_chart(fig_subcategory_profit, use_container_width=True)

    with st.expander("Profit by Sub-Category View Data"):
        st.write(profit_by_subcategory_df.style.background_gradient(cmap="Oranges"))
        csv = profit_by_subcategory_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="ProfitBySubCategory.csv", mime="text/csv",
                           help='Click here to download the data as a CSV file')



elif selected == "Time Series Analysis":
    st.subheader('Time Series Analysis')

    # Ensure Order Date is in datetime format
    filtered_df["Order Date"] = pd.to_datetime(filtered_df["Order Date"])

    # Extract year and month for grouping
    filtered_df["month_year"] = filtered_df["Order Date"].dt.to_period("M")
    linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"])["Sales"].sum()).reset_index()
    linechart.columns = ["Month_Year", "Sales"]

    # Convert Period to datetime for plotting
    linechart["Month_Year"] = linechart["Month_Year"].dt.to_timestamp()

    # Plotting the basic line chart
    fig2 = px.line(linechart, x="Month_Year", y="Sales", labels={"Sales": "Amount"}, height=500, width=1000, template="gridon")
    fig2.update_layout(title="Sales Over Time")
    st.plotly_chart(fig2, use_container_width=True)


    with st.expander("View Data of TimeSeries:"):
        st.write(linechart.T.style.background_gradient(cmap="Blues"))
        csv = linechart.to_csv(index=False).encode("utf-8")
        st.download_button('Download Data', data=csv, file_name="TimeSeries.csv", mime='text/csv')






elif selected == "Treemap Visualization":
    st.subheader("Hierarchical view of Sales using TreeMap")
    fig3 = px.treemap(filtered_df, path=["Region", "Category", "Sub-Category"], values="Sales", hover_data=["Sales"],
                      color="Sales", color_continuous_scale="Viridis")
    fig3.update_layout(width=800, height=650)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader('Segment-wise Sales')
    fig_segment = px.pie(filtered_df, values="Sales", names="Segment", template="plotly_dark")
    fig_segment.update_traces(textinfo='label+percent', textposition="inside")
    st.plotly_chart(fig_segment, use_container_width=True)


    # Detailed download options
    st.subheader('Download Detailed Data')
    detailed_data = filtered_df.groupby(["Region", "Category", "Sub-Category"]).agg({"Sales": "sum", "Profit": "sum"}).reset_index()
    csv = detailed_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Detailed Sales Data", data=csv, file_name="Detailed_Sales_Data.csv", mime="text/csv")
    
    
    

    # Aggregate data by shipping mode
    df_agg = df.groupby('Ship Mode', as_index=False).agg({'Quantity': 'sum'})

    # Create a pie chart for shipping modes
    fig_category = px.pie(
        df_agg,
        values="Quantity",  # Column for pie slice sizes
        names="Ship Mode",  # Column for pie slice labels
        template="gridon"
    )
    fig_category.update_traces(textinfo='label+percent', textposition="inside")

    # Display the pie chart in the Streamlit app
    st.subheader('Shipping Modes Distribution')
    st.plotly_chart(fig_category, use_container_width=True)
    
  
    # Download button for aggregated data
    st.subheader('Download Aggregated Data')

    # Convert the aggregated data to CSV
    csv = df_agg.to_csv(index=False).encode('utf-8')

    # Create a download button for the CSV file
    st.download_button(
        label="Download Aggregated Data as CSV",
        data=csv,
        file_name="Aggregated_Shipping_Data.csv",
        mime="text/csv"
    )
    


