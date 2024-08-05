import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from branca.colormap import linear
import matplotlib.pyplot as plt
import mplcursors
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from sklearn.metrics import mean_squared_error

# Set the page configuration
st.set_page_config(page_title="Healthcare Analysis", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["Home", "General MENA View", "Lebanon Analysis", "Models","Lebanon and Saudi Arabia", "Recommendations"]
selected_page = st.sidebar.radio("Go to", pages)

# Load data using Streamlit's caching to improve performance
@st.cache_data
def load_data():
    data_mena = pd.read_csv("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/IHME-GBD_2021_DATA-088f3c05-1.csv")
    data_lebanon_saudi = pd.read_csv("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/combined_lebanon_saudi.csv")
#   data_other_measures = pd.read_csv("C:\\Users\\Lenovo\\OneDrive\\01_Education\\02_AUB\\03_Summer2024\\MSBA350E - Healthcare Analytics\\HealthcareProject-2\\Data\\IHME-GBD_2021_DATA-11a14e86-1.csv")
    filtered_alcohol = pd.read_csv("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/Risk_Adjusted/Alcohol.csv")
    filtered_obese = pd.read_csv("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/Risk_Adjusted/Obesity.csv")
    filtered_sanitation = pd.read_csv("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/Risk_Adjusted/Safe%20Sanitation.csv")
    filtered_smoking = pd.read_csv("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/Risk_Adjusted/Smoking.csv")
    data_yll = pd.read_csv("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/IHME-GBD_2021_DATA-5012c6c5-1.csv")
    data_doctors = pd.read_csv("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/Doctors.csv")
    data_nurses = pd.read_csv("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/Nurses.csv")
    data_wdi = pd.read_csv("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/WDI_Adjusted.csv")
    data_financial_hardships = pd.read_csv("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/financial_hardships.csv")
    return data_mena, data_lebanon_saudi, filtered_alcohol, filtered_obese, filtered_sanitation, filtered_smoking, data_yll, data_doctors, data_nurses, data_wdi, data_financial_hardships


@st.cache_data
def load_shapefile():
    mena_shapefile = gpd.read_file("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/MENA.geo.json")
    return mena_shapefile

mena_shapefile = load_shapefile()

data_mena, data_lebanon_saudi, filtered_alcohol, filtered_obese, filtered_sanitation, filtered_smoking, data_yll, data_doctors, data_nurses, data_wdi, data_financial_hardships = load_data()

# Filter data for neoplasms in Lebanon and remove "Occupational carcinogens"
neoplasms_data_lebanon = data_lebanon_saudi[(data_lebanon_saudi['location_name'] == 'Lebanon') &
                                            (data_lebanon_saudi['cause_name'] == 'Neoplasms') &
                                            (data_lebanon_saudi['rei_name'] != 'Occupational carcinogens')]

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Home Page
# Home Page
# Home Page
if selected_page == "Home":
    # Custom CSS for styling
    st.markdown("""
        <style>
        .big-font {
            font-size:22px !important;
        }
        .center-text {
            text-align: center;
        }
        .header-font {
            font-size: 42px !important;
        }
        .subheader-font {
            font-size: 27px !important;
        }
        .content-font {
            font-size: 22px !important;
        }
        ul {
            font-size: 22px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Create three columns
    col1, col2, col3 = st.columns([1, 0.5, 1])

    with col1:
        st.write("")

    with col2:
        # Display the logo in the middle column
        st.image("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/AUBlogo.png", use_column_width=True)

    with col3:
        st.write("")

    # Page title and introduction
    st.markdown("<h1 class='header-font center-text'>Healthcare Analysis of Causes of Death and Their Risk Factors in the Middle East</h1>", unsafe_allow_html=True)
    st.markdown("<div class='center-text subheader-font'>Welcome to the healthcare analysis project focused on the Middle East, with a particular focus on neoplasms in Lebanon. This project aims to provide insights into the causes of death and their associated risk factors.</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    # Overview Section
    st.markdown("<h2 class='subheader-font'>Project Objective</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class='content-font'>
        The objective of this project is to analyze healthcare data to understand the causes of death and their risk factors in the Middle East. Through a detailed analysis, with a particular focus on neoplasms, we aim to provide actionable insights for decision-makers to address the critical health issues in the region.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='subheader-font'>Key Insights</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class='content-font'>
        <ul>
            <li><strong>High Death Rate Due to Neoplasms in Lebanon:</strong> Lebanon has an unusually high death rate due to neoplasms compared to other MENA countries.</li>
            <li><strong>Increasing Trend of Neoplasm Deaths:</strong> The trend of deaths due to neoplasms in Lebanon shows an alarming increase over the years.</li>
            <li><strong>Significant Impact of Smoking:</strong> Smoking is identified as the highest contributing risk factor for neoplasms.</li>
            <li><strong>Forecast of Neoplasm Deaths:</strong> The forecast indicates a continuing increase in deaths due to neoplasms until 2030.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='subheader-font'>Navigation Guide</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class='content-font'>
        <ul>
            <li><strong>General MENA View:</strong> Overview of death rates across MENA countries.</li>
            <li><strong>Lebanon Analysis:</strong> Detailed analysis of neoplasms in Lebanon.</li>
            <li><strong>Models:</strong> Forecasting models for neoplasm deaths.</li>
            <li><strong>Lebanon and Saudi Arabia:</strong> Comparative analysis of risk factors and healthcare systems between Lebanon and Saudi Arabia.</li>
            <li><strong>Recommendations:</strong> Actionable insights and recommendations.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    # Footer Section
    st.markdown("---")
    st.markdown("""
        <div class='content-font'>
        <strong>Created by:</strong>
        <ul>
            <li>Kinan Murad</li>
            <li>Romanos Rizk</li>
            <li>Sasha Nasser</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; font-size:20px; color:gray;">
        Healthcare Analytics - Final Project by Kinan Murad, Romanos Rizk, and Sasha Nasser
    </div>
    """, unsafe_allow_html=True)
    
#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------

# General MENA View Page
elif selected_page == "General MENA View":
        # Custom CSS for styling
    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
        }
        .center-text {
            text-align: center;
        }
        .header-font {
            font-size: 40px !important;
        }
        .subheader-font {
            font-size: 25px !important;
        }
        .content-font {
            font-size: 20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
        # Create three columns
    col1, col2, col3 = st.columns([1, 0.5, 1])

    with col1:
        st.write("")

    with col2:
        # Display the logo in the middle column
        st.image("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/AUBlogo.png", use_column_width=True)

    with col3:
        st.write("")

    # Page Title
    st.markdown("<h1 class='header-font center-text'>General MENA View: Exploring the Death Rates Across the MENA Region for Various Causes of Death</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filter selection
    cause = st.selectbox("Select Cause of Death", data_mena['cause_name'].unique())
    year = st.slider("Select Year", int(data_mena['year'].min()), int(data_mena['year'].max()), step=1)

    # Filter data
    filtered_data = data_mena[(data_mena['cause_name'] == cause) & (data_mena['year'] == year)]

    # Merging filtered data with shapefile data
    merged_data = mena_shapefile.merge(filtered_data, left_on='name', right_on='location_name', how='left')

    # Map Visualization with Explanation
    st.write("### Geographic Distribution of Death Rates Due to Neoplasms")
    col1, col2 = st.columns([2, 2])

    with col1:
        # Map Visualization using Folium
        m = folium.Map(location=[25, 45], zoom_start=4)

        # Define the color scale
        colormap = linear.Blues_09.scale(filtered_data['val'].min(), filtered_data['val'].max())
        colormap.caption = 'Rate of Death (per 100,000)'

        # Adding the merged GeoDataFrame to the map with a color scale and tooltips
        folium.Choropleth(
            geo_data=merged_data,
            data=filtered_data,
            columns=['location_name', 'val'],
            key_on='feature.properties.name',
            fill_color='Blues',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Rate of Death (per 100,000)',
            highlight=True,
            reset=True
        ).add_to(m)

        # Add tooltips to show the rate on hover
        folium.GeoJson(
            merged_data,
            style_function=lambda feature: {
                'fillColor': colormap(feature['properties']['val']) if feature['properties']['val'] else 'gray',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['name', 'val'],
                aliases=['Country:', 'Rate:'],
                localize=True
            )
        ).add_to(m)

        # Add colormap to the map
        colormap.add_to(m)

        # Display the map
        st_folium(m, width=700, height=500)

    with col2:
        st.write("#### Explanation:")
        st.markdown("""
            <div style="font-size: 20px;">
            Here, we explore the spatial distribution of death rates across MENA countries for the selected cause and year.
            The map highlights the varying death rates with a color gradient, where darker colors represent higher death rates.
            <br><br> It is evident from the map that Lebanon has a significantly higher death rate due to neoplasms compared to other MENA countries.
            This observation raises a critical concern and prompts further investigation into the underlying factors contributing to the high neoplasm mortality rate in Lebanon.
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Trend Analysis with Explanation
    st.write("### Trend Analysis: Compare the Trends of Death Rates for the Selected Cause Across MENA Countries")
    col3, col4 = st.columns([2, 2])

    with col3:
        # Filter data for the selected cause across all years (ignore the year slider for this)
        trend_data = data_mena[(data_mena['cause_name'] == cause) & (data_mena['location_name'] != 'Turkey')]

        # Aggregate the data to ensure one line per country
        aggregated_trend_data = trend_data.groupby(['location_name', 'year'], as_index=False)['val'].mean()

        # Create a Plotly figure
        fig = px.line(
            aggregated_trend_data,
            x='year',
            y='val',
            color='location_name',
            labels={'val': 'Death Rate (per 100,000)', 'year': 'Year', 'location_name': 'Country'},
            title=""
        )

        fig.update_traces(mode='lines+markers')
        fig.update_layout(legend_title_text='Country')

        # Display the Plotly figure
        st.plotly_chart(fig, use_container_width=False, width=700)

    with col4:
        st.write("#### Explanation:")
        st.markdown("""
            <div style="font-size: 20px;">
            This section provides a temporal view of death rates for the selected cause across all MENA countries.
            By observing trends over the years, we can identify patterns and changes in mortality rates, helping us understand how specific health issues evolve over time.
            <br><br> Notably, the trend analysis reveals that Lebanon consistently exhibits a higher death rate due to neoplasms compared to other MENA countries.
            This persistent high rate underscores the severity of neoplasms in Lebanon and emphasizes the need for targeted public health interventions and policies to address this critical issue.
            </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; font-size:20px; color:gray;">
        Healthcare Analytics - Final Project by Kinan Murad, Romanos Rizk, and Sasha Nasser
    </div>
    """, unsafe_allow_html=True)
#-------------------------------------------------------------------------------------------------------------------------

# Lebanon Analysis Page
elif selected_page == "Lebanon Analysis":
    # Custom CSS for styling
    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
        }
        .center-text {
            text-align: center;
        }
        .header-font {
            font-size: 40px !important;
        }
        .subheader-font {
            font-size: 25px !important;
        }
        .content-font {
            font-size: 20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
        # Create three columns
    col1, col2, col3 = st.columns([1, 0.5, 1])

    with col1:
        st.write("")

    with col2:
        # Display the logo in the middle column
        st.image("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/AUBlogo.png", use_column_width=True)

    with col3:
        st.write("")
    
    # Page Title
    st.markdown("<h1 class='header-font center-text'>A comprehensive analysis of specific causes and risk factors related to neoplasms in Lebanon</h1>", unsafe_allow_html=True)
    
    st.markdown("---")

    # Filter selection
    metrics = neoplasms_data_lebanon['metric_name'].unique()
    metrics = [metric for metric in metrics if metric != 'Percent']
    metric = st.selectbox("Select Metric", metrics)

    # Filter data by selected metric
    filtered_data = neoplasms_data_lebanon[neoplasms_data_lebanon['metric_name'] == metric]

    # Remove "Occupational carcinogens" risk factor
    filtered_data = filtered_data[filtered_data['rei_name'] != "Occupational carcinogens"]

    #st.markdown("---")

    # Line graph for trends over time
    st.markdown("<h2 class='subheader-font'>Trends in Death Rates Over Time for Neoplasms in Lebanon</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        line_data = filtered_data.groupby('year', as_index=False)['val'].sum()
        fig_trend = px.line(line_data, x='year', y='val',
                            labels={'val': 'Death Rate (per 100,000)', 'year': 'Year'})
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        st.write("#### Explanation:")
        st.markdown("""
            <div style="font-size: 20px;">
            The line graph above demonstrates the trends in death rates over time for neoplasms in Lebanon. <br><br> It shows a consistent increase in death rates over the years, highlighting the growing burden of neoplasms on the Lebanese population. <br><br> This persistent rise indicates a significant public health concern that requires urgent attention and intervention.
            </div>
        """, unsafe_allow_html=True)

    #st.markdown("---")

    # Clustered bar chart for age and gender distribution
    st.markdown("<h2 class='subheader-font'>Death Rates by Age Group and Gender</h2>", unsafe_allow_html=True)
    
    col3, col4 = st.columns([2, 2])

    with col3:
        aggregated_data = filtered_data.groupby(['age_name', 'sex_name'], as_index=False)['val'].sum()
        age_order = ["15-19 years", "20-24 years", "25-29 years", "30-34 years", "35-39 years", "40-44 years", 
                     "45-49 years", "50-74 years", "75+ years"]
        aggregated_data['age_name'] = pd.Categorical(aggregated_data['age_name'], categories=age_order, ordered=True)
        aggregated_data = aggregated_data.sort_values('age_name')
        fig_age_gender = px.bar(aggregated_data, x='age_name', y='val', color='sex_name', barmode='group',
                                labels={'val': 'Death Rate (per 100,000)', 'age_name': 'Age Group', 'sex_name': 'Gender'})
        st.plotly_chart(fig_age_gender, use_container_width=True)
    
    with col4:
        st.write("#### Explanation:")
        st.markdown("""
            <div style="font-size: 20px;">
            This bar chart illustrates the distribution of death rates by age group and gender for neoplasms in Lebanon. <br><br> It is evident that older age groups, particularly those aged 50-74 years and 75+ years, experience significantly higher death rates. <br><br> Additionally, males show higher mortality rates compared to females across most age groups. This demographic information is crucial for targeting prevention and intervention strategies.
            </div>
        """, unsafe_allow_html=True)

    #st.markdown("---")

    # Percentage of YLL Due to Neoplasms in Lebanon (2019)
    st.markdown("<h2 class='subheader-font'>Percentage of Years of Life Lost (YLL) Due to Neoplasms in Lebanon (2019)</h2>", unsafe_allow_html=True)
    
    col5, col6 = st.columns([2, 2])

    with col5:
        # Filter data for the year 2019
        yll_2019 = data_yll[data_yll['year'] == 2019]

        # Extract the val value
        percent_yll_2019 = yll_2019['val'].values[0] if not yll_2019.empty else None

        # Adjust the percentage value
        if percent_yll_2019 is not None:
            percent_yll_2019 *= 100

        # Display the value in Streamlit with a nicer format
        if percent_yll_2019 is not None:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; margin: 10px;">
                <h1 style="font-size: 48px; color: #87CEEB;">{percent_yll_2019:.2f}%</h1>
                <p>Percentage of YLL due to neoplasms out of all YLL in Lebanon due to all diseases</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("Data for 2019 is not available.")
    
    with col6:
        st.write("#### Explanation:")
        st.markdown("""
            <div style="font-size: 20px;">
            In 2019, neoplasms accounted for 22.32% of all Years of Life Lost (YLL) in Lebanon due to all diseases. This high percentage underscores the significant impact of neoplasms on the population's overall health and life expectancy. It highlights the urgent need for effective public health interventions and policies to reduce the burden of neoplasms.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Risk Factors Analysis Section
    st.markdown("<h2 class='subheader-font'>Risk Factors Analysis: Contribution of Various Risk Factors to Neoplasms</h2>", unsafe_allow_html=True)
    
    col7, col8 = st.columns([2, 2])

    with col7:
        risk_factor_data = filtered_data.groupby('rei_name', as_index=False)['val'].sum()
        risk_factor_data = risk_factor_data.sort_values(by='val')
        categories = risk_factor_data['rei_name'].tolist()
        values = risk_factor_data['val'].tolist()
        fig_spider = go.Figure()
        fig_spider.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Risk Factors'
        ))
        fig_spider.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values)]
                )),
            showlegend=False
        )
        st.plotly_chart(fig_spider, use_container_width=True)
    
    with col8:
        st.write("#### Explanation:")
        st.markdown("""
            <div style="font-size: 20px;">
            The spider chart illustrates the contribution of various risk factors to neoplasms in Lebanon. <br><br> Smoking emerges as the most significant risk factor, followed by high body-mass index, and diet low in fruits. <br><br> These findings emphasize the importance of targeting these risk factors in public health campaigns and intervention programs to mitigate the impact of neoplasms.
            </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; font-size:20px; color:gray;">
        Healthcare Analytics - Final Project by Kinan Murad, Romanos Rizk, and Sasha Nasser
    </div>
    """, unsafe_allow_html=True)
#-------------------------------------------------------------------------------------------------------------------

# Models Page
elif selected_page == "Models":
        # Custom CSS for styling
    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
        }
        .center-text {
            text-align: center;
        }
        .header-font {
            font-size: 40px !important;
        }
        .subheader-font {
            font-size: 25px !important;
        }
        .content-font {
            font-size: 20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
        # Create three columns
    col1, col2, col3 = st.columns([1, 0.5, 1])

    with col1:
        st.write("")

    with col2:
        # Display the logo in the middle column
        st.image("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/AUBlogo.png", use_column_width=True)

    with col3:
        st.write("")
    
    # Page Title
    st.markdown("<h1 class='header-font center-text'>Predictive Modeling: Statistical and predictive models applied to forecast the future trend of neoplasms death rate in Lebanon</h1>", unsafe_allow_html=True)
    #st.markdown("<div class='subheader-font center-text'>Statistical and predictive models applied to forecast the future trend of neoplasms death rate in Lebanon.</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load your data (assuming the data is already loaded into neoplasms_data_lebanon DataFrame)
    # Ensure 'year' and 'val' columns exist in the DataFrame
    # Group the data by year to ensure each year has only one row
    neoplasms_data_lebanon = neoplasms_data_lebanon.groupby('year')['val'].sum().reset_index()

    # Set the index to 'year'
    neoplasms_data_lebanon['year'] = pd.to_datetime(neoplasms_data_lebanon['year'], format='%Y')
    neoplasms_data_lebanon.set_index('year', inplace=True)

    # Split the data into training and testing sets (90% train, 10% test)
    train_size = int(len(neoplasms_data_lebanon) * 0.9)
    train, test = neoplasms_data_lebanon[:train_size], neoplasms_data_lebanon[train_size:]

    # Fit an ARIMA model on the training set
    model = ARIMA(train['val'], order=(1, 1, 1))
    fit_model = model.fit()

    # Forecast the values for the test period
    forecast = fit_model.forecast(steps=len(test))
    test['forecast'] = forecast

    # Calculate accuracy measures
    rmse = np.sqrt(mean_squared_error(test['val'], test['forecast']))
    mean_val = test['val'].mean()  # Mean of the actual values in the test set
    rmse_percentage = (rmse / mean_val) * 100

    # Fit the ARIMA model on the entire dataset
    model_full = ARIMA(neoplasms_data_lebanon['val'], order=(1, 1, 1))
    fit_model_full = model_full.fit()

    # Calculate the number of years to forecast until 2030
    last_year_in_data = neoplasms_data_lebanon.index.year.max()
    forecast_years = 2030 - last_year_in_data

    # Create the `start` and `end` parameters for the prediction
    start = len(neoplasms_data_lebanon)
    end = start + forecast_years - 1

    # Forecast the values until 2030
    forecast_full = fit_model_full.get_prediction(start=start, end=end)
    forecast_values_full = forecast_full.predicted_mean
    conf_int_full = forecast_full.conf_int()

    # Create a forecast index with datetime values
    forecast_index_full = pd.date_range(start=neoplasms_data_lebanon.index[-1] + pd.DateOffset(years=1), periods=forecast_years, freq='Y')

    # Convert the forecast index to a simpler date format for display
    forecast_index_display = [date.strftime('%Y') for date in forecast_index_full]

    # Plot the historical data, the test forecast, and the extended forecast
    col1, col2, col3 = st.columns([4, 3, 2])
    
    with col1:
        plt.figure(figsize=(8, 5))
        plt.plot(train.index, train['val'], label='Train')
        plt.plot(test.index, test['val'], label='Test')
        plt.plot(test.index, test['forecast'], label='Test Forecast')
        plt.plot(neoplasms_data_lebanon.index, neoplasms_data_lebanon['val'], label='Historical', linestyle='--')
        plt.plot(forecast_index_full, forecast_values_full, label='Forecast until 2030')
        plt.fill_between(forecast_index_full, conf_int_full.iloc[:, 0], conf_int_full.iloc[:, 1], color='k', alpha=0.1)
        plt.title('Forecast of Neoplasms Death Rate in Lebanon (until 2030)')
        plt.xlabel('Year')
        plt.ylabel('Death Rate')
        plt.legend()
        st.pyplot(plt)
    
    with col3:
        st.write("### Forecast Values until 2030:")
        # Create a DataFrame for the forecasted values with the simpler date format
        forecast_df = pd.DataFrame({
            'Year': forecast_index_display,
            'Forecasted Death Rate': forecast_values_full
        })
        st.write(forecast_df)

    # Model Performance Section
    st.markdown("---")
    st.write("### Model Performance")
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; margin: 10px;">
            <h1 style="font-size: 48px; color: #87CEEB;">{rmse:.2f}</h1>
            <p>RMSE</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; margin: 10px;">
            <h1 style="font-size: 48px; color: #87CEEB;">{mean_val:.2f}</h1>
            <p>Mean Value</p>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; margin: 10px;">
            <h1 style="font-size: 48px; color: #87CEEB;">{rmse_percentage:.2f}%</h1>
            <p>RMSE as % of Mean</p>
        </div>
        """, unsafe_allow_html=True)

    # Explanation Section
    st.markdown("---")
    st.write("#### Explanation:")
    st.markdown("""
        <div style="font-size: 22px;">
        This section presents the results of predictive modeling applied to the data on neoplasms death rates in Lebanon. 
        The ARIMA model was used to forecast future death rates up to the year 2030. 
        The chart illustrates the historical data, the model's fit on the training data, the forecasted values for the test period, and the extended forecast until 2030.
        <br><br>
        Key observations from the model:
        <ul>
            <li>The RMSE value indicates the model's prediction error, which is reasonably low compared to the mean value of the test set.</li>
            <li>The forecast indicates a continued increase in the death rate due to neoplasms, highlighting a significant public health concern.</li>
            <li>The confidence intervals provide a range within which the actual values are expected to fall, emphasizing the model's reliability.</li>
        </ul>
        These insights underscore the urgent need for targeted interventions and policies to address the rising trend of neoplasms in Lebanon.
        </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; font-size:20px; color:gray;">
        Healthcare Analytics - Final Project by Kinan Murad, Romanos Rizk, and Sasha Nasser
    </div>
    """, unsafe_allow_html=True)
#-------------------------------------------------------------------------------------------------------------------
# Lebanon and Saudi Arabia Comparison Page
elif selected_page == "Lebanon and Saudi Arabia":
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
        }
        .center-text {
            text-align: center;
        }
        .header-font {
            font-size: 40px !important;
        }
        .subheader-font {
            font-size: 25px !important;
        }
        .content-font {
            font-size: 20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
        # Create three columns
    col1, col2, col3 = st.columns([1, 0.5, 1])

    with col1:
        st.write("")

    with col2:
        # Display the logo in the middle column
        st.image("https://raw.githubusercontent.com/Romanos-Rizk/MSBA350E/main/AUBlogo.png", use_column_width=True)

    with col3:
        st.write("")
    
    # Page Title
    st.markdown("<h1 class='header-font center-text'>Comparative Analysis: Exploring the differences in risk factors, healthcare systems, and financial hardships between Lebanon and Saudi Arabia</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Risk Factor Comparison
    st.markdown("<h2 class='subheader-font'>Risk Factor Comparison: Analyzing the progression of various risk factors in Lebanon and Saudi Arabia.</h2>", unsafe_allow_html=True)
    risk_factor = st.selectbox("Select Risk Factor", ["Obesity", "Smoking", "Alcohol Consumption", "Safe Sanitation"])
    
    datasets = {
        "Obesity": filtered_obese,
        "Smoking": filtered_smoking,
        "Alcohol Consumption": filtered_alcohol,
        "Safe Sanitation": filtered_sanitation
    }

    color_palette = {
        "Lebanon": "#56B4E9",
        "Saudi Arabia": "#0072B2"
    }

    data = datasets[risk_factor]
    filtered_data = data[data['GEO_NAME_SHORT'].isin(['Lebanon', 'Saudi Arabia'])]
    filtered_data['DIM_TIME'] = filtered_data['DIM_TIME'].astype(int)

    col1, col2 = st.columns([2, 2])

    with col1:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=filtered_data, x='DIM_TIME', y='METRIC', hue='GEO_NAME_SHORT', palette=color_palette)
        plt.title(f'Progression of {risk_factor} in Lebanon vs Saudi Arabia')
        plt.xlabel('Year')
        plt.ylabel(f'{risk_factor} Rate')
        plt.legend(title='Country')
        sns.despine()
        st.pyplot(plt)

    with col2:
        st.write("### Explanation")
        st.markdown("""
            <div class='content-font'>
            This chart illustrates the progression of various risk factors over time in Lebanon and Saudi Arabia. By comparing these trends, we can gain insights into the health challenges faced by each country and the effectiveness of their public health interventions.
            <br><br>
        """, unsafe_allow_html=True)
        if risk_factor == "Smoking":
            st.markdown("""
                <div class='content-font'>
                Lebanon has a significantly higher smoking rate compared to Saudi Arabia. This is a major concern as smoking is the highest contributing risk factor for neoplasms. 
                The consistently high smoking rates in Lebanon indicate a pressing need for more effective smoking cessation programs and public health campaigns to reduce tobacco use.
                </div>
            """, unsafe_allow_html=True)
        elif risk_factor == "Obesity":
            st.markdown("""
                <div class='content-font'>
                Both countries have increasing obesity rates, but Lebanon's rate is higher. This trend is alarming as obesity is a risk factor for various chronic diseases, including neoplasms. 
                The upward trend in obesity rates suggests a growing public health issue that requires urgent attention through policies promoting healthy diets and physical activity.
                </div>
            """, unsafe_allow_html=True)
        elif risk_factor == "Alcohol Consumption":
            st.markdown("""
                <div class='content-font'>
                Lebanon has a higher alcohol consumption rate compared to Saudi Arabia. Higher rates of alcohol consumption in Lebanon pose additional health risks, including liver diseases and certain types of cancer. 
                The stark contrast in alcohol consumption rates reflects cultural and regulatory differences between the two countries.
                </div>
            """, unsafe_allow_html=True)
        elif risk_factor == "Safe Sanitation":
            st.markdown("""
                <div class='content-font'>
                Saudi Arabia has consistently high safe sanitation rates, while Lebanon has gradually improved. Better sanitation in Lebanon could lead to improved health outcomes by reducing the prevalence of waterborne diseases and other sanitation-related health issues.
                The disparity in sanitation rates highlights the importance of continued infrastructure development and public health initiatives in Lebanon.
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Healthcare Professional Density Comparison
    st.markdown("<h2 class='subheader-font'>Healthcare Professional Density: Comparing the density of doctors and nurses in Lebanon and Saudi Arabia over the years.</h2>", unsafe_allow_html=True)
    profession = st.selectbox("Select Profession", ["Doctors", "Nurses"])

    datasets = {
        "Doctors": data_doctors,
        "Nurses": data_nurses
    }

    data = datasets[profession]
    data['DIM_TIME'] = data['DIM_TIME'].astype(int)

    col3, col4 = st.columns([2, 2])

    with col3:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=data, x='DIM_TIME', y='RATE_PER_10000_N', hue='GEO_NAME_SHORT', palette='Blues')
        plt.title(f'Progression of {profession} Density Through the Years')
        plt.xlabel('Year')
        plt.ylabel(f'{profession} Density (per 10,000)')
        plt.legend(title='Country')
        sns.despine()
        st.pyplot(plt)

    with col4:
        st.write("### Explanation")
        st.markdown("""
            <div style='font-size: 20px;'>
            This section compares the density of healthcare professionals in Lebanon and Saudi Arabia over the years.
            Understanding the availability of healthcare professionals in each country helps us gauge the capacity of their healthcare systems.
            <br><br>
            </div>
        """, unsafe_allow_html=True)

        if profession == "Doctors":
            st.markdown("""
                <div style='font-size: 20px;'>
                The density of doctors has increased in both countries over the years. 
                However, Saudi Arabia shows a more rapid increase in doctors' density, reflecting significant investments in healthcare infrastructure and workforce.
                </div>
            """, unsafe_allow_html=True)
        elif profession == "Nurses":
            st.markdown("""
                <div style='font-size: 20px;'>
                The density of nurses has shown a steady increase in both countries. 
                Saudi Arabia has a noticeably much higher density of nurses compared to Lebanon, indicating better healthcare infrastructure and workforce capacity.
                Lebanon, however, has had more fluctuations compared to Saudi Arabia, indicating potential challenges in maintaining a stable nursing workforce.
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")


    # Health Expenditure Trends
    st.markdown("<h2 class='subheader-font'>Health Expenditure Trends: Analyzing the trends in health expenditure metrics in Lebanon and Saudi Arabia.</h2>", unsafe_allow_html=True)

    # Ensure the necessary data processing for `melted_data` is included
    id_vars = ['Country Name', 'Indicator Name']
    value_vars = [col for col in data_wdi.columns if col not in id_vars + ['Country Code']]

    melted_data = pd.melt(data_wdi, id_vars=id_vars, value_vars=value_vars, var_name='Year', value_name='Value')
    melted_data['Year'] = pd.to_numeric(melted_data['Year'], errors='coerce')
    melted_data = melted_data.dropna(subset=['Year', 'Value'])
    melted_data['Year'] = melted_data['Year'].astype(int)

    country = st.selectbox("Select Country", melted_data['Country Name'].unique())

    filtered_data = melted_data[melted_data['Country Name'] == country]

    col5, col6 = st.columns([2, 2])

    with col5:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=filtered_data, x='Year', y='Value', hue='Indicator Name', style='Indicator Name', markers=False, dashes=False, palette='Blues')
        plt.title(f'Trend of Health Expenditure Metrics in {country}')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.legend(title='Indicator Name', fontsize='small', title_fontsize='small')
        sns.despine()
        st.pyplot(plt)

    with col6:
        st.write("### Explanation")
        st.markdown("""
            <div style='font-size: 20px;'>
            This section presents the trends in various health expenditure metrics. The data provides insights into how each country allocates resources for healthcare and highlights any significant changes over the years.
            <br><br>
            </div>
        """, unsafe_allow_html=True)
        
        if country == "Lebanon":
            st.markdown("""
                <div style='font-size: 20px;'>
                In Lebanon, the data shows a significant reliance on private health expenditure. This indicates that the healthcare system in Lebanon is predominantly private, which may limit access to healthcare services for lower-income individuals.
                </div>
            """, unsafe_allow_html=True)
        elif country == "Saudi Arabia":
            st.markdown("""
                <div style='font-size: 20px;'>
                In Saudi Arabia, the data shows a high level of government health expenditure. This reflects a robust public healthcare system, providing more accessible healthcare services to the population compared to Lebanon.
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")


    # Financial Hardship Comparison
    st.markdown("<h2 class='subheader-font'>Financial Hardship Comparison: Comparing the financial burden of healthcare on households in Lebanon and Saudi Arabia.</h2>", unsafe_allow_html=True)

    lebanon_2012 = data_financial_hardships[(data_financial_hardships['GEO_NAME_SHORT'] == 'Lebanon') & (data_financial_hardships['DIM_TIME'] == 2012)]
    saudi_2013 = data_financial_hardships[(data_financial_hardships['GEO_NAME_SHORT'] == 'Saudi Arabia') & (data_financial_hardships['DIM_TIME'] == 2013)]

    percent_pop_lebanon_2012 = lebanon_2012['PERCENT_POP_N'].values[0] if not lebanon_2012.empty else None
    percent_pop_saudi_2013 = saudi_2013['PERCENT_POP_N'].values[0] if not saudi_2013.empty else None

    col7, col8 = st.columns([2, 2])

    with col7:
        if percent_pop_lebanon_2012 is not None:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; margin: 10px;">
                <h2>Lebanon, 2012</h2>
                <h1 style="font-size: 48px; color: #87CEEB;">{percent_pop_lebanon_2012:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("Data for Lebanon (2012) is not available.")

    with col8:
        if percent_pop_saudi_2013 is not None:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; margin: 10px;">
                <h2>Saudi Arabia, 2013</h2>
                <h1 style="font-size: 48px; color: #0072B2;">{percent_pop_saudi_2013:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("Data for Saudi Arabia (2013) is not available.")

    st.markdown("### Explanation")
    st.markdown("""
        <div style='font-size: 20px;'>
        This section compares the financial hardship caused by healthcare expenditures in Lebanon and Saudi Arabia. 
        The percentages represent the proportion of households spending more than 10% of their total budget on healthcare. 
        A higher percentage indicates a greater financial burden on the population.
        <br><br>
        In Lebanon, a significant 26.60% of households face this financial burden, which can lead to many individuals potentially skipping necessary medical tests. 
        This lack of early detection can result in complications or even death from diseases like neoplasms, as early diagnosis is crucial for better health outcomes.
        <br><br>
        In contrast, only 1.73% of households in Saudi Arabia face this level of financial hardship, indicating better access to healthcare services without overwhelming financial strain.
        </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; font-size:20px; color:gray;">
        Healthcare Analytics - Final Project by Kinan Murad, Romanos Rizk, and Sasha Nasser
    </div>
    """, unsafe_allow_html=True)
#-------------------------------------------------------------------------------------------------------------------------





#-------------------------------------------------------------------------------------------------------------------------

# Recommendations Page
elif selected_page == "Recommendations":
    st.title("Recommendations")
    st.write("Final recommendations and conclusions based on the analysis.")


    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 10px; right: 10px; font-size:20px; color:gray;">
        Healthcare Analytics - Final Project by Kinan Murad, Romanos Rizk, and Sasha Nasser
    </div>
    """, unsafe_allow_html=True)