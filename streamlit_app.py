import streamlit as st
import pandas as pd
import geopandas as gpd
import math
from pathlib import Path
import eurostat as eust
from datetime import datetime
import pydeck as pdk
import folium
from streamlit_folium import folium_static
from streamlit_folium import st_folium
import streamlit.components.v1 as components

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='EU energy dashboard',
    page_icon=':electric_plug:', # This is an emoji shortcode. Could be a URL too.
    layout="wide",
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

def get_eust_data():
    data = eust.get_data_df('nrg_cb_pem')

    df = pd.DataFrame(data)
    
    # Rename the column 'geo\\TIME_PERIOD' to 'geo'
    df.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

    # Drop the column 'freq'
    df = df.drop(columns=['freq'])
    
    # Melt the dataframe to long format
    df_melted = pd.melt(df, id_vars=['siec', 'unit', 'geo'], var_name='year_month', value_name='value')
    
    # Create a combined column for 'siec' and 'unit' to handle different units
    df_melted['siec_unit'] = df_melted['siec'] + '_' + df_melted['unit']
    
    # Pivot the dataframe so that the combined 'siec_unit' values become columns
    df_pivoted = df_melted.pivot_table(index=['year_month', 'geo'], columns='siec_unit', values='value').reset_index()
    
    # Flatten the columns after pivoting
    df_pivoted.columns.name = None

    # Convert 'year_month' to datetime objects
    df_pivoted['year_month'] = pd.to_datetime(df_pivoted['year_month'])
    
    # Extract date part (datetime.date objects)
    df_pivoted['year_month'] = df_pivoted['year_month'].dt.date
    
    # Display the transformed dataframe
    return df_pivoted

@st.cache_data
def get_nuts():
    DATA_FILENAME = Path(__file__).parent/'data/NUTS_RG_20M_2021_4326.geojson'
    nuts = gpd.read_file(DATA_FILENAME)
    nuts = nuts.query('LEVL_CODE == 0')
    return nuts

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :electric_plug: Energy in the EU :flag-eu:

In Future: Browse energy data from the [eurostat Database](https://ec.europa.eu/eurostat/data/database). This data is updated quarterly by eurostat and queried via API.

For now it's still GDP data from the [World Bank Open Data](https://data.worldbank.org/).
'''

# Add some spacing
''
''

# Create two columns
col1, col2 = st.columns(2)
df_eust = get_eust_data()

with st.sidebar:

    from_date, to_date = st.slider(
        'Which dates are you interested in?',
        min_value=min(df_eust['year_month']),
        max_value=max(df_eust['year_month']),
        value=(min(df_eust['year_month']), max(df_eust['year_month'])),
        format="YYYY-MM"
    )

    countries = df_eust['geo'].unique()
    
    if not len(countries):
        st.warning("Select at least one country")
    
    selected_countries = st.multiselect(
        'Which countries would you like to view?',
        countries,
        ['CZ', 'FR', 'ES', 'DE'])

    filtered_df_eust = df_eust[
        (df_eust['geo'].isin(selected_countries))
        & (df_eust['year_month'] <= to_date)
        & (from_date <= df_eust['year_month'])
    ]

    picked_unit = 'C0000_GWH'

# Add content to the first column
with col1:
    
    
    st.header("Map")

    ## Sample data
    #data = {'lat': [37.76, 34.05], 'lon': [-122.4, -118.25]}
    #df = pd.DataFrame(data)
    
    # Display the map
    #st.map(df)

    # Define a PyDeck layer
    #layer = pdk.Layer(
    #    'HexagonLayer',
    #    df,
    #    get_position='[lon, lat]',
    #    auto_highlight=True,
    #    elevation_scale=50,
    #    pickable=True,
    #    elevation_range=[0, 3000],
    #    extruded=True,
    #    coverage=1
    #)
    
    # Set the viewport location
    #view_state = pdk.ViewState(
    #    longitude=-122.4,
    #    latitude=37.76,
    #    zoom=6,
    #    pitch=50
    #)
    
    # Render the map with PyDeck
    #st.pydeck_chart(pdk.Deck(
    #    layers=[layer],
    #    initial_view_state=view_state
    #))

    nuts = get_nuts()


    oneYear_df_eust = df_eust[
        (df_eust['geo'].isin(selected_countries))
        & (df_eust['year_month'] == to_date)
    ]
    merged = nuts.merge(oneYear_df_eust, left_on='CNTR_CODE', right_on='geo')

    # Ensure the GeoDataFrame contains only necessary columns
    merged = merged[['CNTR_CODE', picked_unit, 'geometry']]
    #st.write(merged)
    # Convert GeoDataFrame to GeoJSON
    geojson_data = merged.to_json()
    #st.write(geojson_data)
    # Create a base map
    m = folium.Map(location=[54.5260, 15.2551], zoom_start=4)
    
    # Add a choropleth layer to the map
    
    #folium.GeoJson(
    #    geo_data=geojson_data,
    #    name='NUTS Level 0',
    #    tooltip=folium.GeoJsonTooltip(fields=['CNTR_CODE'], aliases=['Country Code:'])
    #).add_to(m)
    
    
    
    folium.Choropleth(
        geo_data=geojson_data,
        data=merged,
        columns=['CNTR_CODE', picked_unit],
        key_on='feature.properties.CNTR_CODE',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Legend Name',
        tooltip=folium.GeoJsonTooltip(fields=['CNTR_CODE'], aliases=['Country Code:'])
    ).add_to(m)
    
    #folium_static(m)

    st_folium(m)
    # m.save('map.html')

    # Display the map in Streamlit using an iframe
    #components.iframe(src="map.html", width=700, height=500)
    #first_year = gdp_df[gdp_df['Year'] == from_year]
    #last_year = gdp_df[gdp_df['Year'] == to_year]
    
    #st.header(f'GDP in {to_year}', divider='gray')
    
    
    
    #cols = st.columns(4)
    
    #for i, country in enumerate(selected_countries):
        #col = cols[i % len(cols)]
    
        #with col:
         #   first_gdp = first_year[gdp_df['Country Code'] == country]['GDP'].iat[0] / 1000000000
          #  last_gdp = last_year[gdp_df['Country Code'] == country]['GDP'].iat[0] / 1000000000
    
           # if math.isnan(first_gdp):
           #     growth = 'n/a'
            #    delta_color = 'off'
           # else:
            #    growth = f'{last_gdp / first_gdp:,.2f}x'
            #    delta_color = 'normal'
    
            #st.metric(
             #   label=f'{country} GDP',
              #  value=f'{last_gdp:,.0f}B',
               # delta=growth,
                #delta_color=delta_color
            #)

# Add content to the second column
with col2:
    st.header("Carts")

    #min_date = df_eust['timestamp'].min()
    #max_date = df_eust['timestamp'].max()

    #st.write(f'From date: {from_date} to date: {to_date}')
    #st.write(df_eust)

    
    
    # Filter the data

    #st.write(filtered_df_eust)
    # Display the selected dates
    #st.write(f'From date: {from_date} to date: {to_date}')

    st.line_chart(filtered_df_eust, x='year_month', y=picked_unit,color='geo')


        
    # center on Liberty Bell, add marker
    #m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)
    #folium.Marker(
    #    [39.949610, -75.150282], popup="Liberty Bell", tooltip="Liberty Bell"
    #).add_to(m)
    
    # call to render Folium map in Streamlit
    #st_data = st_folium(m, width=725)

