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

@st.cache_data
def get_eust_data(dataframe: str):
    data = eust.get_data_df(dataframe)

    df = pd.DataFrame(data)
    
    # Rename the column 'geo\\TIME_PERIOD' to 'geo'
    df.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

    # Drop the column 'freq'
    df = df.drop(columns=['freq'])
    
    # Melt the dataframe to long format
    df_melted = pd.melt(df, id_vars=['siec', 'unit', 'geo'], var_name='year_month', value_name='value')
    
    # Pivot the dataframe so that the combined 'siec_unit' values become columns
    df_pivoted = df_melted.pivot_table(index=['year_month', 'geo'], columns='siec', values='value').reset_index()
    
    # Flatten the columns after pivoting
    df_pivoted.columns.name = None

    # Convert 'year_month' to datetime objects
    df_pivoted['year_month'] = pd.to_datetime(df_pivoted['year_month'])
    
    # Extract date part (datetime.date objects)
    df_pivoted['year_month'] = df_pivoted['year_month'].dt.date

    # Create a dictionary mapping siec to unit
    siec_unit_dict = df_melted.set_index('siec')['unit'].to_dict()
    
    # Display the transformed dataframe
    return df_pivoted, siec_unit_dict

@st.cache_data
def get_nuts():
    DATA_FILENAME = Path(__file__).parent/'data/NUTS_RG_20M_2021_4326.geojson'
    nuts = gpd.read_file(DATA_FILENAME)
    nuts = nuts.query('LEVL_CODE == 0')
    return nuts

@st.cache_data
def dic_units(df_name):
    dic_units = eust.get_dic(df_name, 'siec', frmt='df')

    # create dictionary
    descr_to_val = dict(zip(dic_units['descr'], dic_units['val']))

    return dic_units, descr_to_val


@st.cache_data
def dic_countries(df_name):
    dic_countries = eust.get_dic(df_name, 'geo', frmt='df')

    # create dictionary
    country_to_code = dict(zip(dic_countries['descr'], dic_countries['val']))

    return dic_countries, country_to_code

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :electric_plug: Energy in the EU :flag-eu:

Browse energy data from the [eurostat Database](https://ec.europa.eu/eurostat/data/database). This data is updated monthly by eurostat and queried via API.
'''

# Add some spacing
''
''

# Create two columns
col1, col2 = st.columns(2)



with st.sidebar:

    df_name = 'nrg_cb_pem'

    df_eust, dic_eust = get_eust_data(df_name)
    
    from_date, to_date = st.slider(
        'Which dates are you interested in?',
        min_value=min(df_eust['year_month']),
        max_value=max(df_eust['year_month']),
        value=(min(df_eust['year_month']), max(df_eust['year_month'])),
        format="YYYY-MM"
    )
    #st.write(f'from: {from_date} to: {to_date}.')
    from_date = from_date.replace(day=1)
    to_date = to_date.replace(day=1)

    #st.write(f'NEW: from: {from_date} to: {to_date}.')
    
    countries = df_eust['geo'].unique().tolist()

    # st.write(countries)
    
    dic_countries, country_to_code = dic_countries(df_name)

    filtered_dic_countries = dic_countries[dic_countries['val'].isin(countries)]

    filtered_country_to_code = dict(zip(filtered_dic_countries['descr'], filtered_dic_countries['val']))
    
    if not len(countries):
        st.warning("Select at least one country")
    
    selected_countries = st.multiselect(
        'Which countries would you like to view?',
        filtered_dic_countries['descr'],
        filtered_dic_countries['descr']
    )

    st.write(selected_countries)
    
    selected_countries_code = [filtered_country_to_code[descr] for descr in selected_countries if descr in filtered_country_to_code]

    
    filtered_df_eust = df_eust[
        (df_eust['geo'].isin(selected_countries_code))
        & (df_eust['year_month'] <= to_date)
        & (from_date <= df_eust['year_month'])
    ]

    # create dictionary
    dic_units, descr_to_val = dic_units(df_name)
    
    # Exclude 'geo' and 'year_month' from the available siec values
    available_siec_values = df_eust.columns.difference(['geo', 'year_month']).tolist()
    
    # Filter dic_units based on available siec values
    filtered_dic_units = dic_units[dic_units['val'].isin(available_siec_values)]

    # create dictionary
    filtered_descr_to_val = dict(zip(filtered_dic_units['descr'], filtered_dic_units['val']))
    
    #st.write(filtered_dic_units)
    
    picked_unit_name = st.selectbox(
        'Which energy source would you like to view?',
        filtered_dic_units['descr'],
        index=filtered_dic_units['descr'].tolist().index('Total')
    )

    #st.write(picked_unit_name)
    
    picked_unit = filtered_descr_to_val[picked_unit_name]

    #st.write(picked_unit)
    
# Add content to the first column
with col1:
    
    
    st.header("Map")


    #st.write(df_eust)
    #st.write(dic_eust)
    #st.write(filtered_df_eust)
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
    #st.write(oneYear_df_eust)
    merged = nuts.merge(oneYear_df_eust, left_on='CNTR_CODE', right_on='geo')
    #st.write(merged)
    # Ensure the GeoDataFrame contains only necessary columns
    merged = merged[['CNTR_CODE', picked_unit, 'geometry']]
    #st.write(merged)
    # Convert GeoDataFrame to GeoJSON
    geojson_data = merged.to_json()
    #st.write(geojson_data)
    # Create a base map
    m = folium.Map(location=[55.00, 13.0], zoom_start=3)
    
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

