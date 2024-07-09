import streamlit as st
import pandas as pd
import math
from pathlib import Path
import eurostat as eust

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

    # Convert year_month to timestamps (seconds since epoch)
    df_pivoted['timestamp'] = pd.to_datetime(df_pivoted['year_month'])
    
    # Display the transformed dataframe
    return df_pivoted

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

# Add content to the first column
with col1:
    st.header("OLD GDP-stuff")
    gdp_df = get_gdp_data()

    min_value = gdp_df['Year'].min()
    max_value = gdp_df['Year'].max()
    
    from_year, to_year = st.slider(
        'Which years are you interested in?',
        min_value=min_value,
        max_value=max_value,
        value=[min_value, max_value])
    
    countries = gdp_df['Country Code'].unique()
    
    if not len(countries):
        st.warning("Select at least one country")
    
    selected_countries = st.multiselect(
        'Which countries would you like to view?',
        countries,
        ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])
    
    ''
    ''
    ''
    
    # Filter the data
    filtered_gdp_df = gdp_df[
        (gdp_df['Country Code'].isin(selected_countries))
        & (gdp_df['Year'] <= to_year)
        & (from_year <= gdp_df['Year'])
    ]
    
    st.header('GDP over time', divider='gray')
    
    ''
    
    st.line_chart(
        filtered_gdp_df,
        x='Year',
        y='GDP',
        color='Country Code',
    )
    
    ''
    ''
    
    
    first_year = gdp_df[gdp_df['Year'] == from_year]
    last_year = gdp_df[gdp_df['Year'] == to_year]
    
    st.header(f'GDP in {to_year}', divider='gray')
    
    ''
    
    cols = st.columns(4)
    
    for i, country in enumerate(selected_countries):
        col = cols[i % len(cols)]
    
        with col:
            first_gdp = first_year[gdp_df['Country Code'] == country]['GDP'].iat[0] / 1000000000
            last_gdp = last_year[gdp_df['Country Code'] == country]['GDP'].iat[0] / 1000000000
    
            if math.isnan(first_gdp):
                growth = 'n/a'
                delta_color = 'off'
            else:
                growth = f'{last_gdp / first_gdp:,.2f}x'
                delta_color = 'normal'
    
            st.metric(
                label=f'{country} GDP',
                value=f'{last_gdp:,.0f}B',
                delta=growth,
                delta_color=delta_color
            )

# Add content to the second column
with col2:
    st.header("Energy-Contet-Test")
    eust_df = get_eust_data()

    # Set up the Streamlit slider
    from_date, to_date = st.slider(
        'Which dates are you interested in?',
        min_value=eust_df['timestamp'].min(),
        max_value=eust_df['timestamp'].max(),
        value=[eust_df['timestamp'].min(), eust_df['timestamp'].max()],
        format="YYYY-MM-DD"
    )
    
    # Display the selected dates
    st.write(f'From date: {from_date.date()} to date: {to_date.date()}')

    st.line_chart(eust_df, x='year_month', y='C0000_GWH',color='geo')
    
