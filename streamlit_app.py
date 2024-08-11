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
import xml.etree.ElementTree as ET
import requests
import plotly.express as px
import numpy as np

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='EU energy dashboard',
    page_icon=':electric_plug:', # This is an emoji shortcode. Could be a URL too.
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_eust_data(dataframe: str, lst_vars: list):
    data = eust.get_data_df(dataframe)

    df = pd.DataFrame(data)

    # Rename the column 'geo\\TIME_PERIOD' to 'geo'
    df.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

    # Melt the dataframe to long format
    df_melted = pd.melt(df, id_vars=lst_vars, var_name='datetime', value_name='value')
    
    # Convert 'date' to datetime objects
    df_melted['datetime'] = pd.to_datetime(df_melted['datetime'])

    # Extract date part (datetime.date objects)
    df_melted['date'] = df_melted['datetime'].dt.date
    
    # Display the transformed dataframe
    return df_melted

@st.cache_data
def per_capita(df):
    pop = eust.get_data_df('demo_gind', filter_pars={'indic_de': 'AVG'})
    pop = pop.drop(columns=['freq', 'indic_de'])
    pop.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

    for ind in df.index:
        year = str(df.loc[ind, 'datetime'].year)
        geo = df.loc[ind, 'geo']
        pop_filtered = pop[pop['geo'] == geo]
        
        pop_val = None  # Initialize pop_val in the outer scope

        if year in pop_filtered.columns and not pop_filtered[year].empty:
            pop_val = pop_filtered[year].iloc[0]
        else:
            for i in range(5):
                year_temp = year - i - 1
                if year_temp in pop_filtered.columns and not pop_filtered[year_temp].empty:
                    pop_val = pop_filtered[year_temp].iloc[0]
                    break

        if pop_val:
            df.loc[ind, 'value'] = df.loc[ind, 'value'] / pop_val
        else:
            df.loc[ind, 'value'] = None

    return df


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

@st.cache_data
def get_dic_df(df_name):
   
    dic_df = eust.get_dic(df_name, frmt='dict')
    
    for key in dic_df:
        temp = eust.get_dic(df_name, key, frmt='dict', full = False)
        temp_key = dic_df[key]
        temp_key.update({'pars': temp})
        dic_df.update({key: temp_key})

    return dic_df

@st.cache_data
def get_toc():

    def parse_xml_to_dict(element):
        data_dict = {}
        # Parse attributes of the current element
        data_dict.update(element.attrib)
    
        # Parse text content of the current element
        if element.text and element.text.strip():
            data_dict['text'] = element.text.strip()
    
        # Parse child elements
        children = list(element)
        if children:
            child_dict = {}
            for child in children:
                child_tag = child.tag.split('}')[-1]  # Strip namespace
                child_dict.setdefault(child_tag, []).append(parse_xml_to_dict(child))
            data_dict.update(child_dict)
    
        return data_dict
    
    def find_branch_with_code(element, target_code):
        for child in element:
            # Check if this is a <nt:code> element and if its text matches the target_code
            if child.tag.endswith('code') and child.text == target_code:
                return element
            # Recursively search in child elements
            result = find_branch_with_code(child, target_code)
            if result is not None:
                return result
        return None
    
    def extract_datasets(data_dict):
        datasets = {}
        
        def recursive_extract(element):
            # Check if the current element is of type "dataset"
            if 'type' in element and element['type'] == 'dataset':
                title = None
                # Extract required fields
                code = element.get('code')[0]['text']
                last_update = element.get('lastUpdate')[0]['text']
                last_modified = element.get('lastModified')[0]['text']
                data_start = element.get('dataStart')[0]['text']
                data_end = element.get('dataEnd')[0]['text']
                values = element.get('values')[0]['text']
                metadata = next((meta['text'] for meta in element.get('metadata', []) if meta.get('format') == 'html'), None)
                download_link = next((link['text'] for link in element.get('downloadLink', []) if link.get('format') == 'tsv'), None)
    
                # Find the title in English
                for title_element in element.get('title', []):
                    if title_element['language'] == 'en':
                        title = title_element['text']
                        break
                
                # If code is found, add to the datasets dictionary
                if code:
                    datasets[code] = {
                        'title': title,
                        'lastUpdate': last_update,
                        'lastModified': last_modified,
                        'dataStart': data_start,
                        'dataEnd': data_end,
                        'values': values,
                        'metadata': metadata,
                        'downloadLink': download_link
                    }
            
            # Recursively process child elements
            for key, value in element.items(): # why is key not used but written?
                if isinstance(value, list):
                    for child in value:
                        if isinstance(child, dict):
                            recursive_extract(child)
        
        # Start the recursion with the top-level dictionary
        recursive_extract(data_dict)
        return datasets
    
    # Function to get Eurostat TOC XML
    def get_eurostat_toc_xml():
        url = 'https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/xml'
        response = requests.get(url)
        response.raise_for_status()  # Ensure we notice bad responses
        return response.content

    def filter_dict_by_codes(data, codes):
        return {k: v for k, v in data.items() if k in codes}
        
    # Get XML data
    xml_data = get_eurostat_toc_xml()
     
    # Parse the XML data
    root = ET.fromstring(xml_data)
    
    # Find the branch with the target <nt:code> value
    target_code = 'nrg'
    target_branch = find_branch_with_code(root, target_code)
    
    # Parse the target branch to a dictionary if found
    if target_branch is not None:
        data_dict = parse_xml_to_dict(target_branch)
    else:
        data_dict = {}
    
    # Extract datasets from the parsed data
    datasets_dict = extract_datasets(data_dict)

    codes = ['nrg_chdd_m', 'nrg_cb_sffm', 'nrg_cb_oilm', 'nrg_cb_cosm', 'nrg_cb_gasm', 'nrg_cb_em', 'nrg_cb_eim',
'nrg_cb_pem', 'nrg_t_m', 'nrg_ti_m', 'nrg_ti_oilm', 'nrg_ti_gasm', 'nrg_ti_coifpm', 'nrg_te_m', 'nrg_te_oilm',
'nrg_te_gasm', 'nrg_stk_m', 'nrg_stk_oilm', 'nrg_stk_oom', 'nrg_stk_oam', 'nrg_stk_oem', 'nrg_stk_gasm']

    
    #Not working:
    # 276 │   lst_vars_selec.remove('unit') 'unit' not in list
    #Crude oil supply - monthly data
    #Crude oil imports by field of produ

    #App crashed:
    #get_eust_df()
    #Crude oil imports by field of produ


    #maybe:
    #Exports of oil and petroleum pro
    
    
    datasets_dict = filter_dict_by_codes(datasets_dict, codes)
    
    toc_names = []

    for key in datasets_dict:
        temp = datasets_dict[key]['title']
        toc_names += [temp,]

    # Inspect the parsed data
    return datasets_dict, toc_names

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

sidebar, mainpage = st.columns([1,4])

with sidebar:

    tot_or_cap = st.radio(
        "Display data in:",
        ['Total', 'Per Capita'])
    
    toc, toc_names = get_toc()

    df_name_long = st.selectbox(
        'Which dataset would you like to view?',
        toc_names,
        index = toc_names.index('Net electricity generation by type of fuel - monthly data')
    )
    
    df_name = [i for i in toc if toc[i]['title']==df_name_long][0]

    dic_df = get_dic_df(df_name)
    
    data_start = datetime.strptime(toc[df_name]['dataStart'], '%Y-%m').date()
    data_end = datetime.strptime(toc[df_name]['dataEnd'], '%Y-%m').date()
    
    from_date, to_date = st.slider(
        'Which dates are you interested in?',
        min_value= data_start,
        max_value= data_end,
        value=(data_start, data_end),
        format="YYYY-MM"
    )

    from_date = from_date.replace(day=1)
    to_date = to_date.replace(day=1)

    dict_filters = {}
    
    lst_vars_selec = list(dic_df.keys())
    lst_vars_selec.remove('geo')
    lst_vars_selec.remove('unit')

    for i in lst_vars_selec:
        values = [value for value in dic_df[i]['pars'].values()]
        var_name = dic_df[i]['name']
        selection = st.selectbox(
                f'Pick {var_name}:',
                values
            )
        selec = [subkey for subkey, val in dic_df[i]['pars'].items() if val == selection]
        dict_filters.update({i: selec})

        if dic_df[i]['descr'] is not None:
            st.write(dic_df[i]['descr'])

    countries = [value for value in dic_df['geo']['pars'].values()]
    
    selected_countries = st.multiselect(
        'Which countries would you like to view?',
        countries,
        countries
    )

    if not len(selected_countries):
        st.warning("Select at least one country")

    selected_countries_code = [i for i in dic_df['geo']['pars'] if dic_df['geo']['pars'][i] in selected_countries]
    
    dict_filters.update({'geo': selected_countries_code})

    lst_vars = list(dic_df.keys())
    
    df_eust = get_eust_data(df_name, lst_vars)

    df_filtered = df_eust[
        (df_eust['date'] >= from_date)&
        (df_eust['date'] <= to_date)
    ]
    
    for key in dict_filters:
        temp_filt = dict_filters[key]
        df_filtered = df_filtered[df_filtered[key].isin(temp_filt)]
    
    units = df_filtered['unit'].unique()

    units = units.tolist()
    
    if len(units) > 1:
        selection = st.selectbox(
            f'Pick Unit:',
            units
        )

        df_filtered = df_filtered[df_filtered['unit'] == selection]

    unit = df_filtered['unit'].unique()

    unit = list(unit)

    dict_filters.update({'unit': unit})
    
    if tot_or_cap == 'Per Capita':
        df_filtered = per_capita(df_filtered)

with mainpage:

    unique_geos = df_filtered['geo'].unique()
    color_map = {geo: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, geo in enumerate(unique_geos)}

    
    with st.container():

        fig_line_chart = px.line(df_filtered, x='date', y='value', color='geo', color_discrete_map=color_map)
        st.plotly_chart(fig_line_chart)
        # st.line_chart(df_filtered, x='date', y='value',color='geo')
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Add content to the first column
    with col1:
        
        
        #st.header("Map")
        #st.write(df_filtered)
        nuts = get_nuts()
    
        oneYear_df_eust = df_filtered[df_eust['date'] == to_date]
    
        merged = nuts.merge(oneYear_df_eust, left_on='CNTR_CODE', right_on='geo')
    
        # Ensure the GeoDataFrame contains only necessary columns
        merged = merged[['CNTR_CODE', 'value', 'geometry']]

        merged['value'] = merged['value'].fillna(-1)

        fig = px.choropleth(
            merged,
            geojson=merged.geometry.__geo_interface__,
            locations=merged.index,
            color='value',
            hover_name='CNTR_CODE',
            hover_data=['value'],
            color_continuous_scale='Viridis',
            color_continuous_scale=[[0, 'grey'], [0.0001, 'yellow'], [1, 'purple']],
            range_color=(-1, merged['value'].max()), 
            labels={'value': 'Legend Name'}
        )
        
        # Update layout for dark theme and disable scrolling
        fig.update_geos(
            fitbounds="locations",
            visible=False,
            projection_type="mercator",
            lonaxis_range=[-10, 35],
            lataxis_range=[34, 71],
        )
        fig.update_layout(
            geo=dict(
                bgcolor='rgba(0,0,0,0)',
                showland=True,
                landcolor="black",
                showocean=True,
                oceancolor="black",
                lakecolor="black",
                showcountries=True,
                countrycolor="white"
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin={"r":0,"t":0,"l":0,"b":0},
            dragmode=False,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Convert GeoDataFrame to GeoJSON
        geojson_data = merged.to_json()
    
        # Create a base map
        m = folium.Map(location=[55.00, 13.0],
                       # zoom_start=3,
                       zoom_control=False,
                       scrollWheelZoom=False,
                       dragging=False)
    
        folium.Choropleth(
            geo_data=geojson_data,
            data=merged,
            columns=['CNTR_CODE', 'value'],
            key_on='feature.properties.CNTR_CODE',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Legend Name',
            tooltip=folium.GeoJsonTooltip(fields=['CNTR_CODE'], aliases=['Country Code:'])
        ).add_to(m)
    
        bounds = [[34.5, -10.5], [71.0, 35]]  # [min_lat, min_lng], [max_lat, max_lng]
        
        m.fit_bounds(bounds)
        
        css = """
        <div style="width:100%;height:0;padding-bottom:100%;position:absolute;">
          <div style="position:absolute;top:0;left:0;width:100%;height:100%;">
            {map}
          </div>
        </div>
        """
        # Get the HTML representation of the Folium map
        map_html = m.get_root().render()
        
        # Insert the map into the CSS container
        html = css.format(map=map_html)
        
        # Render the map with Streamlit
        st.components.v1.html(html, height=500, scrolling=False)
    
    # Add content to the second column
    with col2:
        st.header("Radial-Bar-Cart")
        #if tot_or_cap == 'Per Capita':
            #st.write(df_filtered2)
        monthly_mean = df_filtered
        
        monthly_mean['month'] = monthly_mean['datetime'].dt.month

        df_filtered['month_name'] = df_filtered['datetime'].dt.month_name()
        
        # Group by month and calculate the mean of the 'value' column
        monthly_mean = monthly_mean.groupby(['geo', 'month', 'month_name'])['value'].mean().reset_index()

        line_r_range = [0.000000000000001, monthly_mean['value'].max()]
        
        fig_line_polar = px.line_polar(monthly_mean,
                                     r = 'value', log_r = False, range_r = line_r_range,
                                     theta = 'month_name',
                                     color = 'geo', color_discrete_map=color_map, line_close=True, template="plotly_dark", 
                                    )

        st.plotly_chart(fig_line_polar)

        #min_date = df_eust['timestamp'].min()
        #max_date = df_eust['timestamp'].max()
    
        #st.write(f'From date: {from_date} to date: {to_date}')
        #st.write(df_eust)
    
        
        
        # Filter the data
    
        #st.write(filtered_df_eust)
        # Display the selected dates
        #st.write(f'From date: {from_date} to date: {to_date}')
    
        # st.line_chart(df_filtered, x='date', y='value',color='geo')
        # center on Liberty Bell, add marker
        #m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)
        #folium.Marker(
        #    [39.949610, -75.150282], popup="Liberty Bell", tooltip="Liberty Bell"
        #).add_to(m)
        
        # call to render Folium map in Streamlit
        #st_data = st_folium(m, width=725)
