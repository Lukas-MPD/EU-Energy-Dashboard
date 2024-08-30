import streamlit as st
import pandas as pd
import geopandas as gpd
from pathlib import Path
import eurostat as eust
from datetime import datetime
import xml.etree.ElementTree as ET
import requests
import plotly.express as px
from shapely.geometry import MultiPolygon, Polygon
import os

st.set_page_config(
    page_title='EU energy dashboard',
    page_icon=':electric_plug:', 
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

    df.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

    df_melted = pd.melt(df, id_vars=lst_vars, var_name='datetime', value_name='value')
   
    df_melted['datetime'] = pd.to_datetime(df_melted['datetime'])

    df_melted['date'] = df_melted['datetime'].dt.date

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
        
        pop_val = None 

        if year in pop_filtered.columns and not pop_filtered[year].empty:
            pop_val = pop_filtered[year].iloc[0]
        else:
            for i in range(5):
                year_temp = year - i - 1
                if year_temp in pop_filtered.columns and not pop_filtered[year_temp].empty:
                    pop_val = pop_filtered[year_temp].iloc[0]
                    break

        if pop_val:
            df.loc[ind, 'value'] = df.loc[ind, 'value'] / pop_val * 10000
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

    descr_to_val = dict(zip(dic_units['descr'], dic_units['val']))

    return dic_units, descr_to_val


@st.cache_data
def dic_countries(df_name):
    dic_countries = eust.get_dic(df_name, 'geo', frmt='df')

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

        data_dict.update(element.attrib)

        if element.text and element.text.strip():
            data_dict['text'] = element.text.strip()

        children = list(element)
        if children:
            child_dict = {}
            for child in children:
                child_tag = child.tag.split('}')[-1] 
                child_dict.setdefault(child_tag, []).append(parse_xml_to_dict(child))
            data_dict.update(child_dict)
    
        return data_dict
    
    def find_branch_with_code(element, target_code):
        for child in element:
            if child.tag.endswith('code') and child.text == target_code:
                return element
            result = find_branch_with_code(child, target_code)
            if result is not None:
                return result
        return None
    
    def extract_datasets(data_dict):
        datasets = {}
        
        def recursive_extract(element):
            if 'type' in element and element['type'] == 'dataset':
                title = None
                code = element.get('code')[0]['text']
                last_update = element.get('lastUpdate')[0]['text']
                last_modified = element.get('lastModified')[0]['text']
                data_start = element.get('dataStart')[0]['text']
                data_end = element.get('dataEnd')[0]['text']
                values = element.get('values')[0]['text']
                metadata = next((meta['text'] for meta in element.get('metadata', []) if meta.get('format') == 'html'), None)
                download_link = next((link['text'] for link in element.get('downloadLink', []) if link.get('format') == 'tsv'), None)

                for title_element in element.get('title', []):
                    if title_element['language'] == 'en':
                        title = title_element['text']
                        break

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

            for key, value in element.items():
                if isinstance(value, list):
                    for child in value:
                        if isinstance(child, dict):
                            recursive_extract(child)

        recursive_extract(data_dict)
        return datasets

    def get_eurostat_toc_xml():
        url = 'https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/xml'
        response = requests.get(url)
        response.raise_for_status() 
        return response.content

    def filter_dict_by_codes(data, codes):
        return {k: v for k, v in data.items() if k in codes}

    xml_data = get_eurostat_toc_xml()

    root = ET.fromstring(xml_data)

    target_code = 'nrg'
    target_branch = find_branch_with_code(root, target_code)

    if target_branch is not None:
        data_dict = parse_xml_to_dict(target_branch)
    else:
        data_dict = {}

    datasets_dict = extract_datasets(data_dict)

    codes = ['nrg_cb_sffm', 'nrg_cb_oilm', 'nrg_cb_gasm', 'nrg_cb_em', 'nrg_cb_eim', 'nrg_cb_pem']
    
    datasets_dict = filter_dict_by_codes(datasets_dict, codes)
    
    toc_names = []

    for key in datasets_dict:
        temp = datasets_dict[key]['title']
        toc_names += [temp,]

    return datasets_dict, toc_names

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :electric_plug: Energy in the EU :flag-eu:

Browse energy data from the [eurostat Database](https://ec.europa.eu/eurostat/data/database). This data is updated monthly by eurostat and queried via API.
'''

''
''

sidebar, mainpage = st.columns([1,4])

with sidebar:

    tot_or_cap = st.radio(
        "Display data in:",
        ['total', 'per 10,000 inhabitants'])
    
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
                values,
                help=dic_df[i]['descr']
            )
        selec = [subkey for subkey, val in dic_df[i]['pars'].items() if val == selection]
        dict_filters.update({i: selec})

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
    
    if tot_or_cap == 'per 10,000 inhabitants':
        df_filtered = per_capita(df_filtered)

with mainpage:

    unique_geos = df_filtered['geo'].unique()
    color_map = {geo: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, geo in enumerate(unique_geos)}
    toc_df = toc[df_name]

    try:
        toc_df['lastUpdate'] = datetime.strptime(toc_df['lastUpdate'], '%d.%m.%Y').strftime('%Y-%m-%d')
        toc_df['downloadLink'] = datetime.strptime(toc_df['downloadLink'], '%d.%m.%Y').strftime('%Y-%m-%d')
    except:
        os.write(1,b'Formating "lastUpdate" and/or "downloadLink" did not work.\n')

    filtered_descriptions = [
        dic_df[key]['pars'][value] 
        for key, values in dict_filters.items() 
        if key not in ['freq', 'geo']
        for value in values 
        if value in dic_df[key]['pars']
    ]

    filtered_descriptions_str = ", ".join(filtered_descriptions)

    st.subheader(toc_df['title'])

    filtered_descriptions_str
    
    with st.container():

        fig_line_chart = px.line(df_filtered, x='date', y='value', color='geo', color_discrete_map=color_map, labels= {'geo': 'Country', 'value': filtered_descriptions_str, 'date': 'Date'})
        fig_line_chart.for_each_trace(lambda t: t.update(name = dic_df['geo']['pars'][t.name],
                                      legendgroup = dic_df['geo']['pars'][t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, dic_df['geo']['pars'][t.name])
                                     )
                  )
        st.plotly_chart(fig_line_chart)

    col1, col2 = st.columns(2)
    
    with col1:

        map_date = st.slider(
            'Choose a date for the map:',
            min_value= from_date,
            max_value= to_date,
            value=to_date,
            format="YYYY-MM"
        )

        map_date = map_date.replace(day=1)
        
        lon_min, lon_max = -25, 42
        lat_min, lat_max = 35, 75

        nuts = get_nuts()
        
        def filter_multipolygons(gdf, lon_min, lon_max, lat_min, lat_max):
            def filter_polygon(polygon):
                if polygon.centroid.x >= lon_min and polygon.centroid.x <= lon_max and polygon.centroid.y >= lat_min and polygon.centroid.y <= lat_max:
                    return polygon
                return None
        
            filtered_geometries = []
            for geom in gdf.geometry:
                if isinstance(geom, MultiPolygon):
                    filtered_parts = [filter_polygon(p) for p in geom.geoms if filter_polygon(p) is not None]
                    if filtered_parts:
                        filtered_geometries.append(MultiPolygon(filtered_parts))
                    else:
                        filtered_geometries.append(None)
                elif isinstance(geom, Polygon):
                    filtered_geometries.append(filter_polygon(geom))
                else:
                    filtered_geometries.append(None)
            
            gdf = gdf.copy()
            gdf['geometry'] = filtered_geometries
            return gdf.dropna(subset=['geometry'])
        
        nuts = filter_multipolygons(nuts, lon_min, lon_max, lat_min, lat_max)
        oneYear_df_eust = df_filtered[df_eust['date'] == map_date]
    
        merged = nuts.merge(oneYear_df_eust, left_on='CNTR_CODE', right_on='geo')
    
        merged = merged[['CNTR_CODE', 'value', 'geometry']]

        if len(merged['value'].unique()) == 1 and pd.isna(merged['value'].unique()):
            percentile25 = 0.25
            median_value = 0.5
            percentile75 = 0.75
            max_value = 1
        elif len(merged['value'].unique()) == 1:
            percentile25 = 0.25
            median_value = 0.5
            percentile75 = 0.75
            max_value = merged['value'].max()
        else:
            max_value = merged['value'].max()
            percentile25 = merged['value'].quantile(.25) / max_value
            median_value = merged['value'].median() / max_value
            percentile75 = merged['value'].quantile(.75) / max_value
        
        merged['value_nona'] = merged['value'].fillna(-1)
        merged['value_custom'] = merged['value'].apply(lambda x: 'Null' if x == -1 else x)
        merged['country_name'] = merged['CNTR_CODE'].map(dic_df['geo']['pars'])

        try:
            fig = px.choropleth(
                merged,
                geojson=merged.geometry.__geo_interface__,
                locations=merged.index,
                color='value_nona',
                hover_name='country_name',
                hover_data=['value_custom'],
                color_continuous_scale=[[0, 'grey'], [0.0001, 'darkblue'], [percentile25, 'purple'], [median_value, 'yellow'], [percentile75, 'orange'], [1, 'red']],
                range_color=(-1, max_value), 
                labels={'value_nona': ""}
            )
            
            fig.update_geos(
                fitbounds="locations",
                visible=False,
                projection_type="mercator",
                lonaxis_range=[lon_min, lon_max], 
                lataxis_range=[lat_min, lat_max],
            )
            fig.update_layout(
                geo=dict(
                    bgcolor='rgba(0,0,0,0)',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin={"r":0,"t":0,"l":0,"b":0},
                dragmode=False,
            )
            fig.update_traces(
                hovertemplate='<b>%{hovertext}</b><br>' + filtered_descriptions_str + ': %{customdata}<extra></extra>',
                hovertext=merged['country_name'],
                customdata=merged['value_custom']
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.write("No country selected")
            
    with col2:

        monthly_mean = df_filtered
        
        monthly_mean['month'] = monthly_mean['datetime'].dt.month

        df_filtered['month_name'] = df_filtered['datetime'].dt.month_name()
        
        monthly_mean = monthly_mean.groupby(['geo', 'month', 'month_name'])['value'].mean().reset_index()

        line_r_range = [0.000000000000001, monthly_mean['value'].max()]
        
        fig_line_polar = px.line_polar(monthly_mean,
                                     r = 'value', log_r = False, range_r = line_r_range,
                                     theta = 'month_name',
                                     color = 'geo', color_discrete_map=color_map, line_close=True, template="plotly_dark", labels= {'geo': 'Country', 'value': filtered_descriptions_str, 'month_name': 'Month'}
                                    )
        fig_line_polar.for_each_trace(lambda t: t.update(name = dic_df['geo']['pars'][t.name],
                                      legendgroup = dic_df['geo']['pars'][t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, dic_df['geo']['pars'][t.name])
                                     )
                  )
        fig_line_polar.update_layout(showlegend=False)
        st.plotly_chart(fig_line_polar)
    
    with st.container():
        f"This dataset contains data from {toc_df['dataStart']} to {toc_df['dataEnd']} and contains {toc_df['values']} values. It was updated {toc_df['lastUpdate']} and last modified {toc_df['lastModified']}. You can view the metadata [here]({toc_df['metadata']}) and download the data [here]({toc_df['downloadLink']})."

