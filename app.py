import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import numpy as np

# Helper function for date-specific percentile calculation
def calculate_date_specific_percentiles(df, column_name, percentile=0.98):
    """
    Calculate percentile thresholds for each calendar date (month-day combination)
    Args:
        df: DataFrame with datetime column and heat index data
        column_name: 'heatindexmax2m' or 'heatindexmin2m'
        percentile: The percentile to calculate (0.98 for 98th, 0.02 for 2nd)
    Returns:
        DataFrame with date-specific thresholds and extreme event flags
    """
    df = df.copy()
    df['month_day'] = df['datetime'].dt.strftime('%m-%d')
    
    # Calculate percentile for each unique date
    date_thresholds = df.groupby('month_day')[column_name].quantile(percentile).reset_index()
    date_thresholds.columns = ['month_day', f'threshold_{percentile}']
    
    # Merge thresholds back to original data
    df = df.merge(date_thresholds, on='month_day', how='left')
    
    # Flag extreme events
    if percentile > 0.5:  # For 98th percentile (heat)
        df[f'extreme_event'] = df[column_name] > df[f'threshold_{percentile}']
    else:  # For 2nd percentile (cold)
        df[f'extreme_event'] = df[column_name] < df[f'threshold_{percentile}']
    
    return df

def identify_waves(df, column_name='extreme_event', min_consecutive=2):
    """
    Identify heatwaves/coldwaves (consecutive extreme events)
    """
    df = df.sort_values('datetime').copy()
    df['wave_id'] = (df[column_name] != df[column_name].shift()).cumsum()
    df['wave'] = df.groupby('wave_id')[column_name].transform('sum') >= min_consecutive
    df['in_wave'] = df[column_name] & df['wave']
    return df

# Page configuration
st.set_page_config(
    page_title="NC Heat Index Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("North Carolina Heat Index Analysis (1971-2021)")
# st.markdown("""
# **Redefining Extreme Heat and Cold Events for Emergency Healthcare**

# This analysis examines extreme temperature trends across six weather stations in North Carolina using a 
# date-specific percentile methodology that adapts to local climate conditions, as proposed in our research paper.

# **Note:** All measurements use Heat Index (apparent temperature combining air temperature and humidity), 
# not raw temperature. Heat index better represents how hot it actually feels and the health risks to humans.
# """)

# Load data function
@st.cache_data
def load_data():
    folder = 'heat_index_files'
    
    filenames = {
        'KAVL': 'KAVL-heatindex-1971-2021.xlsx',
        'KGSO': 'KGSO-heatindex-1971-2021.xlsx',
        'KHSE': 'KHSE-heatindex-1971-2021.xlsx',
        'KILM': 'KILM-heatindex-1971-2021.xlsx',
        'KCLT': 'KLCT-heatindex-1971-2021.xlsx',
        'KRDU': 'KRDU-heat-index-1971-2021.xlsx',
    }
    
    station_names = {
        'KAVL': 'Asheville (Mountains)',
        'KGSO': 'Greensboro (Piedmont)',
        'KHSE': 'Cape Hatteras (Coastal)',
        'KILM': 'Wilmington (Coastal)',
        'KCLT': 'Charlotte (Piedmont)',
        'KRDU': 'Raleigh-Durham (Piedmont)',
    }
    
    dfs = {}
    for station, file in filenames.items():
        path = os.path.join(folder, file)
        df = pd.read_excel(path)
        df['station'] = station
        df['station_name'] = station_names[station]
        
        # Convert heat index columns to numeric
        for col in ['heatindexmax2m', 'heatindexmin2m', 'heatindexavg2m']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['month_day'] = df['datetime'].dt.strftime('%m-%d')
        df['season'] = df['month'].apply(lambda x: 'Winter' if x in [12,1,2] 
                                          else 'Spring' if x in [3,4,5]
                                          else 'Summer' if x in [6,7,8]
                                          else 'Fall')
        dfs[station] = df
    
    all_data = pd.concat(dfs.values(), ignore_index=True)
    return dfs, all_data

# Load data
with st.spinner('Loading data...'):
    dfs, all_data = load_data()
selected_stations = list(dfs.keys())

year_range = st.sidebar.slider(
    "Year Range",
    min_value=1971,
    max_value=2021,
    value=(1971, 2021)
)

# Key Metrics Section
st.header("Key Metrics")

metric_view = st.radio(
    "View:",
    ["Statewide Average (All 6 Stations)", "Individual Station"],
    horizontal=True,
    key="metric_view"
)

if metric_view == "Statewide Average (All 6 Stations)":
    # Calculate metrics - all stations combined
    recent_years = all_data[(all_data['year'] >= 2015) & (all_data['month'].isin([6, 7, 8]))]
    early_years = all_data[(all_data['year'] <= 1980) & (all_data['month'].isin([6, 7, 8]))]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        recent_avg = recent_years['heatindexmax2m'].mean()
        early_avg = early_years['heatindexmax2m'].mean()
        st.metric(
            "Avg Summer High (2015-2021)",
            f"{recent_avg:.1f}°F",
            delta=f"+{recent_avg - early_avg:.1f}°F vs 1971-1980"
        )
    
    with col2:
        recent_min = recent_years['heatindexmin2m'].mean()
        early_min = early_years['heatindexmin2m'].mean()
        st.metric(
            "Avg Summer Low (2015-2021)",
            f"{recent_min:.1f}°F",
            delta=f"+{recent_min - early_min:.1f}°F vs 1971-1980"
        )
    
    with col3:
        summer_data = all_data[all_data['month'].isin([6, 7, 8])]
        hottest_year = summer_data.groupby('year')['heatindexmax2m'].mean().idxmax()
        st.metric(
            "Hottest Summer Year",
            f"{int(hottest_year)}"
        )
    
    with col4:
        # Calculate date-specific 98th percentile for summer
        summer_all = all_data[all_data['month'].isin([6, 7, 8])].copy()
        summer_extreme = calculate_date_specific_percentiles(
            summer_all, 
            'heatindexmax2m', 
            percentile=0.98
        )
        extreme_days = summer_extreme['extreme_event'].sum()
        st.metric(
            "Extreme Heat Days (>98th %ile)",
            f"{extreme_days:,}",
            help="Days exceeding date-specific 98th percentile threshold"
        )

else:  # Individual Station view
    selected_metric_station = st.selectbox(
        "Select Station",
        options=list(dfs.keys()),
        format_func=lambda x: dfs[x]['station_name'].iloc[0],
        key="metric_station"
    )
    
    station_data = dfs[selected_metric_station]
    recent_years = station_data[(station_data['year'] >= 2015) & (station_data['month'].isin([6, 7, 8]))]
    early_years = station_data[(station_data['year'] <= 1980) & (station_data['month'].isin([6, 7, 8]))]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        recent_avg = recent_years['heatindexmax2m'].mean()
        early_avg = early_years['heatindexmax2m'].mean()
        st.metric(
            "Avg Summer High (2015-2021)",
            f"{recent_avg:.1f}°F",
            delta=f"+{recent_avg - early_avg:.1f}°F vs 1971-1980"
        )
    
    with col2:
        recent_min = recent_years['heatindexmin2m'].mean()
        early_min = early_years['heatindexmin2m'].mean()
        st.metric(
            "Avg Summer Low (2015-2021)",
            f"{recent_min:.1f}°F",
            delta=f"+{recent_min - early_min:.1f}°F vs 1971-1980"
        )
    
    with col3:
        summer_data = station_data[station_data['month'].isin([6, 7, 8])]
        hottest_year = summer_data.groupby('year')['heatindexmax2m'].mean().idxmax()
        st.metric(
            "Hottest Summer Year",
            f"{int(hottest_year)}"
        )
    
    with col4:
        # Calculate date-specific 98th percentile for this station
        summer_station = station_data[station_data['month'].isin([6, 7, 8])].copy()
        summer_extreme = calculate_date_specific_percentiles(
            summer_station, 
            'heatindexmax2m', 
            percentile=0.98
        )
        extreme_days = summer_extreme['extreme_event'].sum()
        st.metric(
            "Extreme Heat Days (>98th %ile)",
            f"{extreme_days}",
            help="Days exceeding date-specific 98th percentile threshold"
        )

# Tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Methodology",
    "Extreme Temperature Days", 
    "Heatwaves & Coldwaves",
    "WWA Comparison",
    "Additional Analysis"
])

# TAB 1: METHODOLOGY
with tab1:
    st.header("Research Methodology")
    st.markdown("""
    Our analysis uses a date-specific percentile approach that adapts to local climate conditions,
    rather than fixed temperature thresholds that don't account for regional differences.
    """)
    
    # Methodology visualization
    st.subheader("How We Calculate Percentile Thresholds")
    
    st.markdown("""
    **For each calendar date (e.g., June 15):**
    1. Collect all June 15th values from 1971-2021 (51 years)
    2. Calculate the 98th percentile (heat) and 2nd percentile (cold) of those 51 values
    3. Any June 15th exceeding its specific threshold is an "extreme heat/cold day"
    4. Two or more consecutive extreme days = "heatwave/coldwave"
    """)
    
    # Enhanced Key Metrics for Methodology
    st.subheader("Extreme Events Comparison: Early vs Recent Decades")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate extreme heat days (summer)
    summer_all = all_data[all_data['month'].isin([3,4,5,6,7,8,9,10])].copy()
    summer_extreme = calculate_date_specific_percentiles(summer_all, 'heatindexmax2m', 0.98)
    
    early_heat = summer_extreme[(summer_extreme['year'] >= 1971) & (summer_extreme['year'] <= 1980)]
    recent_heat = summer_extreme[(summer_extreme['year'] >= 2012) & (summer_extreme['year'] <= 2021)]
    
    early_heat_days = early_heat['extreme_event'].sum()
    recent_heat_days = recent_heat['extreme_event'].sum()
    
    # Calculate extreme cold days (winter)
    winter_all = all_data[all_data['month'].isin([11,12,1,2])].copy()
    winter_extreme = calculate_date_specific_percentiles(winter_all, 'heatindexmin2m', 0.02)
    
    early_cold = winter_extreme[(winter_extreme['year'] >= 1971) & (winter_extreme['year'] <= 1980)]
    recent_cold = winter_extreme[(winter_extreme['year'] >= 2012) & (winter_extreme['year'] <= 2021)]
    
    early_cold_days = early_cold['extreme_event'].sum()
    recent_cold_days = recent_cold['extreme_event'].sum()
    
    with col1:
        st.metric(
            "Extreme Heat Days (1971-1980)",
            f"{early_heat_days:,}",
            help="Days exceeding 98th percentile (March-October)"
        )
    
    with col2:
        increase = recent_heat_days - early_heat_days
        pct_increase = (increase / early_heat_days * 100) if early_heat_days > 0 else 0
        st.metric(
            "Extreme Heat Days (2012-2021)",
            f"{recent_heat_days:,}",
            delta=f"+{increase:,} days ({pct_increase:.1f}%)",
            delta_color="inverse",
            help="Shows increase in extreme heat events"
        )
    
    with col3:
        st.metric(
            "Extreme Cold Days (1971-1980)",
            f"{early_cold_days:,}",
            help="Days below 2nd percentile (November-February)"
        )
    
    with col4:
        decrease = recent_cold_days - early_cold_days
        pct_decrease = (decrease / early_cold_days * 100) if early_cold_days > 0 else 0
        st.metric(
            "Extreme Cold Days (2012-2021)",
            f"{recent_cold_days:,}",
            delta=f"{decrease:,} days ({pct_decrease:.1f}%)",
            delta_color="normal",
            help="Shows decrease in extreme cold events"
        )
    
    st.markdown("---")
    
    st.subheader("Explore Any Date's Distribution (March - October)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Date picker for SUMMER
        selected_summer_date = st.date_input(
            "Select a summer date (for 98th percentile - heat threshold):",
            value=pd.to_datetime("2024-06-15"),
            min_value=pd.to_datetime("2024-03-01"),
            max_value=pd.to_datetime("2024-10-31"),
            key="summer_date_picker"
        )
    
    with col2:
        # Select station for methodology demo
        demo_station = st.selectbox(
            "Select Station:",
            options=selected_stations,
            format_func=lambda x: dfs[x]['station_name'].iloc[0],
            key="method_station"
        )
    
    # ===== SUMMER DISTRIBUTION =====
    demo_month = selected_summer_date.month
    demo_day = selected_summer_date.day
    demo_date_str = f"{demo_month:02d}-{demo_day:02d}"
    
    demo_df = dfs[demo_station].copy()
    demo_data = demo_df[demo_df['month_day'] == demo_date_str].dropna(subset=['heatindexmax2m'])
    
    if len(demo_data) > 0:
        p98 = demo_data['heatindexmax2m'].quantile(0.98)
        mean_val = demo_data['heatindexmax2m'].mean()
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{mean_val:.1f}°F")
        with col2:
            st.metric("98th Percentile", f"{p98:.1f}°F")
        with col3:
            temp_range = demo_data['heatindexmax2m'].max() - demo_data['heatindexmax2m'].min()
            st.metric("Temperature Range", f"{temp_range:.1f}°F")
        
        # Create summer distribution
        fig_summer = go.Figure()
        
        # Histogram
        fig_summer.add_trace(go.Histogram(
            x=demo_data['heatindexmax2m'],
            nbinsx=20,
            name='Distribution',
            marker_color='lightcoral',
            opacity=0.7,
            histnorm='probability'
        ))
        
        # Add ONLY extreme heat shading (no cold)
        fig_summer.add_vrect(
            x0=p98, 
            x1=demo_data['heatindexmax2m'].max() + 2,
            fillcolor="red", 
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text="Extreme Heat (>98th %ile)",
            annotation_position="top right",
            annotation=dict(font_size=12, font_color="darkred")
        )
        
        # Add 98th percentile line
        fig_summer.add_vline(
            x=p98, 
            line_dash="dash", 
            line_color="red", 
            line_width=3,
            annotation_text=f"98th Percentile: {p98:.1f}°F",
            annotation_position="top"
        )
        
        # Add mean line
        fig_summer.add_vline(
            x=mean_val, 
            line_dash="dot", 
            line_color="green", 
            line_width=2,
            annotation_text=f"Mean: {mean_val:.1f}°F",
            annotation_position="bottom"
        )
        
        fig_summer.update_layout(
            title=f"Summer: {selected_summer_date.strftime('%B %d')} Heat Index Distribution at {dfs[demo_station]['station_name'].iloc[0]}<br><sub>Showing {len(demo_data)} years of data (1971-2021)</sub>",
            xaxis_title=f"Temperature of {selected_summer_date.strftime('%B %d')} Across All Years (°F)",
            yaxis_title="Frequency (Probability)",
            height=450,
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_summer, use_container_width=True, key="summer_methodology")
        
        st.info(f"""
        **Heat Threshold:** Any {selected_summer_date.strftime('%B %d')} with heat index > {p98:.1f}°F is an extreme heat day.
        """)
    
    # ===== WINTER DISTRIBUTION =====
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_winter_date = st.date_input(
            "Select a winter date (for 2nd percentile - cold threshold):",
            value=pd.to_datetime("2024-01-15"),
            min_value=pd.to_datetime("2023-11-01"),
            max_value=pd.to_datetime("2024-02-29"),
            key="winter_date_picker"
        )
    
    winter_month = selected_winter_date.month
    winter_day = selected_winter_date.day
    winter_date_str = f"{winter_month:02d}-{winter_day:02d}"
    
    winter_data = demo_df[demo_df['month_day'] == winter_date_str].dropna(subset=['heatindexmin2m'])
    
    if len(winter_data) > 0:
        p2 = winter_data['heatindexmin2m'].quantile(0.02)
        mean_val_winter = winter_data['heatindexmin2m'].mean()
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{mean_val_winter:.1f}°F")
        with col2:
            st.metric("2nd Percentile", f"{p2:.1f}°F")
        with col3:
            temp_range_winter = winter_data['heatindexmin2m'].max() - winter_data['heatindexmin2m'].min()
            st.metric("Temperature Range", f"{temp_range_winter:.1f}°F")
        
        # Create winter distribution
        fig_winter = go.Figure()
        
        # Histogram
        fig_winter.add_trace(go.Histogram(
            x=winter_data['heatindexmin2m'],
            nbinsx=20,
            name='Distribution',
            marker_color='lightblue',
            opacity=0.7,
            histnorm='probability'
        ))
        
        # Add ONLY extreme cold shading
        fig_winter.add_vrect(
            x0=winter_data['heatindexmin2m'].min() - 2,
            x1=p2,
            fillcolor="blue", 
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text="Extreme Cold (<2nd %ile)",
            annotation_position="top left",
            annotation=dict(font_size=12, font_color="darkblue")
        )
        
        # Add 2nd percentile line
        fig_winter.add_vline(
            x=p2, 
            line_dash="dash", 
            line_color="blue", 
            line_width=3,
            annotation_text=f"2nd Percentile: {p2:.1f}°F",
            annotation_position="top"
        )
        
        # Add mean line
        fig_winter.add_vline(
            x=mean_val_winter, 
            line_dash="dot", 
            line_color="green", 
            line_width=2,
            annotation_text=f"Mean: {mean_val_winter:.1f}°F",
            annotation_position="bottom"
        )
        
        fig_winter.update_layout(
            title=f"Winter: {selected_winter_date.strftime('%B %d')} Heat Index Distribution at {dfs[demo_station]['station_name'].iloc[0]}<br><sub>Showing {len(winter_data)} years of data (1971-2021)</sub>",
            xaxis_title=f"Temperature of {selected_winter_date.strftime('%B %d')} Across All Years (°F)",
            yaxis_title="Frequency (Probability)",
            height=450,
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_winter, use_container_width=True, key="winter_methodology")
        
        st.info(f"""
        **Cold Threshold:** Any {selected_winter_date.strftime('%B %d')} with heat index < {p2:.1f}°F is an extreme cold day
        """)
    
    # Flow diagram
    st.markdown("---")
    st.subheader("Analysis Flow")
    st.markdown("""
```
    Calculate percentiles for each date across 51 years
                    ↓
        ┌───────────────────────┬───────────────────────┐
        ↓                       ↓                       ↓
    Extreme Heat Days    Extreme Cold Days     Heatwaves/Coldwaves
    (Mar-Oct, >98th)     (Nov-Feb, <2nd)      (2+ consecutive
                                               extreme days)
```
    """)

# TAB 2: EXTREME HEAT EVENTS
# TAB 2: EXTREME HEAT & COLD EVENTS (MERGED)
with tab2:
    st.header("Extreme Temperature Days")
    st.markdown("""
    **Extreme Heat Days:** Days exceeding date-specific 98th percentile (March-October)  
    **Extreme Cold Days:** Days below date-specific 2nd percentile (November-February)
    """)
    
    event_type = st.radio(
        "Select Event Type:",
        ["Extreme Heat Days", "Extreme Cold Days"],
        horizontal=True,
        key="event_type"
    )
    
    if event_type == "Extreme Heat Days":
        st.subheader("Extreme Heat Days by Station (March - October)")
        
        # Calculate extreme heat days for all stations
        extreme_heat_data = []
        
        for station in selected_stations:
            df = dfs[station]
            summer = df[(df['month'].isin([3,4,5,6,7,8,9,10])) & 
                        (df['year'] >= year_range[0]) & 
                        (df['year'] <= year_range[1])].copy()
            
            # Calculate for max (daytime)
            max_extreme = calculate_date_specific_percentiles(summer.dropna(subset=['heatindexmax2m']), 
                                                              'heatindexmax2m', 0.98)
            max_counts = max_extreme.groupby('year')['extreme_event'].sum()
            
            # Calculate for min (nighttime)
            min_extreme = calculate_date_specific_percentiles(summer.dropna(subset=['heatindexmin2m']), 
                                                              'heatindexmin2m', 0.98)
            min_counts = min_extreme.groupby('year')['extreme_event'].sum()
            
            for year in range(year_range[0], year_range[1] + 1):
                extreme_heat_data.append({
                    'station': station,
                    'station_name': df['station_name'].iloc[0],
                    'year': year,
                    'max_extreme': max_counts.get(year, 0),
                    'min_extreme': min_counts.get(year, 0)
                })
        
        extreme_heat_df = pd.DataFrame(extreme_heat_data)
        
        # Create faceted plot
        fig_heat = make_subplots(
            rows=2, cols=3,
            subplot_titles=[dfs[s]['station_name'].iloc[0] for s in selected_stations],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            x_title="Year",
            y_title="Count of Extreme Heat Days"
        )
        
        for idx, station in enumerate(selected_stations):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            station_data = extreme_heat_df[extreme_heat_df['station'] == station]
            
            # Max HI (darker red) - with smoothing
            fig_heat.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['max_extreme'],
                    name="Max HI > 98p", 
                    mode='lines', 
                    line=dict(color='darkred', width=2.5, shape='spline'),
                    legendgroup="heat_max", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Min HI (light orange) - with smoothing
            fig_heat.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['min_extreme'],
                    name="Min HI > 98p", 
                    mode='lines', 
                    line=dict(color='#FFA500', width=2.5, shape='spline'),
                    legendgroup="heat_min", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
        
        fig_heat.update_layout(
            title_text="Count of Extreme Heat Days by Station",
            height=800,
            showlegend=True,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        fig_heat.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_heat.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig_heat, use_container_width=True, key="extreme_heat_facet")
        
        # Trend analysis
        st.subheader("Extreme Heat Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Daytime Extreme Heat (Max HI):**")
            for station in selected_stations[:3]:
                station_data = extreme_heat_df[extreme_heat_df['station'] == station]
                early = station_data[station_data['year'] <= 1990]['max_extreme'].mean()
                recent = station_data[station_data['year'] >= 2010]['max_extreme'].mean()
                change = ((recent - early) / early * 100) if early > 0 else 0
                st.write(f"**{station_data['station_name'].iloc[0]}**: {change:+.1f}% change")
        
        with col2:
            st.markdown("**Nighttime Extreme Heat (Min HI):**")
            for station in selected_stations[3:]:
                station_data = extreme_heat_df[extreme_heat_df['station'] == station]
                early = station_data[station_data['year'] <= 1990]['min_extreme'].mean()
                recent = station_data[station_data['year'] >= 2010]['min_extreme'].mean()
                change = ((recent - early) / early * 100) if early > 0 else 0
                st.write(f"**{station_data['station_name'].iloc[0]}**: {change:+.1f}% change")
    
    else:  # Extreme Cold Days
        st.subheader("Extreme Cold Days by Station (November - February)")
        
        # Calculate extreme cold days for all stations
        extreme_cold_data = []
        
        for station in selected_stations:
            df = dfs[station]
            winter = df[(df['month'].isin([11,12,1,2])) & 
                        (df['year'] >= year_range[0]) & 
                        (df['year'] <= year_range[1])].copy()
            
            # Calculate for max (daytime)
            max_extreme = calculate_date_specific_percentiles(winter.dropna(subset=['heatindexmax2m']), 
                                                              'heatindexmax2m', 0.02)
            max_counts = max_extreme.groupby('year')['extreme_event'].sum()
            
            # Calculate for min (nighttime)
            min_extreme = calculate_date_specific_percentiles(winter.dropna(subset=['heatindexmin2m']), 
                                                              'heatindexmin2m', 0.02)
            min_counts = min_extreme.groupby('year')['extreme_event'].sum()
            
            for year in range(year_range[0], year_range[1] + 1):
                extreme_cold_data.append({
                    'station': station,
                    'station_name': df['station_name'].iloc[0],
                    'year': year,
                    'max_extreme': max_counts.get(year, 0),
                    'min_extreme': min_counts.get(year, 0)
                })
        
        extreme_cold_df = pd.DataFrame(extreme_cold_data)
        
        # Create faceted plot
        fig_cold = make_subplots(
            rows=2, cols=3,
            subplot_titles=[dfs[s]['station_name'].iloc[0] for s in selected_stations],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            x_title="Year",
            y_title="Count of Extreme Cold Days"
        )
        
        for idx, station in enumerate(selected_stations):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            station_data = extreme_cold_df[extreme_cold_df['station'] == station]
            
            # Max HI (navy blue) - with smoothing
            fig_cold.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['max_extreme'],
                    name="Max HI < 2p", 
                    mode='lines', 
                    line=dict(color='navy', width=2.5, shape='spline'),
                    legendgroup="cold_max", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Min HI (light blue) - with smoothing
            fig_cold.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['min_extreme'],
                    name="Min HI < 2p", 
                    mode='lines', 
                    line=dict(color='skyblue', width=2.5, shape='spline'),
                    legendgroup="cold_min", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
        
        fig_cold.update_layout(
            title_text="Count of Extreme Cold Days by Station",
            height=800,
            showlegend=True,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        fig_cold.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_cold.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig_cold, use_container_width=True, key="extreme_cold_facet")
        
        # Trend analysis
        st.subheader("Extreme Cold Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Daytime Extreme Cold (Max HI):**")
            for station in selected_stations[:3]:
                station_data = extreme_cold_df[extreme_cold_df['station'] == station]
                early = station_data[station_data['year'] <= 1990]['max_extreme'].mean()
                recent = station_data[station_data['year'] >= 2010]['max_extreme'].mean()
                change = ((recent - early) / early * 100) if early > 0 else 0
                st.write(f"**{station_data['station_name'].iloc[0]}**: {change:+.1f}% change")
        
        with col2:
            st.markdown("**Nighttime Extreme Cold (Min HI):**")
            for station in selected_stations[3:]:
                station_data = extreme_cold_df[extreme_cold_df['station'] == station]
                early = station_data[station_data['year'] <= 1990]['min_extreme'].mean()
                recent = station_data[station_data['year'] >= 2010]['min_extreme'].mean()
                change = ((recent - early) / early * 100) if early > 0 else 0
                st.write(f"**{station_data['station_name'].iloc[0]}**: {change:+.1f}% change")

# TAB 3: HEATWAVES & COLDWAVES (renumbered, was tab 4)
with tab3:
    st.header("Heatwaves & Coldwaves")
    st.markdown("""
    **Heatwave:** 2+ consecutive days exceeding date-specific 98th percentile (March-October)  
    **Coldwave:** 2+ consecutive days below date-specific 2nd percentile (November-February)
    """)
    
    wave_type = st.radio(
        "Select Wave Type:",
        ["Heatwaves", "Coldwaves"],
        horizontal=True,
        key="wave_type"
    )
    
    if wave_type == "Heatwaves":
        st.subheader("Heatwave Analysis (March - October)")
        
        # [Same calculation code as before...]
        heatwave_data = []
        
        for station in selected_stations:
            df = dfs[station].copy()
            summer = df[(df['month'].isin([3,4,5,6,7,8,9,10])) & 
                        (df['year'] >= year_range[0]) & 
                        (df['year'] <= year_range[1])].copy()
            
            max_extreme = calculate_date_specific_percentiles(
                summer.dropna(subset=['heatindexmax2m']), 
                'heatindexmax2m', 
                0.98
            )
            max_extreme = identify_waves(max_extreme, 'extreme_event', min_consecutive=2)
            heatwave_max_counts = max_extreme[max_extreme['in_wave']].groupby('year').size()
            
            min_extreme = calculate_date_specific_percentiles(
                summer.dropna(subset=['heatindexmin2m']), 
                'heatindexmin2m', 
                0.98
            )
            min_extreme = identify_waves(min_extreme, 'extreme_event', min_consecutive=2)
            heatwave_min_counts = min_extreme[min_extreme['in_wave']].groupby('year').size()
            
            for year in range(year_range[0], year_range[1] + 1):
                heatwave_data.append({
                    'station': station,
                    'station_name': df['station_name'].iloc[0],
                    'year': year,
                    'heatwave_max': heatwave_max_counts.get(year, 0),
                    'heatwave_min': heatwave_min_counts.get(year, 0)
                })
        
        heatwave_df = pd.DataFrame(heatwave_data)
        
        # Create faceted plot with SMOOTH lines
        fig_heatwaves = make_subplots(
            rows=2, cols=3,
            subplot_titles=[dfs[s]['station_name'].iloc[0] for s in selected_stations],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            x_title="Year",
            y_title="Count of Heatwaves"
        )
        
        for idx, station in enumerate(selected_stations):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            station_data = heatwave_df[heatwave_df['station'] == station]
            
            # HI-max Heatwaves (red) - SMOOTH
            fig_heatwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['heatwave_max'],
                    name="HI-max Heatwaves", 
                    mode='lines', 
                    line=dict(color='red', width=2.5, shape='spline'),
                    legendgroup="heatwave_max", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # HI-min Heatwaves (orange) - SMOOTH
            fig_heatwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['heatwave_min'],
                    name="HI-min Heatwaves", 
                    mode='lines', 
                    line=dict(color='orange', width=2.5, shape='spline'),
                    legendgroup="heatwave_min", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
        
        fig_heatwaves.update_layout(
            title_text="Count of Heatwaves by Station (March - October)",
            height=800,
            showlegend=True,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        fig_heatwaves.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_heatwaves.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig_heatwaves, use_container_width=True, key="heatwaves_facet")
        
        # [Same summary statistics as before...]
    
        
        # Summary statistics
        st.subheader("Heatwave Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Daytime Heatwaves (HI-max) - Recent Increase:**")
            for station in selected_stations[:3]:
                station_data = heatwave_df[heatwave_df['station'] == station]
                early = station_data[station_data['year'] <= 1990]['heatwave_max'].sum()
                recent = station_data[station_data['year'] >= 2010]['heatwave_max'].sum()
                st.write(f"**{station_data['station_name'].iloc[0]}**: {early} days (1971-1990) → {recent} days (2010-2021)")
        
        with col2:
            st.markdown("**Nighttime Heatwaves (HI-min) - Recent Increase:**")
            for station in selected_stations[3:]:
                station_data = heatwave_df[heatwave_df['station'] == station]
                early = station_data[station_data['year'] <= 1990]['heatwave_min'].sum()
                recent = station_data[station_data['year'] >= 2010]['heatwave_min'].sum()
                st.write(f"**{station_data['station_name'].iloc[0]}**: {early} days (1971-1990) → {recent} days (2010-2021)")
        
        st.info("**Key Finding:** Nighttime heatwaves (orange lines) are particularly concerning as they prevent physiological recovery from daytime heat, compounding health risks.")
    
    else:  # Coldwaves
        st.subheader("Coldwave Analysis (November - February)")
        
        # Calculate coldwaves for all stations
        coldwave_data = []
        
        for station in selected_stations:
            df = dfs[station].copy()
            
            # Coldwaves (November-February)
            winter = df[(df['month'].isin([11,12,1,2])) & 
                        (df['year'] >= year_range[0]) & 
                        (df['year'] <= year_range[1])].copy()
            
            # Max (daytime) coldwaves
            max_cold = calculate_date_specific_percentiles(
                winter.dropna(subset=['heatindexmax2m']), 
                'heatindexmax2m', 
                0.02
            )
            max_cold = identify_waves(max_cold, 'extreme_event', min_consecutive=2)
            coldwave_max_counts = max_cold[max_cold['in_wave']].groupby('year').size()
            
            # Min (nighttime) coldwaves
            min_cold = calculate_date_specific_percentiles(
                winter.dropna(subset=['heatindexmin2m']), 
                'heatindexmin2m', 
                0.02
            )
            min_cold = identify_waves(min_cold, 'extreme_event', min_consecutive=2)
            coldwave_min_counts = min_cold[min_cold['in_wave']].groupby('year').size()
            
            for year in range(year_range[0], year_range[1] + 1):
                coldwave_data.append({
                    'station': station,
                    'station_name': df['station_name'].iloc[0],
                    'year': year,
                    'coldwave_max': coldwave_max_counts.get(year, 0),
                    'coldwave_min': coldwave_min_counts.get(year, 0)
                })
        
        coldwave_df = pd.DataFrame(coldwave_data)
        
        # Create faceted plot (2x3 grid)
        fig_coldwaves = make_subplots(
            rows=2, cols=3,
            subplot_titles=[dfs[s]['station_name'].iloc[0] for s in selected_stations],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            x_title="Year",
            y_title="Count of Coldwaves"
        )
        
        for idx, station in enumerate(selected_stations):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            station_data = coldwave_df[coldwave_df['station'] == station]
            
            # HI-max Coldwaves (dark blue)
            fig_coldwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['coldwave_max'],
                    name="HI-max Coldwaves", 
                    mode='lines', 
                    line=dict(color='darkblue', width=2.5, shape='spline'),
                    legendgroup="coldwave_max", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # HI-min Coldwaves (light blue)
            fig_coldwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['coldwave_min'],
                    name="HI-min Coldwaves", 
                    mode='lines', 
                    line=dict(color='lightblue', width=2.5, shape='spline'),
                    legendgroup="coldwave_min", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
        
        fig_coldwaves.update_layout(
            title_text="Count of Coldwaves by Station (November - February)",
            height=800,
            showlegend=True,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update all x and y axes
        fig_coldwaves.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_coldwaves.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig_coldwaves, use_container_width=True, key="coldwaves_facet")
        
        # Summary statistics
        st.subheader("Coldwave Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Daytime Coldwaves (HI-max) - Trend:**")
            for station in selected_stations[:3]:
                station_data = coldwave_df[coldwave_df['station'] == station]
                early = station_data[station_data['year'] <= 1990]['coldwave_max'].sum()
                recent = station_data[station_data['year'] >= 2010]['coldwave_max'].sum()
                change = recent - early
                st.write(f"**{station_data['station_name'].iloc[0]}**: {early} days (1971-1990) → {recent} days (2010-2021)")
        
        with col2:
            st.markdown("**Nighttime Coldwaves (HI-min) - Trend:**")
            for station in selected_stations[3:]:
                station_data = coldwave_df[coldwave_df['station'] == station]
                early = station_data[station_data['year'] <= 1990]['coldwave_min'].sum()
                recent = station_data[station_data['year'] >= 2010]['coldwave_min'].sum()
                change = recent - early
                st.write(f"**{station_data['station_name'].iloc[0]}**: {early} days (1971-1990) → {recent} days (2010-2021)")
        
        st.info("**Key Finding:** Coldwaves are generally decreasing as winters warm, but extreme cold events still pose significant health risks when they occur.")

# TAB 4: WWA COMPARISON
with tab4:
    st.header("Comparison with NWS Watches, Warnings, and Advisories")
    
    
    # Placeholder visualization
    st.markdown("### Sample Comparison Structure")
    st.markdown("""
    The comparison would show:
    - Red dots: Our calculated heatwave days (using 98th percentile)
    - Black dots: WWA issued by NWS
    - Gap between them shows missed events by current warning system
    """)

# TAB 5: ADDITIONAL ANALYSIS
with tab5:
    st.header("Additional Analysis")
    
    analysis_choice = st.selectbox(
        "Select Analysis Type:",
        ["Temperature Distribution by Station", "Seasonal Patterns", "Decadal Trends", "Station Comparison"]
    )
    
    if analysis_choice == "Temperature Distribution by Station":
        # Box plot
        summer_data = []
        for station in selected_stations:
            df = dfs[station]
            summer = df[(df['month'].isin([6, 7, 8])) & 
                        (df['year'] >= year_range[0]) & 
                        (df['year'] <= year_range[1])]
            summer_data.append(summer[['station_name', 'heatindexmax2m']].dropna())
        
        combined = pd.concat(summer_data)
        
        fig_box = px.box(
            combined,
            x='station_name',
            y='heatindexmax2m',
            color='station_name',
            title="Summer Heat Index Distribution by Station",
            labels={'heatindexmax2m': 'Heat Index (°F)', 'station_name': 'Station'}
        )
        fig_box.update_layout(height=500, showlegend=False, template="plotly_white")
        st.plotly_chart(fig_box, use_container_width=True, key="box_dist")
    
    elif analysis_choice == "Seasonal Patterns":
        # Heatmap of monthly averages
        monthly_data = []
        for station in selected_stations:
            df = dfs[station]
            filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
            monthly_avg = filtered.groupby('month')['heatindexmax2m'].mean()
            monthly_data.append({
                'Station': df['station_name'].iloc[0],
                **{f'Month_{m}': monthly_avg.get(m, None) for m in range(1, 13)}
            })
        
        monthly_df = pd.DataFrame(monthly_data)
        month_cols = [f'Month_{m}' for m in range(1, 13)]
        
        fig_heat_monthly = go.Figure(data=go.Heatmap(
            z=monthly_df[month_cols].values,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=monthly_df['Station'],
            colorscale='RdYlBu_r',
            text=monthly_df[month_cols].values,
            texttemplate='%{text:.1f}°F',
            textfont={"size": 10},
            colorbar=dict(title="Heat Index (°F)")
        ))
        
        fig_heat_monthly.update_layout(
            title="Average Monthly Heat Index by Station",
            xaxis_title="Month",
            yaxis_title="Station",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_heat_monthly, use_container_width=True, key="seasonal_heatmap")
    
    elif analysis_choice == "Decadal Trends":
        st.markdown("### Temperature Change by Decade")
        
        for station in selected_stations:
            df = dfs[station]
            summer = df[df['month'].isin([6,7,8])].dropna(subset=['heatindexmax2m'])
            
            decade_avg = summer.groupby(summer['year'] // 10 * 10)['heatindexmax2m'].mean()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                fig_decade = go.Figure()
                fig_decade.add_trace(go.Bar(
                    x=[f"{int(d)}s" for d in decade_avg.index],
                    y=decade_avg.values,
                    marker_color='coral'
                ))
                fig_decade.update_layout(
                    title=f"{df['station_name'].iloc[0]} - Average Summer Heat Index by Decade",
                    xaxis_title="Decade",
                    yaxis_title="Heat Index (°F)",
                    height=300,
                    template="plotly_white"
                )
                st.plotly_chart(fig_decade, use_container_width=True, key=f"decade_{station}")
            
            with col2:
                if len(decade_avg) > 1:
                    change = decade_avg.iloc[-1] - decade_avg.iloc[0]
                    st.metric(
                        "Total Change",
                        f"{change:+.1f}°F",
                        delta=f"{(change/decade_avg.iloc[0]*100):+.1f}%"
                    )
    
    else:  # Station Comparison
        st.markdown("### Geographic Comparison")
        
        # Average by region
        regions = {
            'Mountains': ['KAVL'],
            'Piedmont': ['KGSO', 'KCLT', 'KRDU'],
            'Coastal': ['KHSE', 'KILM']
        }
        
        region_data = []
        for region, stations in regions.items():
            region_stations = [s for s in stations if s in selected_stations]
            if region_stations:
                dfs_region = [dfs[s] for s in region_stations]
                combined_region = pd.concat(dfs_region)
                summer = combined_region[combined_region['month'].isin([6,7,8])]
                summer = summer[(summer['year'] >= year_range[0]) & (summer['year'] <= year_range[1])]
                
                avg_by_year = summer.groupby('year')['heatindexmax2m'].mean()
                
                for year, avg in avg_by_year.items():
                    region_data.append({
                        'Region': region,
                        'Year': year,
                        'Avg Heat Index': avg
                    })
        
        if region_data:
            region_df = pd.DataFrame(region_data)
            
            fig_region = px.line(
                region_df,
                x='Year',
                y='Avg Heat Index',
                color='Region',
                title="Average Summer Heat Index by Geographic Region",
                markers=True
            )
            fig_region.update_layout(height=500, template="plotly_white")
            st.plotly_chart(fig_region, use_container_width=True, key="region_comparison")

# Footer
st.markdown("---")
st.markdown("""
""")