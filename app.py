import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import numpy as np
import xarray as xr  # ADD THIS
from scipy import stats
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

# Load data function
@st.cache_data
def load_data():
    
    project_folder = 'filtered_1974_2024'
    local_folder = 'heat_index_files'
    
  
    if os.path.exists(os.path.join(project_folder, 'KAVLheatindex19742024.xlsx')):
        folder = project_folder
        filenames = {
            'KAVL': 'KAVLheatindex19742024.xlsx',
            'KGSO': 'KGSOheatindex19742024.xlsx',
            'KHSE': 'KHSEheatindex19742024.xlsx',
            'KILM': 'KILMheatindex19742024.xlsx',
            'KCLT': 'KLCTheatindex19742024.xlsx',
            'KRDU': 'KRDUheatindex19742024.xlsx',
        }
    else:
        folder = local_folder
        filenames = {
            'KAVL': 'KAVL-heatindex-1971-2021.xlsx',
            'KGSO': 'KGSO-heatindex-1974-2024.xlsx',
            'KHSE': 'KHSE-heatindex-1974-2024.xlsx',
            'KILM': 'KILM-heatindex-1974-2024.xlsx',
            'KCLT': 'KLCT-heatindex-1974-2024.xlsx',
            'KRDU': 'KRDU-heat-index-1974-2024.xlsx',
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
                df[col] = (df[col] - 32) * 5/9  # Convert F to C
        
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

ADDITIONAL_REGIONS = {
    'east_africa': {
        'name': 'East Africa',
        'full_name': 'East Africa (Uganda, Kenya, Tanzania)',
        'file': 'era5_east_africa_1974_2024.nc',
        'area': [5, 29, -12, 42],  # [North, West, South, East]
        'cities': {
            'KAMPALA': (0.3476, 32.5825, 'Kampala'),
            'NAIROBI': (-1.2921, 36.8219, 'Nairobi'),
            'MOMBASA': (-4.0435, 39.6682, 'Mombasa'),
            'DAR': (-6.7924, 39.2083, 'Dar es Salaam'),
            'ARUSHA': (-3.3869, 36.6830, 'Arusha'),
            'KISUMU': (-0.1022, 34.7617, 'Kisumu'),
        },
        'seasons': {
            "Hot Season (Dec-Feb)": [12, 1, 2],
            "Long Rains (Mar-May)": [3, 4, 5],
            "Cool Dry (Jun-Aug)": [6, 7, 8],
            "Short Rains (Sep-Nov)": [9, 10, 11],
            "Annual Average": list(range(1, 13))
        }
    },
    'madagascar': {
        'name': 'Madagascar',
        'full_name': 'Madagascar',
        'file': 'era5_madagascar_1974_2024.nc',
        'area': [-11.5, 43, -26, 51],  # [North, West, South, East]
        'cities': {
            'ANTANANARIVO': (-18.8792, 47.5079, 'Antananarivo'),
            'TOAMASINA': (-18.1443, 49.3958, 'Toamasina'),
            'MAHAJANGA': (-15.7167, 46.3167, 'Mahajanga'),
            'FIANARANTSOA': (-21.4417, 47.0833, 'Fianarantsoa'),
            'TOLIARA': (-23.3500, 43.6667, 'Toliara'),
        },
        'seasons': {
            "Hot Wet (Dec-Feb)": [12, 1, 2],
            "Autumn (Mar-May)": [3, 4, 5],
            "Cool Dry (Jun-Aug)": [6, 7, 8],
            "Spring (Sep-Nov)": [9, 10, 11],
            "Annual Average": list(range(1, 13))
        }
    },
    'brazil': {
        'name': 'Brazil',
        'full_name': 'Brazil',
        'file': 'era5_brazil_1974_2024.nc',
        'area': [6, -74, -34, -34],  # [North, West, South, East]
        'cities': {
            'SAO_PAULO': (-23.5505, -46.6333, 'SÃ£o Paulo'),
            'RIO': (-22.9068, -43.1729, 'Rio de Janeiro'),
            'BRASILIA': (-15.7975, -47.8919, 'BrasÃ­lia'),
            'SALVADOR': (-12.9714, -38.5014, 'Salvador'),
            'MANAUS': (-3.1190, -60.0217, 'Manaus'),
            'RECIFE': (-8.0476, -34.8770, 'Recife'),
            'BELEM': (-1.4558, -48.4902, 'BelÃ©m'),
        },
        'seasons': {
            "Southern Summer (Dec-Feb)": [12, 1, 2],
            "Autumn (Mar-May)": [3, 4, 5],
            "Southern Winter (Jun-Aug)": [6, 7, 8],
            "Spring (Sep-Nov)": [9, 10, 11],
            "Annual Average": list(range(1, 13))
        }
    }
}



def identify_waves(df, column_name='extreme_event', min_consecutive=2):
    """
    Identify heatwaves/coldwaves (consecutive extreme events)
    """
    df = df.sort_values('datetime').copy()
    df['wave_id'] = (df[column_name] != df[column_name].shift()).cumsum()
    df['wave'] = df.groupby('wave_id')[column_name].transform('sum') >= min_consecutive
    df['in_wave'] = df[column_name] & df['wave']
    return df

def analyze_date_specific_exceedances(df, date_str):
    """
    Analyze how many times a specific calendar date exceeded/fell below its percentile threshold
    
    Args:
        df: DataFrame with extreme event flags and month_day column
        date_str: String like '06-15' for June 15
    
    Returns:
        Dictionary with analysis results
    """
    specific_date_data = df[df['month_day'] == date_str].copy()
    
    if len(specific_date_data) == 0:
        return None
    
    # Count exceedances
    times_exceeded = specific_date_data['extreme_event'].sum() if 'extreme_event' in specific_date_data.columns else 0
    total_years = len(specific_date_data)
    percentage = (times_exceeded / total_years * 100) if total_years > 0 else 0
    
    # Get threshold value (find column starting with 'threshold_')
    threshold_cols = [col for col in specific_date_data.columns if col.startswith('threshold_')]
    threshold = specific_date_data[threshold_cols[0]].iloc[0] if threshold_cols and len(specific_date_data) > 0 else None
    
    # Get actual values from appropriate column
    if 'heatindexmax2m' in specific_date_data.columns:
        actual_values = specific_date_data['heatindexmax2m'].dropna()
    elif 'heatindexmin2m' in specific_date_data.columns:
        actual_values = specific_date_data['heatindexmin2m'].dropna()
    else:
        actual_values = pd.Series()
    
    return {
        'date': date_str,
        'times_exceeded': int(times_exceeded),
        'total_years': total_years,
        'percentage': percentage,
        'threshold': threshold,
        'mean': actual_values.mean() if len(actual_values) > 0 else None,
        'max': actual_values.max() if len(actual_values) > 0 else None,
        'min': actual_values.min() if len(actual_values) > 0 else None
    }


def apply_loess_smoothing(x, y, frac=0.2):
    """
    Apply LOESS (Locally Weighted Scatterplot Smoothing) to data
    
    Args:
        x: x values (e.g., years)
        y: y values (e.g., counts)
        frac: fraction of data points to use for smoothing (default 0.2, same as R's f=1/5)
    
    Returns:
        smoothed y values
    """
    from scipy.signal import savgol_filter
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    # Remove NaN values
    mask = ~np.isnan(y)
    x_clean = np.array(x)[mask]
    y_clean = np.array(y)[mask]
    
    if len(x_clean) < 3:
        return y
    
    # Apply LOESS smoothing
    smoothed = lowess(y_clean, x_clean, frac=frac, return_sorted=False)
    
    # Create full array with original length, inserting NaN where data was missing
    result = np.full(len(y), np.nan)
    result[mask] = smoothed
    
    return result

# ============================================================================
# ERA5 DATA PROCESSING FUNCTIONS
# ============================================================================

# ============================================================================
# ERA5 DATA PROCESSING FUNCTIONS
# ============================================================================

def process_era5_data(era5_file='era5_temperature_nc_1974_2024.nc'):
    """
    Process ERA5 NetCDF and load temperature data
    """
    try:
        # Open dataset
        ds = xr.open_dataset(era5_file)
        
        # Get temperature (t2m) and convert K to Â°C
        temp = ds['t2m']
        temp_c = temp - 273.15
        
        # Create a copy with proper time handling
        temp_c = temp_c.copy()
        
        # Extract year and month from valid_time and add as new coordinates
        years = temp_c['valid_time'].dt.year.values
        months = temp_c['valid_time'].dt.month.values
        
        # Add year and month as coordinates (not dimensions)
        temp_c.coords['year'] = ('valid_time', years)
        temp_c.coords['month'] = ('valid_time', months)
        
        return temp_c, ds
        
    except Exception as e:
        st.error(f"Error loading ERA5 data: {str(e)}")
        st.code(f"""
Error details:
{type(e).__name__}: {str(e)}

Traceback:
{__import__('traceback').format_exc()}
        """)
        return None, None


def calculate_temperature_trends(temp_data, months=[6,7,8]):
    """
    Calculate temperature change (slope) for each grid cell
    
    Args:
        temp_data: xarray DataArray with temperature
        months: list of months to analyze (default: summer [6,7,8])
    
    Returns:
        lats, lons, slopes (Â°F/decade), r_squared, p_values
    """
    try:
        # Create a boolean mask for the months we want
        month_mask = temp_data['month'].isin(months)
        
        # Filter using the mask
        seasonal_temp = temp_data.where(month_mask, drop=True)
        
        # Calculate annual seasonal averages
        # Group by year and take mean across valid_time dimension
        annual_avg = seasonal_temp.groupby('year').mean(dim='valid_time')
        
        # Get coordinates
        lats = annual_avg.latitude.values
        lons = annual_avg.longitude.values
        years = annual_avg.year.values
        
        # Initialize arrays
        slopes = np.full((len(lats), len(lons)), np.nan)
        r_squared = np.full((len(lats), len(lons)), np.nan)
        p_values = np.full((len(lats), len(lons)), np.nan)
        
        # Calculate slopes with progress indicator
        progress_bar = st.progress(0)
        total_cells = len(lats) * len(lons)
        processed = 0
        
        for i in range(len(lats)):
            for j in range(len(lons)):
                # Get temperature time series for this grid cell
                temps = annual_avg.isel(latitude=i, longitude=j).values
                
                # Only calculate if we have valid data
                if not np.isnan(temps).any() and len(temps) > 2:
                    try:
                        slope, intercept, r_val, p_val, std_err = stats.linregress(years, temps)
                        slopes[i, j] = slope * 10  # Convert to Â°F per decade
                        r_squared[i, j] = r_val ** 2
                        p_values[i, j] = p_val
                    except:
                        pass
                
                processed += 1
                if processed % 100 == 0:
                    progress_bar.progress(processed / total_cells)
        
        progress_bar.progress(1.0)
        
        return lats, lons, slopes, r_squared, p_values
    
    except Exception as e:
        st.error(f"Error calculating trends: {str(e)}")
        st.code(f"""
Error details:
{type(e).__name__}: {str(e)}

Traceback:
{__import__('traceback').format_exc()}
        """)
        return None, None, None, None, None


def get_temperature_snapshot(temp_data, year=2020, months=[6,7,8]):
    """
    Get average temperature for a specific year and season
    
    Args:
        temp_data: xarray DataArray with temperature
        year: specific year to analyze
        months: list of months for the season
    
    Returns:
        lats, lons, average temperature values
    """
    try:
        # Create boolean masks
        year_mask = temp_data['year'] == year
        month_mask = temp_data['month'].isin(months)
        
        # Apply both masks
        subset = temp_data.where(year_mask & month_mask, drop=True)
        
        if len(subset['valid_time']) == 0:
            st.warning(f"No data found for year {year}, months {months}")
            return None, None, None
        
        # Average over the valid_time dimension
        avg_temp = subset.mean(dim='valid_time')
        
        # Get coordinates
        lats = avg_temp.latitude.values
        lons = avg_temp.longitude.values
        temps = avg_temp.values
        
        return lats, lons, temps
    
    except Exception as e:
        st.error(f"Error getting snapshot: {str(e)}")
        st.code(f"""
Error details:
{type(e).__name__}: {str(e)}

Traceback:
{__import__('traceback').format_exc()}
        """)
        return None, None, None

def interpolate_grid(lats, lons, data, factor=3):
    """
    Interpolate data to a finer grid for smoother visualization
    
    Args:
        lats, lons: original grid coordinates
        data: 2D array of values
        factor: upsampling factor (3 = 3x more resolution)
    
    Returns:
        new_lats, new_lons, interpolated_data
    """
    from scipy.interpolate import griddata
    
    # Create finer grid
    lat_fine = np.linspace(lats.min(), lats.max(), len(lats) * factor)
    lon_fine = np.linspace(lons.min(), lons.max(), len(lons) * factor)
    lon_grid, lat_grid = np.meshgrid(lon_fine, lat_fine)
    
    # Original grid points
    lons_mesh, lats_mesh = np.meshgrid(lons, lats)
    points = np.column_stack([lats_mesh.ravel(), lons_mesh.ravel()])
    values = data.ravel()
    
    # Remove NaN values
    mask = ~np.isnan(values)
    points = points[mask]
    values = values[mask]
    
    # Interpolate
    data_fine = griddata(points, values, (lat_grid, lon_grid), method='cubic')
    
    return lat_fine, lon_fine, data_fine

def create_matplotlib_heatmap_inline(lats_mesh, lons_mesh, data, station_coords=None,
                                      title='', cbar_label='', cmap='RdBu_r',
                                      vmin=None, vmax=None, diverging=False, dpi=150):
    """
    Create a publication-quality matplotlib heatmap with state borders
    
    Args:
        lats_mesh, lons_mesh: 2D meshgrids of coordinates
        data: 2D array of values to plot
        station_coords: dict of {code: (lat, lon, name)} for stations (optional)
        title: plot title
        cbar_label: colorbar label
        cmap: colormap name
        vmin, vmax: colorbar limits (optional)
        diverging: if True, center colorbar at 0
        dpi: resolution for the figure
    
    Returns:
        matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
    
    # Handle diverging colormaps
    if diverging and vmin is None and vmax is None:
        abs_max = np.nanmax(np.abs(data))
        vmin = -abs_max
        vmax = abs_max
    
    # Create contour plot
    contour = ax.contourf(lons_mesh, lats_mesh, data, 
                          levels=50, cmap=cmap, 
                          vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', 
                        pad=0.02, fraction=0.046)
    cbar.set_label(cbar_label, fontsize=11, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add state borders
    try:
        import json
        import urllib.request
        
        url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
        with urllib.request.urlopen(url, timeout=10) as response:
            states = json.load(response)
        
        # Get data extent
        lat_min, lat_max = np.nanmin(lats_mesh), np.nanmax(lats_mesh)
        lon_min, lon_max = np.nanmin(lons_mesh), np.nanmax(lons_mesh)
        
        # Plot state borders
        for feature in states['features']:
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                
                # Only plot if within data extent
                if any(lat_min <= lat <= lat_max and lon_min <= lon <= lon_max 
                       for lat, lon in zip(lats, lons)):
                    ax.plot(lons, lats, 'k-', linewidth=0.8, alpha=0.6, zorder=3)
            
            elif feature['geometry']['type'] == 'MultiPolygon':
                for polygon in feature['geometry']['coordinates']:
                    coords = polygon[0]
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    
                    if any(lat_min <= lat <= lat_max and lon_min <= lon <= lon_max 
                           for lat, lon in zip(lats, lons)):
                        ax.plot(lons, lats, 'k-', linewidth=0.8, alpha=0.6, zorder=3)
    except:
        pass  # Continue without borders if download fails
    
    # Add stations if provided
    if station_coords:
        for code, (lat, lon, name) in station_coords.items():
            ax.plot(lon, lat, 'ko', markersize=8, markeredgecolor='white', 
                    markeredgewidth=2, zorder=4)
            
            # Add text with white outline
            txt = ax.text(lon, lat + 0.15, name, fontsize=9, ha='center', 
                         weight='bold', zorder=5)
            txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                                  path_effects.Normal()])
    
    # Format axes
    ax.set_xlabel('Longitude', fontsize=12, weight='bold')
    ax.set_ylabel('Latitude', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', pad=15)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set aspect ratio and limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([np.nanmin(lons_mesh), np.nanmax(lons_mesh)])
    ax.set_ylim([np.nanmin(lats_mesh), np.nanmax(lats_mesh)])
    
    plt.tight_layout()
    
    return fig

def add_state_borders(fig, lat_range=None, lon_range=None):
    """
    Add US state borders to a plotly figure, clipped to the data extent
    
    Args:
        fig: plotly figure object to add borders to
        lat_range: tuple of (min_lat, max_lat) to clip borders
        lon_range: tuple of (min_lon, max_lon) to clip borders
    
    Returns:
        fig: modified figure with state borders
    """
    import json
    import urllib.request
    
    try:
        # Load US states GeoJSON
        url = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'
        with urllib.request.urlopen(url) as response:
            states_geojson = json.load(response)
        
        # Focus on Southeast states around North Carolina
        target_states = ['North Carolina', 'South Carolina', 'Virginia', 
                        'Tennessee', 'Georgia', 'West Virginia', 'Kentucky',
                        'Alabama', 'Maryland', 'Delaware', 'Florida']
        
        # If ranges not provided, use reasonable defaults for NC region
        if lat_range is None:
            lat_range = (24, 40)
        if lon_range is None:
            lon_range = (-93, -75)
        
        min_lat, max_lat = lat_range
        min_lon, max_lon = lon_range
        
        def clip_point(lon, lat):
            """Check if point is within bounds"""
            return (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat)
        
        # Extract and plot borders for each state
        for feature in states_geojson['features']:
            state_name = feature['properties']['name']
            
            if state_name in target_states:
                geometry = feature['geometry']
                
                if geometry['type'] == 'Polygon':
                    coords_list = [geometry['coordinates']]
                elif geometry['type'] == 'MultiPolygon':
                    coords_list = geometry['coordinates']
                else:
                    continue
                
                # Plot each polygon, but only segments within bounds
                for coords in coords_list:
                    for ring in coords:
                        # Filter coordinates to only those within bounds
                        lons_full = [coord[0] for coord in ring]
                        lats_full = [coord[1] for coord in ring]
                        
                        # Create line segments, only keeping those in bounds
                        lons_clipped = []
                        lats_clipped = []
                        
                        for i in range(len(lons_full)):
                            lon, lat = lons_full[i], lats_full[i]
                            
                            # Check if point or next point is in bounds
                            in_bounds = clip_point(lon, lat)
                            
                            if in_bounds:
                                lons_clipped.append(lon)
                                lats_clipped.append(lat)
                            else:
                                # If we have accumulated points, plot them
                                if len(lons_clipped) > 1:
                                    fig.add_trace(go.Scatter(
                                        x=lons_clipped,
                                        y=lats_clipped,
                                        mode='lines',
                                        line=dict(color='black', width=1),
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ))
                                lons_clipped = []
                                lats_clipped = []
                        
                        # Plot any remaining segments
                        if len(lons_clipped) > 1:
                            fig.add_trace(go.Scatter(
                                x=lons_clipped,
                                y=lats_clipped,
                                mode='lines',
                                line=dict(color='black', width=1),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not load state borders: {e}")
        return fig

def process_era5_data_for_region(era5_file):
    """
    Process ERA5 NetCDF for additional regions in Celsius
    Handles both 'time' and 'valid_time' dimension names
    """
    try:
        ds = xr.open_dataset(era5_file)
        
        # Get temperature and convert K to Â°C
        temp = ds['t2m']
        temp_c = temp - 273.15
        temp_c = temp_c.copy()
        
        # Determine time dimension name
        time_dim = 'valid_time' if 'valid_time' in temp_c.dims else 'time'
        
        years = temp_c[time_dim].dt.year.values
        months = temp_c[time_dim].dt.month.values
        
        temp_c.coords['year'] = (time_dim, years)
        temp_c.coords['month'] = (time_dim, months)
        temp_c.attrs['time_dim'] = time_dim
        
        return temp_c, ds
        
    except Exception as e:
        st.error(f"Error loading ERA5 data: {str(e)}")
        return None, None


def calculate_temperature_trends_region(temp_data, months=[6,7,8]):
    """
    Calculate temperature change (slope) for each grid cell in Â°C/decade
    Works with any region's ERA5 data
    """
    try:
        time_dim = temp_data.attrs.get('time_dim', 'valid_time')
        if time_dim not in temp_data.dims:
            time_dim = 'time' if 'time' in temp_data.dims else 'valid_time'
        
        month_mask = temp_data['month'].isin(months)
        seasonal_temp = temp_data.where(month_mask, drop=True)
        annual_avg = seasonal_temp.groupby('year').mean(dim=time_dim)
        
        lats = annual_avg.latitude.values
        lons = annual_avg.longitude.values
        years = annual_avg.year.values
        
        slopes = np.full((len(lats), len(lons)), np.nan)
        r_squared = np.full((len(lats), len(lons)), np.nan)
        p_values = np.full((len(lats), len(lons)), np.nan)
        
        progress_bar = st.progress(0)
        total_cells = len(lats) * len(lons)
        processed = 0
        
        for i in range(len(lats)):
            for j in range(len(lons)):
                temps = annual_avg.isel(latitude=i, longitude=j).values
                
                if not np.isnan(temps).any() and len(temps) > 2:
                    try:
                        slope, intercept, r_val, p_val, std_err = stats.linregress(years, temps)
                        slopes[i, j] = slope * 10  # Â°C per decade
                        r_squared[i, j] = r_val ** 2
                        p_values[i, j] = p_val
                    except:
                        pass
                
                processed += 1
                if processed % 100 == 0:
                    progress_bar.progress(min(processed / total_cells, 1.0))
        
        progress_bar.progress(1.0)
        
        return lats, lons, slopes, r_squared, p_values
    
    except Exception as e:
        st.error(f"Error calculating trends: {str(e)}")
        return None, None, None, None, None


def get_temperature_snapshot_region(temp_data, year=2020, months=[6,7,8]):
    """
    Get average temperature for a specific year and season in Â°C
    Works with any region's ERA5 data
    """
    try:
        time_dim = temp_data.attrs.get('time_dim', 'valid_time')
        if time_dim not in temp_data.dims:
            time_dim = 'time' if 'time' in temp_data.dims else 'valid_time'
        
        year_mask = temp_data['year'] == year
        month_mask = temp_data['month'].isin(months)
        
        subset = temp_data.where(year_mask & month_mask, drop=True)
        
        if len(subset[time_dim]) == 0:
            return None, None, None
        
        avg_temp = subset.mean(dim=time_dim)
        
        lats = avg_temp.latitude.values
        lons = avg_temp.longitude.values
        temps = avg_temp.values
        
        return lats, lons, temps
    
    except Exception as e:
        st.error(f"Error getting snapshot: {str(e)}")
        return None, None, None


def create_heatmap_with_borders(lats_mesh, lons_mesh, data, station_coords=None,
                                 title='', cbar_label='', cmap='coolwarm',
                                 vmin=None, vmax=None, diverging=False, dpi=150,
                                 geojson_url=None, target_countries=None):
    """
    Create a publication-quality matplotlib heatmap with country/state borders
    Uses cool colors (blue) for cool temperatures, warm colors (red) for warm temps
    """
    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
    
    if diverging and vmin is None and vmax is None:
        abs_max = np.nanmax(np.abs(data))
        vmin = -abs_max
        vmax = abs_max
    
    contour = ax.contourf(lons_mesh, lats_mesh, data, 
                          levels=50, cmap=cmap, 
                          vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', 
                        pad=0.02, fraction=0.046)
    cbar.set_label(cbar_label, fontsize=11, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add country borders if URL provided
    if geojson_url:
        try:
            import json
            import urllib.request
            
            with urllib.request.urlopen(geojson_url, timeout=15) as response:
                borders = json.load(response)
            
            lat_min, lat_max = np.nanmin(lats_mesh), np.nanmax(lats_mesh)
            lon_min, lon_max = np.nanmin(lons_mesh), np.nanmax(lons_mesh)
            
            for feature in borders['features']:
                if target_countries:
                    name = feature['properties'].get('name', '') or feature['properties'].get('ADMIN', '')
                    if name not in target_countries:
                        continue
                
                geom_type = feature['geometry']['type']
                
                if geom_type == 'Polygon':
                    for ring in feature['geometry']['coordinates']:
                        lons_border = [c[0] for c in ring]
                        lats_border = [c[1] for c in ring]
                        if any(lat_min <= lat <= lat_max and lon_min <= lon <= lon_max 
                               for lat, lon in zip(lats_border, lons_border)):
                            ax.plot(lons_border, lats_border, 'k-', linewidth=0.8, alpha=0.6, zorder=3)
                
                elif geom_type == 'MultiPolygon':
                    for polygon in feature['geometry']['coordinates']:
                        for ring in polygon:
                            lons_border = [c[0] for c in ring]
                            lats_border = [c[1] for c in ring]
                            if any(lat_min <= lat <= lat_max and lon_min <= lon <= lon_max 
                                   for lat, lon in zip(lats_border, lons_border)):
                                ax.plot(lons_border, lats_border, 'k-', linewidth=0.8, alpha=0.6, zorder=3)
        except Exception as e:
            pass  # Continue without borders
    
    # Add city markers
    if station_coords:
        for code, (lat, lon, name) in station_coords.items():
            ax.plot(lon, lat, 'ko', markersize=8, markeredgecolor='white', 
                    markeredgewidth=2, zorder=4)
            
            txt = ax.text(lon, lat + 0.4, name, fontsize=9, ha='center', 
                         weight='bold', zorder=5)
            txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                                  path_effects.Normal()])
    
    ax.set_xlabel('Longitude', fontsize=12, weight='bold')
    ax.set_ylabel('Latitude', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', pad=15)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([np.nanmin(lons_mesh), np.nanmax(lons_mesh)])
    ax.set_ylim([np.nanmin(lats_mesh), np.nanmax(lats_mesh)])
    
    plt.tight_layout()
    
    return fig

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
st.title("North Carolina Heat Index Analysis (1974-2024)")
# st.markdown("""
# **Redefining Extreme Heat and Cold Events for Emergency Healthcare**

# This analysis examines extreme temperature trends across six weather stations in North Carolina using a 
# date-specific percentile methodology that adapts to local climate conditions, as proposed in our research paper.

# **Note:** All measurements use Heat Index (apparent temperature combining air temperature and humidity), 
# not raw temperature. Heat index better represents how hot it actually feels and the health risks to humans.
# """)



# Tabs for different visualizations
tab1, tab2, tab3, tab4, tab5, tab6= st.tabs([
    "Methodology",
    "Extreme Temperature Days", 
    "Heatwaves & Coldwaves",
    "WWA Comparison",
    "Regional Heatmap",
    "Additional Heat Maps"
])

# TAB 1: METHODOLOGY
# Enhanced Tab 1 - Methodology Section
# This replaces the section starting around line 706 in the original app.py

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
    1. Collect all June 15th values from 1974-2024 (51 years)
    2. Calculate the 98th percentile (heat) and 2nd percentile (cold) of those 51 values
    3. Any June 15th exceeding its specific threshold is an "extreme heat/cold day"
    4. Two or more consecutive extreme days = "heatwave/coldwave"
    """)
    
    # Enhanced Key Metrics for Methodology
    st.subheader("Extreme Events Comparison: Early vs Recent Decades")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate extreme heat days (March-October, not just summer!)
    march_to_october = all_data[all_data['month'].isin([3,4,5,6,7,8,9,10])].copy()
    heat_extreme = calculate_date_specific_percentiles(march_to_october, 'heatindexmax2m', 0.98)
    
    early_heat = heat_extreme[(heat_extreme['year'] >= 1974) & (heat_extreme['year'] <= 1983)]
    recent_heat = heat_extreme[(heat_extreme['year'] >= 2015) & (heat_extreme['year'] <= 2024)]
    
    early_heat_days = early_heat['extreme_event'].sum()
    recent_heat_days = recent_heat['extreme_event'].sum()
    
    # Calculate extreme cold days (winter)
    winter_all = all_data[all_data['month'].isin([11,12,1,2])].copy()
    winter_extreme = calculate_date_specific_percentiles(winter_all, 'heatindexmin2m', 0.02)
    
    early_cold = winter_extreme[(winter_extreme['year'] >= 1974) & (winter_extreme['year'] <= 1983)]
    recent_cold = winter_extreme[(winter_extreme['year'] >= 2015) & (winter_extreme['year'] <= 2024)]
    
    early_cold_days = early_cold['extreme_event'].sum()
    recent_cold_days = recent_cold['extreme_event'].sum()
    
    with col1:
        st.metric(
            "Extreme Heat Days (1974-1983)",
            f"{early_heat_days:,}",
            help="Days exceeding 98th percentile (March-October)"
        )
    
    with col2:
        increase = recent_heat_days - early_heat_days
        pct_increase = (increase / early_heat_days * 100) if early_heat_days > 0 else 0
        st.metric(
            "Extreme Heat Days (2015-2024)",
            f"{recent_heat_days:,}",
            delta=f"+{increase:,} days ({pct_increase:.1f}%)",
            delta_color="inverse",
            help="Shows increase in extreme heat events"
        )
    
    with col3:
        st.metric(
            "Extreme Cold Days (1974-1983)",
            f"{early_cold_days:,}",
            help="Days below 2nd percentile (November-February)"
        )
    
    with col4:
        decrease = recent_cold_days - early_cold_days
        pct_decrease = (decrease / early_cold_days * 100) if early_cold_days > 0 else 0
        st.metric(
            "Extreme Cold Days (2015-2024)",
            f"{recent_cold_days:,}",
            delta=f"{decrease:,} days ({pct_decrease:.1f}%)",
            delta_color="normal",
            help="Shows decrease in extreme cold events"
        )
    
    st.markdown("---")
    
    # ===========================================================================
    # NEW FEATURE: Date-Specific Exceedance Analysis
    # ===========================================================================

    
    # Original date distribution visualization continues below...
    st.subheader("Explore Any Date's Distribution (March - October)")
    
    # ... [REST OF ORIGINAL TAB 1 CODE CONTINUES HERE] ...
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
            st.metric("Mean", f"{mean_val:.1f}°C")
        with col2:
            st.metric("98th Percentile", f"{p98:.1f}°C")
        with col3:
            temp_range = demo_data['heatindexmax2m'].max() - demo_data['heatindexmax2m'].min()
            st.metric("Temperature Range", f"{temp_range:.1f}°C")
        
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
            annotation_text=f"98th Percentile: {p98:.1f}°C",
            annotation_position="top"
        )
        
        # Add mean line
        fig_summer.add_vline(
            x=mean_val, 
            line_dash="dot", 
            line_color="green", 
            line_width=2,
            annotation_text=f"Mean: {mean_val:.1f}°C",
            annotation_position="bottom"
        )
        
        fig_summer.update_layout(
            title=f"Summer: {selected_summer_date.strftime('%B %d')} Heat Index Distribution at {dfs[demo_station]['station_name'].iloc[0]}<br><sub>Showing {len(demo_data)} years of data (1974-2024)</sub>",
            xaxis_title=f"Temperature of {selected_summer_date.strftime('%B %d')} Across All Years (°C)",
            yaxis_title="Frequency (Probability)",
            height=450,
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_summer, use_container_width=True, key="summer_methodology")
        
        st.info(f"""
        **Heat Threshold:** Any {selected_summer_date.strftime('%B %d')} with heat index > {p98:.1f}°C is an extreme heat day.
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
            st.metric("Mean", f"{mean_val_winter:.1f}°C")
        with col2:
            st.metric("2nd Percentile", f"{p2:.1f}°C")
        with col3:
            temp_range_winter = winter_data['heatindexmin2m'].max() - winter_data['heatindexmin2m'].min()
            st.metric("Temperature Range", f"{temp_range_winter:.1f}°C")
        
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
            annotation_text=f"2nd Percentile: {p2:.1f}°C",
            annotation_position="top"
        )
        
        # Add mean line
        fig_winter.add_vline(
            x=mean_val_winter, 
            line_dash="dot", 
            line_color="green", 
            line_width=2,
            annotation_text=f"Mean: {mean_val_winter:.1f}°C",
            annotation_position="bottom"
        )
        
        fig_winter.update_layout(
            title=f"Winter: {selected_winter_date.strftime('%B %d')} Heat Index Distribution at {dfs[demo_station]['station_name'].iloc[0]}<br><sub>Showing {len(winter_data)} years of data (1974-2024)</sub>",
            xaxis_title=f"Temperature of {selected_winter_date.strftime('%B %d')} Across All Years (°C)",
            yaxis_title="Frequency (Probability)",
            height=450,
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_winter, use_container_width=True, key="winter_methodology")
        
        st.info(f"""
        **Cold Threshold:** Any {selected_winter_date.strftime('%B %d')} with heat index < {p2:.1f}°C is an extreme cold day
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
            summer = df[(df['month'].isin([3,4,5,6,7,8,9,10]))].copy()
            
            # Calculate for max (daytime)
            max_extreme = calculate_date_specific_percentiles(summer.dropna(subset=['heatindexmax2m']), 
                                                              'heatindexmax2m', 0.98)
            max_counts = max_extreme.groupby('year')['extreme_event'].sum()
            
            # Calculate for min (nighttime)
            min_extreme = calculate_date_specific_percentiles(summer.dropna(subset=['heatindexmin2m']), 
                                                              'heatindexmin2m', 0.98)
            min_counts = min_extreme.groupby('year')['extreme_event'].sum()
            
            for year in range(1974, 2024):  # Changed to 2024 (exclusive) = 1974-2023
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
            y_title="Count"
        )
        
        for idx, station in enumerate(selected_stations):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            station_data = extreme_heat_df[extreme_heat_df['station'] == station].sort_values('year')
            
            # Apply LOESS smoothing (frac=0.2 matches R's f=1/5)
            max_smoothed = apply_loess_smoothing(station_data['year'].values, 
                                                  station_data['max_extreme'].values, 
                                                  frac=0.2)
            min_smoothed = apply_loess_smoothing(station_data['year'].values, 
                                                  station_data['min_extreme'].values, 
                                                  frac=0.2)
            
            # Calculate linear regression for Max HI
            from scipy import stats as scipy_stats
            slope_max, intercept_max, _, _, _ = scipy_stats.linregress(station_data['year'].values, station_data['max_extreme'].values)
            regression_max = slope_max * station_data['year'].values + intercept_max
            
            slope_min, intercept_min, _, _, _ = scipy_stats.linregress(station_data['year'].values, station_data['min_extreme'].values)
            regression_min = slope_min * station_data['year'].values + intercept_min
            
            # Data points for Max HI (darker red, semi-transparent)
            fig_heat.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['max_extreme'],
                    name="Max HI data", 
                    mode='markers', 
                    marker=dict(color='darkred', size=4, opacity=0.4),
                    legendgroup="heat_max_data", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Max HI (darker red) - LOESS smoothed
            fig_heat.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=max_smoothed,
                    name="Max HI > 98p (LOESS)", 
                    mode='lines', 
                    line=dict(color='darkred', width=2.5),
                    legendgroup="heat_max", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Max HI regression line
            fig_heat.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=regression_max,
                    name="Max HI trend", 
                    mode='lines', 
                    line=dict(color='darkred', width=1.5, dash='dash'),
                    legendgroup="heat_max_reg", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Data points for Min HI (orange, semi-transparent)
            fig_heat.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['min_extreme'],
                    name="Min HI data", 
                    mode='markers', 
                    marker=dict(color='#FFA500', size=4, opacity=0.4),
                    legendgroup="heat_min_data", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Min HI (orange) - LOESS smoothed
            fig_heat.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=min_smoothed,
                    name="Min HI > 98p (LOESS)", 
                    mode='lines', 
                    line=dict(color='#FFA500', width=2.5),
                    legendgroup="heat_min", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Min HI regression line
            fig_heat.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=regression_min,
                    name="Min HI trend", 
                    mode='lines', 
                    line=dict(color='#FFA500', width=1.5, dash='dash'),
                    legendgroup="heat_min_reg", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
        
        # Find max value across all stations for consistent y-axis
        max_y_heat = max([
            extreme_heat_df[extreme_heat_df['station'] == s]['max_extreme'].max()
            for s in selected_stations
        ] + [
            extreme_heat_df[extreme_heat_df['station'] == s]['min_extreme'].max()
            for s in selected_stations
        ])
        
        fig_heat.update_layout(
            title_text="Extreme Heat Days by Station (1974-2023)<br><sup>Days exceeding date-specific 98th percentile (March-October)</sup>",
            height=800,
            showlegend=True,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            )
        )
        
        fig_heat.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_heat.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            range=[0, 20],
            title_text="Count"
        )
        
        st.plotly_chart(fig_heat, use_container_width=True, key="extreme_heat_facet")
        
        # Trend analysis
        st.subheader("Extreme Heat Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Daytime Extreme Heat (Max HI):**")
            for station in selected_stations[:3]:
                station_data = extreme_heat_df[extreme_heat_df['station'] == station]
                early = station_data[station_data['year'] <= 1993]['max_extreme'].mean()
                recent = station_data[station_data['year'] >= 2010]['max_extreme'].mean()
                change = ((recent - early) / early * 100) if early > 0 else 0
                st.write(f"**{station_data['station_name'].iloc[0]}**: {change:+.1f}% change")
        
        with col2:
            st.markdown("**Nighttime Extreme Heat (Min HI):**")
            for station in selected_stations[3:]:
                station_data = extreme_heat_df[extreme_heat_df['station'] == station]
                early = station_data[station_data['year'] <= 1993]['min_extreme'].mean()
                recent = station_data[station_data['year'] >= 2010]['min_extreme'].mean()
                change = ((recent - early) / early * 100) if early > 0 else 0
                st.write(f"**{station_data['station_name'].iloc[0]}**: {change:+.1f}% change")
    
    else:  # Extreme Cold Days
        st.subheader("Extreme Cold Days by Station (November - February)")
        
        # Calculate extreme cold days for all stations
        extreme_cold_data = []
        
        for station in selected_stations:
            df = dfs[station]
            winter = df[(df['month'].isin([11,12,1,2]))].copy()
            
            # Calculate for max (daytime)
            max_extreme = calculate_date_specific_percentiles(winter.dropna(subset=['heatindexmax2m']), 
                                                              'heatindexmax2m', 0.02)
            max_counts = max_extreme.groupby('year')['extreme_event'].sum()
            
            # Calculate for min (nighttime)
            min_extreme = calculate_date_specific_percentiles(winter.dropna(subset=['heatindexmin2m']), 
                                                              'heatindexmin2m', 0.02)
            min_counts = min_extreme.groupby('year')['extreme_event'].sum()
            
            for year in range(1974, 2024):  # Changed to 2024 (exclusive) = 1974-2023
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
            y_title="Count"
        )
        
        for idx, station in enumerate(selected_stations):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            station_data = extreme_cold_df[extreme_cold_df['station'] == station].sort_values('year')
            
            # Apply LOESS smoothing (frac=0.2 matches R's f=1/5)
            max_smoothed = apply_loess_smoothing(station_data['year'].values, 
                                                  station_data['max_extreme'].values, 
                                                  frac=0.2)
            min_smoothed = apply_loess_smoothing(station_data['year'].values, 
                                                  station_data['min_extreme'].values, 
                                                  frac=0.2)
            
            # Calculate linear regression
            from scipy import stats as scipy_stats
            slope_max, intercept_max, _, _, _ = scipy_stats.linregress(station_data['year'].values, station_data['max_extreme'].values)
            regression_max = slope_max * station_data['year'].values + intercept_max
            
            slope_min, intercept_min, _, _, _ = scipy_stats.linregress(station_data['year'].values, station_data['min_extreme'].values)
            regression_min = slope_min * station_data['year'].values + intercept_min
            
            # Data points for Max HI (navy, semi-transparent)
            fig_cold.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['max_extreme'],
                    name="Max HI data", 
                    mode='markers', 
                    marker=dict(color='navy', size=4, opacity=0.4),
                    legendgroup="cold_max_data", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Max HI (navy blue) - LOESS smoothed
            fig_cold.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=max_smoothed,
                    name="Max HI < 2p (LOESS)", 
                    mode='lines', 
                    line=dict(color='navy', width=2.5),
                    legendgroup="cold_max", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Max HI regression line
            fig_cold.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=regression_max,
                    name="Max HI trend", 
                    mode='lines', 
                    line=dict(color='navy', width=1.5, dash='dash'),
                    legendgroup="cold_max_reg", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Data points for Min HI (light blue, semi-transparent)
            fig_cold.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['min_extreme'],
                    name="Min HI data", 
                    mode='markers', 
                    marker=dict(color='skyblue', size=4, opacity=0.4),
                    legendgroup="cold_min_data", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Min HI (light blue) - LOESS smoothed
            fig_cold.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=min_smoothed,
                    name="Min HI < 2p (LOESS)", 
                    mode='lines', 
                    line=dict(color='skyblue', width=2.5),
                    legendgroup="cold_min", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Min HI regression line
            fig_cold.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=regression_min,
                    name="Min HI trend", 
                    mode='lines', 
                    line=dict(color='skyblue', width=1.5, dash='dash'),
                    legendgroup="cold_min_reg", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
        
        # Find max value across all stations for consistent y-axis
        max_y_cold = max([
            extreme_cold_df[extreme_cold_df['station'] == s]['max_extreme'].max()
            for s in selected_stations
        ] + [
            extreme_cold_df[extreme_cold_df['station'] == s]['min_extreme'].max()
            for s in selected_stations
        ])
        
        fig_cold.update_layout(
            title_text="Extreme Cold Days by Station (1974-2023)<br><sup>Days below date-specific 2nd percentile (November-February)</sup>",
            height=800,
            showlegend=True,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            )
        )
        
        fig_cold.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_cold.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            range=[0, 10],
            title_text="Count"
        )
        
        st.plotly_chart(fig_cold, use_container_width=True, key="extreme_cold_facet")
        
        # Trend analysis
        st.subheader("Extreme Cold Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Daytime Extreme Cold (Max HI):**")
            for station in selected_stations[:3]:
                station_data = extreme_cold_df[extreme_cold_df['station'] == station]
                early = station_data[station_data['year'] <= 1993]['max_extreme'].mean()
                recent = station_data[station_data['year'] >= 2010]['max_extreme'].mean()
                change = ((recent - early) / early * 100) if early > 0 else 0
                st.write(f"**{station_data['station_name'].iloc[0]}**: {change:+.1f}% change")
        
        with col2:
            st.markdown("**Nighttime Extreme Cold (Min HI):**")
            for station in selected_stations[3:]:
                station_data = extreme_cold_df[extreme_cold_df['station'] == station]
                early = station_data[station_data['year'] <= 1993]['min_extreme'].mean()
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
            summer = df[(df['month'].isin([3,4,5,6,7,8,9,10]))].copy()
            
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
            
            for year in range(1974, 2024):  # Changed to 2024 (exclusive) = 1974-2023
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
            y_title="Count"
        )
        
        for idx, station in enumerate(selected_stations):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            station_data = heatwave_df[heatwave_df['station'] == station].sort_values('year')
            
            # Apply LOESS smoothing (frac=0.2 matches R's f=1/5)
            max_smoothed = apply_loess_smoothing(station_data['year'].values, 
                                                  station_data['heatwave_max'].values, 
                                                  frac=0.2)
            min_smoothed = apply_loess_smoothing(station_data['year'].values, 
                                                  station_data['heatwave_min'].values, 
                                                  frac=0.2)
            
            # Calculate linear regression
            from scipy import stats as scipy_stats
            slope_max, intercept_max, _, _, _ = scipy_stats.linregress(station_data['year'].values, station_data['heatwave_max'].values)
            regression_max = slope_max * station_data['year'].values + intercept_max
            
            slope_min, intercept_min, _, _, _ = scipy_stats.linregress(station_data['year'].values, station_data['heatwave_min'].values)
            regression_min = slope_min * station_data['year'].values + intercept_min
            
            # Data points for HI-max (red, semi-transparent)
            # fig_heatwaves.add_trace(
            #     go.Scatter(
            #         x=station_data['year'], 
            #         y=station_data['heatwave_max'],
            #         name="HI-max data", 
            #         mode='markers', 
            #         marker=dict(color='red', size=4, opacity=0.4),
            #         legendgroup="heatwave_max_data", 
            #         showlegend=(idx==0)
            #     ),
            #     row=row, col=col
            # )
            
            # HI-max Heatwaves (red) - LOESS smoothed
            fig_heatwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=max_smoothed,
                    name="HI-max Heatwaves (LOESS)", 
                    mode='lines', 
                    line=dict(color='red', width=2.5),
                    legendgroup="heatwave_max", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # HI-max regression line
            fig_heatwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=regression_max,
                    name="HI-max trend", 
                    mode='lines', 
                    line=dict(color='red', width=1.5, dash='dash'),
                    legendgroup="heatwave_max_reg", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Data points for HI-min (orange, semi-transparent)
            fig_heatwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['heatwave_min'],
                    name="HI-min data", 
                    mode='markers', 
                    marker=dict(color='orange', size=4, opacity=0.4),
                    legendgroup="heatwave_min_data", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # HI-min Heatwaves (orange) - LOESS smoothed
            fig_heatwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=min_smoothed,
                    name="HI-min Heatwaves (LOESS)", 
                    mode='lines', 
                    line=dict(color='orange', width=2.5),
                    legendgroup="heatwave_min", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # HI-min regression line
            fig_heatwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=regression_min,
                    name="HI-min trend", 
                    mode='lines', 
                    line=dict(color='orange', width=1.5, dash='dash'),
                    legendgroup="heatwave_min_reg", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
        
        # Find max value across all stations for consistent y-axis
        max_y_heatwave = max([
            heatwave_df[heatwave_df['station'] == s]['heatwave_max'].max()
            for s in selected_stations
        ] + [
            heatwave_df[heatwave_df['station'] == s]['heatwave_min'].max()
            for s in selected_stations
        ])
        
        fig_heatwaves.update_layout(
            title_text="Heatwaves by Station (1974-2023)<br><sup>2+ consecutive days exceeding date-specific 98th percentile (March-October)</sup>",
            height=800,
            showlegend=True,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            )
        )
        
        fig_heatwaves.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_heatwaves.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            range=[0, 10],
            title_text="Count"
        )
        
        st.plotly_chart(fig_heatwaves, use_container_width=True, key="heatwaves_facet")
        
        # [Same summary statistics as before...]
    
        
        # Summary statistics
        st.subheader("Heatwave Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Daytime Heatwaves (HI-max) - Recent Increase:**")
            for station in selected_stations[:3]:
                station_data = heatwave_df[heatwave_df['station'] == station]
                early = station_data[station_data['year'] <= 1993]['heatwave_max'].sum()
                recent = station_data[station_data['year'] >= 2010]['heatwave_max'].sum()
                st.write(f"**{station_data['station_name'].iloc[0]}**: {early} days (1974-1993) â†’ {recent} days (2012-2024)")
        
        with col2:
            st.markdown("**Nighttime Heatwaves (HI-min) - Recent Increase:**")
            for station in selected_stations[3:]:
                station_data = heatwave_df[heatwave_df['station'] == station]
                early = station_data[station_data['year'] <= 1993]['heatwave_min'].sum()
                recent = station_data[station_data['year'] >= 2010]['heatwave_min'].sum()
                st.write(f"**{station_data['station_name'].iloc[0]}**: {early} days (1974-1993) â†’ {recent} days (2012-2024)")
        
        st.info("**Key Finding:** Nighttime heatwaves (orange lines) are particularly concerning as they prevent physiological recovery from daytime heat, compounding health risks.")
    
    else:  # Coldwaves
        st.subheader("Coldwave Analysis (November - February)")
        
        # Calculate coldwaves for all stations
        coldwave_data = []
        
        for station in selected_stations:
            df = dfs[station].copy()
            
            # Coldwaves (November-February)
            winter = df[(df['month'].isin([11,12,1,2]))].copy()
            
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
            
            for year in range(1974, 2024):  # Changed to 2024 (exclusive) = 1974-2023
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
            y_title="Count"
        )
        
        for idx, station in enumerate(selected_stations):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            station_data = coldwave_df[coldwave_df['station'] == station].sort_values('year')
            
            # Apply LOESS smoothing (frac=0.2 matches R's f=1/5)
            max_smoothed = apply_loess_smoothing(station_data['year'].values, 
                                                  station_data['coldwave_max'].values, 
                                                  frac=0.2)
            min_smoothed = apply_loess_smoothing(station_data['year'].values, 
                                                  station_data['coldwave_min'].values, 
                                                  frac=0.2)
            
            # Calculate linear regression
            from scipy import stats as scipy_stats
            slope_max, intercept_max, _, _, _ = scipy_stats.linregress(station_data['year'].values, station_data['coldwave_max'].values)
            regression_max = slope_max * station_data['year'].values + intercept_max
            
            slope_min, intercept_min, _, _, _ = scipy_stats.linregress(station_data['year'].values, station_data['coldwave_min'].values)
            regression_min = slope_min * station_data['year'].values + intercept_min
            
            # Data points for HI-max (dark blue, semi-transparent)
            fig_coldwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['coldwave_max'],
                    name="HI-max data", 
                    mode='markers', 
                    marker=dict(color='darkblue', size=4, opacity=0.4),
                    legendgroup="coldwave_max_data", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # HI-max Coldwaves (dark blue) - LOESS smoothed
            fig_coldwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=max_smoothed,
                    name="HI-max Coldwaves (LOESS)", 
                    mode='lines', 
                    line=dict(color='darkblue', width=2.5),
                    legendgroup="coldwave_max", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # HI-max regression line
            fig_coldwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=regression_max,
                    name="HI-max trend", 
                    mode='lines', 
                    line=dict(color='darkblue', width=1.5, dash='dash'),
                    legendgroup="coldwave_max_reg", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # Data points for HI-min (light blue, semi-transparent)
            fig_coldwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=station_data['coldwave_min'],
                    name="HI-min data", 
                    mode='markers', 
                    marker=dict(color='lightblue', size=4, opacity=0.4),
                    legendgroup="coldwave_min_data", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # HI-min Coldwaves (light blue) - LOESS smoothed
            fig_coldwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=min_smoothed,
                    name="HI-min Coldwaves (LOESS)", 
                    mode='lines', 
                    line=dict(color='lightblue', width=2.5),
                    legendgroup="coldwave_min", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
            
            # HI-min regression line
            fig_coldwaves.add_trace(
                go.Scatter(
                    x=station_data['year'], 
                    y=regression_min,
                    name="HI-min trend", 
                    mode='lines', 
                    line=dict(color='lightblue', width=1.5, dash='dash'),
                    legendgroup="coldwave_min_reg", 
                    showlegend=(idx==0)
                ),
                row=row, col=col
            )
        
        # Find max value across all stations for consistent y-axis
        max_y_coldwave = max([
            coldwave_df[coldwave_df['station'] == s]['coldwave_max'].max()
            for s in selected_stations
        ] + [
            coldwave_df[coldwave_df['station'] == s]['coldwave_min'].max()
            for s in selected_stations
        ])
        
        fig_coldwaves.update_layout(
            title_text="Coldwaves by Station (1974-2023)<br><sup>2+ consecutive days below date-specific 2nd percentile (November-February)</sup>",
            height=800,
            showlegend=True,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            )
        )
        
        fig_coldwaves.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_coldwaves.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            range=[0, 6],
            title_text="Count"
        )
        
        st.plotly_chart(fig_coldwaves, use_container_width=True, key="coldwaves_facet")
        
        # Summary statistics
        st.subheader("Coldwave Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Daytime Coldwaves (HI-max) - Trend:**")
            for station in selected_stations[:3]:
                station_data = coldwave_df[coldwave_df['station'] == station]
                early = station_data[station_data['year'] <= 1993]['coldwave_max'].sum()
                recent = station_data[station_data['year'] >= 2010]['coldwave_max'].sum()
                change = recent - early
                st.write(f"**{station_data['station_name'].iloc[0]}**: {early} days (1974-1993) --> {recent} days (2012-2024)")
        
        with col2:
            st.markdown("**Nighttime Coldwaves (HI-min) - Trend:**")
            for station in selected_stations[3:]:
                station_data = coldwave_df[coldwave_df['station'] == station]
                early = station_data[station_data['year'] <= 1993]['coldwave_min'].sum()
                recent = station_data[station_data['year'] >= 2010]['coldwave_min'].sum()
                change = recent - early
                st.write(f"**{station_data['station_name'].iloc[0]}**: {early} days (1974-1993) --> {recent} days (2012-2024)")
        
        st.info("**Key Finding:** Coldwaves are generally decreasing as winters warm, but extreme cold events still pose significant health risks when they occur.")

# TAB 4: WWA COMPARISON
with tab4:
    st.header("Comparison with NWS Watches, Warnings, and Advisories")
    
    st.markdown("""
    ### Methodology
    
    **Calculation Method:**
    1. For each station, we calculate the 98th percentile threshold for every calendar date using the full historical record (1974-2024)
    2. For the comparison period (2005-2022), we count how many days exceed their date-specific 98th percentile threshold
    3. WWAs are counted as issued by the NWS for each year
    4. We then compare our extreme heat day counts to the official WWA counts

    
    **Notes:**
    - **Our methodology uses:** Date-specific 98th percentile thresholds calculated across 51 years (1974-2024)
    - **NWS WWAs use:** Fixed heat index thresholds (typically 105-110°F depending on region) without date-specific adjustment
    - **Gaps in the data:** No WWAs were ever issued for Asheville during the analyzed period, likely due to cooler mountain climate. Also, WWAs in Wilmington are higher than heat events, but not sure why.
    
    ---
    
    **Visualization:**
    - **Extreme Heat Days** (Red): Days exceeding the 98th percentile threshold using our methodology
    - **WWAs Issued** (Black): Official National Weather Service heat warnings
    
    """)
    
    
    # ============================================================================
    # FILE PATH CONFIGURATION
    # ============================================================================
    # Try project directory first, fall back to user-specified path
    import os
    wwa_file_path = "/Users/shawnkairu/VSCODE/PlanetLab/NC Heat/"  # Project directory path
    if not os.path.exists(wwa_file_path):
        wwa_file_path = "NWS WWA/"  # Fallback path
    
    # ============================================================================
    
    # Load WWA data for all stations
    import os
    
    wwa_data = {}
    wwa_files = {
        'KAVL': f'{wwa_file_path}kavl.wwa.xlsx',
        'KCLT': f'{wwa_file_path}kclt.wwa.xlsx',
        'KGSO': f'{wwa_file_path}kgso.wwa.xlsx',
        'KHSE': f'{wwa_file_path}khse.wwa.xlsx',
        'KILM': f'{wwa_file_path}kilm.wwa.xlsx',
        'KRDU': f'{wwa_file_path}krdu.wwa.xlsx'
    }
    
    for station, filepath in wwa_files.items():
        try:
            if not os.path.exists(filepath):
                st.warning(f"File not found: {filepath}")
                wwa_data[station] = pd.DataFrame({'year': [], 'wwa_count': []})
                continue
            
            df = pd.read_excel(filepath)
            count_col = [col for col in df.columns if 'count' in col.lower() and 'wwa' in col.lower()][0]
            wwa_summary = df[['year', count_col]].dropna()
            wwa_summary.columns = ['year', 'wwa_count']
            wwa_summary['year'] = wwa_summary['year'].astype(int)
            wwa_data[station] = wwa_summary
        except Exception as e:
            st.error(f"Error loading WWA data for {station}: {e}")
            wwa_data[station] = pd.DataFrame({'year': [], 'wwa_count': []})
    
    if all(df.empty for df in wwa_data.values()):
        st.error(f"""
        ### No WWA Data Loaded
        Please update `wwa_file_path = "{wwa_file_path}"` to point to your WWA files.
        """)
        st.stop()
    
    # Calculate extreme heat days for 2005-2022
    extreme_heat_comparison = {}
    
    for station in selected_stations:
        df = dfs[station].copy()
        
        # Ensure numeric conversion
        df['heatindexmax2m'] = pd.to_numeric(df['heatindexmax2m'], errors='coerce')
        
        # Create month_day for date-specific percentiles
        df['month_day'] = df['datetime'].dt.strftime('%m-%d')
        
        # Filter for March-October across ALL years for percentile calculation
        df_march_oct_all = df[df['month'].isin([3,4,5,6,7,8,9,10])].copy()
        
        # Calculate 98th percentile for each calendar date using all years (1974-2024)
        date_thresholds = df_march_oct_all.groupby('month_day')['heatindexmax2m'].quantile(0.98).reset_index()
        date_thresholds.columns = ['month_day', 'threshold_0.98']
        
        # Now filter for 2005-2022 only
        filtered = df_march_oct_all[(df_march_oct_all['year'] >= 2005) & 
                                     (df_march_oct_all['year'] <= 2022)].copy()
        
        # Merge thresholds
        filtered = filtered.merge(date_thresholds, on='month_day', how='left')
        
        # Flag extreme events
        filtered['extreme_event'] = filtered['heatindexmax2m'] > filtered['threshold_0.98']
        
        # Count extreme heat days per year
        yearly_extreme = filtered[filtered['extreme_event']].groupby('year').size().reset_index()
        yearly_extreme.columns = ['year', 'extreme_heat_days']
        
        # Ensure all years 2005-2022 are present
        all_years = pd.DataFrame({'year': range(2005, 2023)})
        yearly_extreme = all_years.merge(yearly_extreme, on='year', how='left').fillna(0)
        yearly_extreme['extreme_heat_days'] = yearly_extreme['extreme_heat_days'].astype(int)
        
        extreme_heat_comparison[station] = yearly_extreme
    
    # Create visualization matching the paper figure style
    st.markdown("### Number of Extreme Heat Days and WWAs Issued")
    st.markdown("**March - October (2005-2022)**")
    
    stations_to_plot = [s for s in selected_stations if not wwa_data[s].empty or s in extreme_heat_comparison]
    
    if len(stations_to_plot) == 0:
        st.warning("No data available for comparison")
    else:
        # Create subplot layout matching the paper (3x2 for up to 6 stations)
        n_stations = len(stations_to_plot)
        n_cols = 3
        n_rows = (n_stations + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=[
                dfs[s]['station_name'].iloc[0].split('-')[0].strip() + 
                (" (Piedmont)" if "Raleigh" in dfs[s]['station_name'].iloc[0] else "")
                for s in stations_to_plot
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.10
        )
        
        # Plot each station with vertical line style (matching the figure)
        for idx, station in enumerate(stations_to_plot):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            # Get data
            extreme_data = extreme_heat_comparison.get(station, pd.DataFrame())
            wwa_df = wwa_data.get(station, pd.DataFrame())
            
            # Merge to get both values for each year
            if not extreme_data.empty and not wwa_df.empty:
                combined = extreme_data.merge(wwa_df, on='year', how='outer').fillna(0)
            elif not extreme_data.empty:
                combined = extreme_data.copy()
                combined['wwa_count'] = 0
            elif not wwa_df.empty:
                combined = wwa_df.copy()
                combined['extreme_heat_days'] = 0
            else:
                continue
            
            combined = combined.sort_values('year')
            
            # For each year, draw a vertical line from WWA to Extreme Heat Day
            for _, row_data in combined.iterrows():
                year = row_data['year']
                extreme_val = row_data.get('extreme_heat_days', 0)
                wwa_val = row_data.get('wwa_count', 0)
                
                # Draw vertical line connecting the two points
                fig.add_trace(
                    go.Scatter(
                        x=[year, year],
                        y=[wwa_val, extreme_val],
                        mode='lines',
                        line=dict(color='black', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
            
            # Plot Extreme Heat Days (Red dots on top)
            if not extreme_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=extreme_data['year'],
                        y=extreme_data['extreme_heat_days'],
                        mode='markers',
                        name='Extreme Heat Days',
                        marker=dict(color='red', size=8, symbol='circle'),
                        showlegend=(idx == 0),
                        legendgroup='extreme',
                        hovertemplate='<b>Year:</b> %{x}<br><b>Extreme Heat Days:</b> %{y}<extra></extra>'
                    ),
                    row=row, col=col
                )
            
            # Plot WWAs Issued (Black dots on bottom)
            if not wwa_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=wwa_df['year'],
                        y=wwa_df['wwa_count'],
                        mode='markers',
                        name='WWAs Issued',
                        marker=dict(color='black', size=8, symbol='circle'),
                        showlegend=(idx == 0),
                        legendgroup='wwa',
                        hovertemplate='<b>Year:</b> %{x}<br><b>WWAs Issued:</b> %{y}<extra></extra>'
                    ),
                    row=row, col=col
                )
            
            # Update axes
            fig.update_xaxes(
                title_text="Year" if row == n_rows else "",
                range=[2004.5, 2021.5],
                dtick=2,
                row=row, col=col
            )
            
            y_max = max(
                extreme_data['extreme_heat_days'].max() if not extreme_data.empty else 0,
                wwa_df['wwa_count'].max() if not wwa_df.empty else 0
            )
            
            fig.update_yaxes(
                title_text="Count" if col == 1 else "",
                range=[0, y_max + 5],
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            height=350 * n_rows,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=16)
            ),
            template="plotly_white",
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("### Summary Statistics")
        
        summary_data = []
        for station in stations_to_plot:
            station_name = dfs[station]['station_name'].iloc[0]
            extreme_data = extreme_heat_comparison.get(station, pd.DataFrame())
            wwa_df = wwa_data.get(station, pd.DataFrame())
            
            total_extreme = int(extreme_data['extreme_heat_days'].sum()) if not extreme_data.empty else 0
            total_wwa = int(wwa_df['wwa_count'].sum()) if not wwa_df.empty else 0
            avg_extreme = extreme_data['extreme_heat_days'].mean() if not extreme_data.empty else 0
            avg_wwa = wwa_df['wwa_count'].mean() if not wwa_df.empty else 0
            difference = total_extreme - total_wwa
            
            summary_data.append({
                'Station': station_name,
                'Total Extreme Heat Days': total_extreme,
                'Total WWAs Issued': total_wwa,
                'Difference': difference,
                'Avg per Year (Extreme)': f"{avg_extreme:.1f}",
                'Avg per Year (WWA)': f"{avg_wwa:.1f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Key findings
        st.markdown("### Key Findings")
        
        total_all_extreme = sum([extreme_heat_comparison[s]['extreme_heat_days'].sum() 
                                for s in stations_to_plot if s in extreme_heat_comparison])
        total_all_wwa = sum([wwa_data[s]['wwa_count'].sum() 
                            for s in stations_to_plot if not wwa_data[s].empty])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Extreme Heat Days",
                f"{int(total_all_extreme)}",
                help="Sum across all stations, 2005-2022"
            )
        
        with col2:
            st.metric(
                "Total WWAs Issued",
                f"{int(total_all_wwa)}",
                help="Sum across all stations, 2005-2022"
            )
        
        with col3:
            if total_all_wwa > 0:
                difference_pct = ((total_all_extreme - total_all_wwa) / total_all_wwa * 100)
                delta_text = f"{difference_pct:+.1f}%"
            else:
                delta_text = "N/A"
            
            st.metric(
                "Difference",
                f"{int(total_all_extreme - total_all_wwa):+d}",
                delta=delta_text,
                help="Extreme heat days minus WWAs issued"
            )
        
        # Interpretation
        if total_all_extreme > total_all_wwa:
            st.markdown(f"""
            - Our methodology identifies {int(total_all_extreme - total_all_wwa)} more events ({abs(difference_pct):.1f}% more) than current WWA thresholds
            - The gap represents days with elevated health risk that don't trigger official warnings
            - This suggests current thresholds may be too conservative for public health protection
            """)
        elif total_all_extreme < total_all_wwa:
            st.markdown(f"""
            **Interpretation:**
            - WWAs were issued **{int(total_all_wwa - total_all_extreme)} more times** than our methodology identified extreme heat days
            - This may indicate WWAs are issued for shorter durations or lower thresholds
            - Note: WWAs can be issued for heat events lasting just a few hours, while our methodology requires a full day above the 98th percentile
            """)
        else:
            st.markdown("**Interpretation:** The methodologies identified a similar number of events overall.")
        
# TAB 5: REGIONAL HEATMAP (ERA5)
with tab5:
    st.header("Regional Temperature Analysis")

    
    # Check if ERA5 file exists
    era5_file = 'era5_temperature_nc_1974_2024.nc'
    
    if not os.path.exists(era5_file):
        st.warning(f"""
        ### ERA5 Data Not Found
        
        To use this feature, please download ERA5 monthly temperature data:
        
        1. **Create account:** https://cds.climate.copernicus.eu/
        2. **Download dataset:** Monthly 2m temperature (1974-2024)
        3. **Region:** North America (lat: 24-40°N, lon: -92 to -75°E)
        4. **Save as:** `{era5_file}` in the same directory as app.py
        
        **Approximate file size:** 200-500 MB
        """)
        
        with st.expander("View Download Code"):
            st.code('''
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': '2m_temperature',
        'year': [str(year) for year in range(1974, 2025)],
        'month': [f'{m:02d}' for m in range(1, 13)],
        'time': '00:00',
        'area': [40, -92, 24, -75],  # N, W, S, E
        'format': 'netcdf',
    },
    'era5_temperature_nc_1974_2024.nc'
)
            ''', language='python')
        
        st.stop()
    
    # Load ERA5 data
    @st.cache_data
    def load_era5_cached(file_path):
        return process_era5_data(file_path)
    
    with st.spinner('Loading ERA5 data...'):
        temp_data, ds = load_era5_cached(era5_file)
    
    if temp_data is None:
        st.error("Failed to load ERA5 data. Please check the file format.")
        st.stop()
    
    # Heatmap type selector
    st.markdown("---")
    map_type = st.radio(
        "**Select Heatmap Type:**",
        [
            "Temperature Change (1974-2024 Trend)",
            "Temperature Distribution (Specific Year)",
            "Compare Two Years"
        ],
        key="heatmap_type"
    )
    
    # Station coordinates for overlay
    station_coords = {
        'KAVL': (35.4363, -82.5415, 'Asheville'),
        'KGSO': (36.0975, -79.9373, 'Greensboro'),
        'KHSE': (35.2677, -75.5458, 'Cape Hatteras'),
        'KILM': (34.2704, -77.9025, 'Wilmington'),
        'KCLT': (35.2144, -80.9473, 'Charlotte'),
        'KRDU': (35.8801, -78.7880, 'Raleigh-Durham'),
    }
    
    # =================================================================
    # TYPE 1: TEMPERATURE CHANGE (TREND) - PRIMARY ANALYSIS
    # =================================================================
    if map_type == "Temperature Change (1974-2024 Trend)":
        st.markdown("""
        ### Rate of Temperature Change (1974-2024)
        
        **Shows:** How fast each location is warming or cooling (degrees Fahrenheit per decade)  
        """)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Season selector
            season_choice = st.selectbox(
                "Select Season:",
                [
                    "Summer (Jun-Aug)",
                    "Winter (Dec-Feb)",
                    "Spring (Mar-May)",
                    "Fall (Sep-Nov)",
                    "Annual Average"
                ],
                key="trend_season"
            )
            
            # Map season to months
            season_months = {
                "Summer (Jun-Aug)": [6, 7, 8],
                "Winter (Dec-Feb)": [12, 1, 2],
                "Spring (Mar-May)": [3, 4, 5],
                "Fall (Sep-Nov)": [9, 10, 11],
                "Annual Average": list(range(1, 13))
            }
            
            months = season_months[season_choice]
        
        # Calculate trends
        with st.spinner('Calculating temperature trends for each grid cell...'):
            lats, lons, slopes, r_squared, p_values = calculate_temperature_trends(
                temp_data, months
            )
        
        # Always apply high-quality smoothing
        st.markdown("---")
        with st.spinner('Creating publication-quality map...'):
            lats_plot, lons_plot, slopes_plot = interpolate_grid(lats, lons, slopes, factor=8)
        
        # Create meshgrid for matplotlib
        lons_mesh, lats_mesh = np.meshgrid(lons_plot, lats_plot)
        
        # CREATE MATPLOTLIB FIGURE
        fig = create_matplotlib_heatmap_inline(
            lats_mesh, lons_mesh, slopes_plot,
            station_coords=station_coords,
            title=f'Temperature Change: {season_choice} (1974-2024)\nRate of warming/cooling across Southeast US',
            cbar_label='Temperature Change (°C/decade)',
            cmap='RdYlBu_r',
            vmin=-0.5,
            vmax=1.5,
            diverging=True,
            dpi=150
        )
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        # Statistics
        st.markdown("### Trend Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mean_change = np.nanmean(slopes)
            st.metric("Regional Average Change", f"{mean_change:.3f}°C/decade")
        
        with col2:
            max_change = np.nanmax(slopes)
            st.metric("Maximum Warming", f"{max_change:.3f}°C/decade")
        

        
    
    # =================================================================
    # TYPE 2: TEMPERATURE DISTRIBUTION (SPECIFIC YEAR/SEASON)
    # =================================================================
    elif map_type == "Temperature Distribution (Specific Year)":
        st.markdown("""
        ### Spatial Temperature Patterns for a Specific Year
        
        **Shows:** Actual temperature distribution across the region for any year and season  
        **Purpose:** Explore how spatial patterns vary from year to year
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_year = st.selectbox(
                "Select Year:",
                list(range(1974, 2024)),
                index=49,  # Default to 2021
                key="snapshot_year"
            )
        
        with col2:
            season_choice = st.selectbox(
                "Select Season:",
                [
                    "Summer (Jun-Aug)",
                    "Winter (Dec-Feb)",
                    "Spring (Mar-May)",
                    "Fall (Sep-Nov)"
                ],
                key="snapshot_season"
            )
            
            season_months = {
                "Summer (Jun-Aug)": [6, 7, 8],
                "Winter (Dec-Feb)": [12, 1, 2],
                "Spring (Mar-May)": [3, 4, 5],
                "Fall (Sep-Nov)": [9, 10, 11]
            }
            
            months = season_months[season_choice]
        
        # Get temperature snapshot
        with st.spinner(f'Loading {season_choice} {selected_year} data...'):
            lats, lons, temps = get_temperature_snapshot(temp_data, selected_year, months)
        
        if temps is None:
            st.error(f"No data available for {season_choice} {selected_year}")
            st.stop()
        
        # Always apply high-quality smoothing
        st.markdown("---")
        with st.spinner('Creating publication-quality map...'):
            lats_plot, lons_plot, temps_plot = interpolate_grid(lats, lons, temps, factor=8)
        
        # Create meshgrid for matplotlib
        lons_mesh, lats_mesh = np.meshgrid(lons_plot, lats_plot)
        
        # CREATE MATPLOTLIB FIGURE
        fig = create_matplotlib_heatmap_inline(
            lats_mesh, lons_mesh, temps_plot,
            station_coords=station_coords,
            title=f'Temperature Distribution: {season_choice} {selected_year}',
            cbar_label='Temperature (°C)',
            cmap='coolwarm',
            diverging=False,
            dpi=150
        )
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Regional Average", f"{np.nanmean(temps):.1f}°C")
        with col2:
            st.metric("Warmest Location", f"{np.nanmax(temps):.1f}°C")
        with col3:
            st.metric("Coolest Location", f"{np.nanmin(temps):.1f}°C")
    
    # =================================================================
    # TYPE 3: COMPARE TWO YEARS
    # =================================================================
    else:  # Compare Two Years
        st.markdown("""
        ### Compare Temperature Patterns Between Two Years
        
        **Shows:** The difference between two years (Year 2 minus Year 1)  
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year1 = st.selectbox(
                "First Year:", 
                list(range(1974, 2024)), 
                index=49, 
                key="compare_year1"
            )
        
        with col2:
            year2 = st.selectbox(
                "Second Year:", 
                list(range(1974, 2024)), 
                index=49, 
                key="compare_year2"
            )
        
        with col3:
            season_choice = st.selectbox(
                "Season:",
                ["Summer (Jun-Aug)", "Winter (Dec-Feb)", "Spring (Mar-May)", "Fall (Sep-Nov)"],
                key="compare_season"
            )
            
            season_map = {
                "Summer (Jun-Aug)": [6,7,8],
                "Winter (Dec-Feb)": [12,1,2],
                "Spring (Mar-May)": [3,4,5],
                "Fall (Sep-Nov)": [9,10,11]
            }
            months = season_map[season_choice]
        
        if year1 == year2:
            st.warning("Please select two different years to compare.")
            st.stop()
        
        # Get both snapshots
        with st.spinner(''):
            lats1, lons1, temps1 = get_temperature_snapshot(temp_data, year1, months)
            lats2, lons2, temps2 = get_temperature_snapshot(temp_data, year2, months)
            
            if temps1 is None or temps2 is None:
                st.error("Data not available for one or both selected years/seasons")
                st.stop()
            
            # Calculate difference
            temp_diff = temps2 - temps1
        
        # Always apply high-quality smoothing
        st.markdown("---")
        with st.spinner(''):
            lats_plot, lons_plot, diff_plot = interpolate_grid(lats1, lons1, temp_diff, factor=8)
        
        # Create meshgrid for matplotlib
        lons_mesh, lats_mesh = np.meshgrid(lons_plot, lats_plot)
        
        # CREATE MATPLOTLIB FIGURE
        fig = create_matplotlib_heatmap_inline(
            lats_mesh, lons_mesh, diff_plot,
            station_coords=station_coords,  # Show stations
            title=f'Temperature Difference: {season_choice}\n{year2} minus {year1}',
            cbar_label='Temperature Difference (°C)',
            cmap='RdBu_r',
            diverging=True,
            dpi=150
        )
        
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        mean_diff = np.nanmean(temp_diff)
        
        with col1:
            st.metric(
                "Average Difference", 
                f"{mean_diff:.2f}°C",
                delta=f"{year2} vs {year1}"
            )
        
        with col2:
            st.metric("Maximum Warming", f"+{np.nanmax(temp_diff):.1f}°C")
        
        with col3:
            st.metric("Maximum Cooling", f"{np.nanmin(temp_diff):.1f}°C")
        
        with col4:
            pct_warmer = np.sum(temp_diff > 0) / np.sum(~np.isnan(temp_diff)) * 100
            st.metric("% Area Warmer", f"{pct_warmer:.1f}%")
        
        if mean_diff > 0:
            st.success(f"**Overall:** {year2} was {abs(mean_diff):.2f}°F warmer than {year1} on average")
        else:
            st.info(f"**Overall:** {year2} was {abs(mean_diff):.2f}°F cooler than {year1} on average")

with tab6:
    st.header("Additional Regional Heat Maps")
    
    st.markdown("""
    Explore temperature patterns and trends across different regions of the world.
    Select a region below to view its heat map analysis.
    """)
    
    # Region selector
    region_key = st.selectbox(
        "Select Region:",
        list(ADDITIONAL_REGIONS.keys()),
        format_func=lambda x: ADDITIONAL_REGIONS[x]['full_name']
    )
    
    region = ADDITIONAL_REGIONS[region_key]
    
    st.markdown(f"### {region['full_name']}")
    
    # Check if data file exists
    era5_file_region = region['file']
    
    if not os.path.exists(era5_file_region):
        st.warning(f"""
        ### ERA5 Data Not Found for {region['name']}
        
        To use this feature, please download ERA5 monthly temperature data:
        
        1. **Run the download script:** `python download_era5_additional_regions.py`
        2. **Or download manually from:** https://cds.climate.copernicus.eu/
        3. **Region coordinates:** {region['area']} [North, West, South, East]
        4. **Save as:** `{era5_file_region}` in the same directory as app.py
        """)
        
        with st.expander("View Download Code"):
            st.code(f"""
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {{
        'product_type': 'monthly_averaged_reanalysis',
        'variable': '2m_temperature',
        'year': [str(year) for year in range(1974, 2025)],
        'month': [f'{{m:02d}}' for m in range(1, 13)],
        'time': '00:00',
        'area': {region['area']},  # [N, W, S, E]
        'format': 'netcdf',
    }},
    '{era5_file_region}'
)
            """, language='python')
        
        st.stop()
    
    # Load ERA5 data for region
    @st.cache_data
    def load_region_data_cached(file_path):
        return process_era5_data_for_region(file_path)
    
    with st.spinner(f'Loading {region["name"]} data...'):
        temp_data_region, ds_region = load_region_data_cached(era5_file_region)
    
    if temp_data_region is None:
        st.error("Failed to load ERA5 data. Please check the file format.")
        st.stop()
    
    # City coordinates for this region
    city_coords = region['cities']
    
    # Heatmap type selector
    st.markdown("---")
    map_type_region = st.radio(
        "**Select Heatmap Type:**",
        [
            "Temperature Change (1974-2024 Trend)",
            "Temperature Distribution (Specific Year)",
            "Compare Two Years"
        ],
        key=f"heatmap_type_{region_key}"
    )
    
    # GeoJSON URL for country borders
    geojson_url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    
    # Target countries for each region
    target_countries_map = {
        'east_africa': ['Uganda', 'Kenya', 'Tanzania', 'Rwanda', 'Burundi', 'South Sudan', 'Ethiopia', 'Somalia'],
        'madagascar': ['Madagascar'],
        'brazil': ['Brazil', 'Argentina', 'Paraguay', 'Uruguay', 'Bolivia', 'Peru', 'Colombia', 'Venezuela', 'Guyana', 'Suriname', 'French Guiana']
    }
    target_countries = target_countries_map.get(region_key)
    
    # =================================================================
    # TYPE 1: TEMPERATURE CHANGE (TREND)
    # =================================================================
    if map_type_region == "Temperature Change (1974-2024 Trend)":
        st.markdown("""
        ### Rate of Temperature Change (1974-2024)
        
        **Shows:** How fast each location is warming or cooling (degrees Celsius per decade)  
        """)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            season_choice_region = st.selectbox(
                "Select Season:",
                list(region['seasons'].keys()),
                key=f"trend_season_{region_key}"
            )
            months_region = region['seasons'][season_choice_region]
        
        with st.spinner('Calculating temperature trends...'):
            lats_r, lons_r, slopes_r, r_squared_r, p_values_r = calculate_temperature_trends_region(
                temp_data_region, months_region
            )
        
        if slopes_r is None:
            st.error("Failed to calculate trends")
            st.stop()
        
        st.markdown("---")
        with st.spinner('Creating map...'):
            lats_plot_r, lons_plot_r, slopes_plot_r = interpolate_grid(lats_r, lons_r, slopes_r, factor=8)
        
        lons_mesh_r, lats_mesh_r = np.meshgrid(lons_plot_r, lats_plot_r)
        
        fig_region = create_heatmap_with_borders(
            lats_mesh_r, lons_mesh_r, slopes_plot_r,
            station_coords=city_coords,
            title=f'Temperature Change: {season_choice_region} (1974-2024)\\n{region["full_name"]}',
            cbar_label='Temperature Change (°C/decade)',
            cmap='RdYlBu_r',
            vmin=-0.3,
            vmax=0.8,
            diverging=True,
            dpi=150,
            geojson_url=geojson_url,
            target_countries=target_countries
        )
        
        st.pyplot(fig_region, use_container_width=True)
        plt.close(fig_region)
        
        # Statistics
        st.markdown("### Trend Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mean_change_r = np.nanmean(slopes_r)
            st.metric("Regional Average Change", f"{mean_change_r:.3f}°C/decade")
        
        with col2:
            max_change_r = np.nanmax(slopes_r)
            st.metric("Maximum Warming", f"{max_change_r:.3f}°C/decade")
        
    
    # =================================================================
    # TYPE 2: TEMPERATURE DISTRIBUTION (SPECIFIC YEAR)
    # =================================================================
    elif map_type_region == "Temperature Distribution (Specific Year)":
        st.markdown("""
        ### Spatial Temperature Patterns for a Specific Year
        
        **Shows:** Actual temperature distribution across the region  
        **Purpose:** Explore how spatial patterns vary from year to year
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_year_region = st.selectbox(
                "Select Year:",
                list(range(1974, 2024)),
                index=49,
                key=f"snapshot_year_{region_key}"
            )
        
        with col2:
            season_choice_region = st.selectbox(
                "Select Season:",
                list(region['seasons'].keys()),
                key=f"snapshot_season_{region_key}"
            )
            months_region = region['seasons'][season_choice_region]
        
        with st.spinner(f'Loading {season_choice_region} {selected_year_region} data...'):
            lats_r, lons_r, temps_r = get_temperature_snapshot_region(temp_data_region, selected_year_region, months_region)
        
        if temps_r is None:
            st.error(f"No data available for {season_choice_region} {selected_year_region}")
            st.stop()
        
        st.markdown("---")
        with st.spinner('Creating map...'):
            lats_plot_r, lons_plot_r, temps_plot_r = interpolate_grid(lats_r, lons_r, temps_r, factor=8)
        
        lons_mesh_r, lats_mesh_r = np.meshgrid(lons_plot_r, lats_plot_r)
        
        # Use 'coolwarm' - blue for cool, red for warm (intuitive!)
        fig_region = create_heatmap_with_borders(
            lats_mesh_r, lons_mesh_r, temps_plot_r,
            station_coords=city_coords,
            title=f'Temperature Distribution: {season_choice_region} {selected_year_region}\\n{region["full_name"]}',
            cbar_label='Temperature (°C)',
            cmap='coolwarm',
            diverging=False,
            dpi=150,
            geojson_url=geojson_url,
            target_countries=target_countries
        )
        
        st.pyplot(fig_region, use_container_width=True)
        plt.close(fig_region)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Regional Average", f"{np.nanmean(temps_r):.1f}°C")
        with col2:
            st.metric("Warmest Location", f"{np.nanmax(temps_r):.1f}°C")
        with col3:
            st.metric("Coolest Location", f"{np.nanmin(temps_r):.1f}°C")
    
    # =================================================================
    # TYPE 3: COMPARE TWO YEARS
    # =================================================================
    else:  # Compare Two Years
        st.markdown("""
        ### Compare Temperature Patterns Between Two Years
        
        **Shows:** The difference between two years (Year 2 minus Year 1)  
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year1_region = st.selectbox(
                "First Year:", 
                list(range(1974, 2024)), 
                index=0, 
                key=f"compare_year1_{region_key}"
            )
        
        with col2:
            year2_region = st.selectbox(
                "Second Year:", 
                list(range(1974, 2024)), 
                index=49, 
                key=f"compare_year2_{region_key}"
            )
        
        with col3:
            season_choice_region = st.selectbox(
                "Season:",
                list(region['seasons'].keys()),
                key=f"compare_season_{region_key}"
            )
            months_region = region['seasons'][season_choice_region]
        
        if year1_region == year2_region:
            st.warning("Please select two different years to compare.")
            st.stop()
        
        with st.spinner('Loading data...'):
            lats1_r, lons1_r, temps1_r = get_temperature_snapshot_region(temp_data_region, year1_region, months_region)
            lats2_r, lons2_r, temps2_r = get_temperature_snapshot_region(temp_data_region, year2_region, months_region)
            
            if temps1_r is None or temps2_r is None:
                st.error("Data not available for one or both selected years/seasons")
                st.stop()
            
            temp_diff_r = temps2_r - temps1_r
        
        st.markdown("---")
        with st.spinner('Creating map...'):
            lats_plot_r, lons_plot_r, diff_plot_r = interpolate_grid(lats1_r, lons1_r, temp_diff_r, factor=8)
        
        lons_mesh_r, lats_mesh_r = np.meshgrid(lons_plot_r, lats_plot_r)
        
        fig_region = create_heatmap_with_borders(
            lats_mesh_r, lons_mesh_r, diff_plot_r,
            station_coords=city_coords,
            title=f'Temperature Difference: {season_choice_region}\\n{year2_region} minus {year1_region} | {region["full_name"]}',
            cbar_label='Temperature Difference (°C)',
            cmap='RdBu_r',
            diverging=True,
            dpi=150,
            geojson_url=geojson_url,
            target_countries=target_countries
        )
        
        st.pyplot(fig_region, use_container_width=True)
        plt.close(fig_region)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        mean_diff_r = np.nanmean(temp_diff_r)
        
        with col1:
            st.metric(
                "Average Difference", 
                f"{mean_diff_r:.2f}°C",
                delta=f"{year2_region} vs {year1_region}"
            )
        
        with col2:
            st.metric("Maximum Warming", f"+{np.nanmax(temp_diff_r):.1f}°C")
        
        with col3:
            st.metric("Maximum Cooling", f"{np.nanmin(temp_diff_r):.1f}°C")
        
        with col4:
            pct_warmer_r = np.sum(temp_diff_r > 0) / np.sum(~np.isnan(temp_diff_r)) * 100
            st.metric("% Area Warmer", f"{pct_warmer_r:.1f}%")
        
        if mean_diff_r > 0:
            st.success(f"**Overall:** {year2_region} was {abs(mean_diff_r):.2f}°C warmer than {year1_region} on average")
        else:
            st.info(f"**Overall:** {year2_region} was {abs(mean_diff_r):.2f}°C cooler than {year1_region} on average")

# Footer
st.markdown("---")
st.markdown("""
""")
