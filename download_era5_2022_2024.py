"""
Download ERA5 Temperature and Dewpoint Data for 2022-2024
For calculating heat index in North Carolina

Requirements:
1. CDS API account: https://cds.climate.copernicus.eu/
2. Install cdsapi: pip install cdsapi
3. Setup ~/.cdsapirc with your credentials

Variables needed:
- 2m_temperature (t2m)
- 2m_dewpoint_temperature (d2m)
"""

import cdsapi
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# North Carolina bounding box
# Format: [North, West, South, East]
NC_AREA = [37, -85, 33, -75]  # Covers all NC stations

# Years to download
YEARS = [2022, 2023, 2024]

# Variables needed for heat index calculation
VARIABLES = [
    '2m_temperature',           # t2m
    '2m_dewpoint_temperature'   # d2m
]

# Station coordinates for reference
STATIONS = {
    'KAVL': {'lat': 35.4361, 'lon': -82.5375, 'name': 'Asheville'},
    'KRDU': {'lat': 35.8776, 'lon': -78.7875, 'name': 'Raleigh-Durham'},
    'KCLT': {'lat': 35.2140, 'lon': -80.9431, 'name': 'Charlotte'},
    'KGSO': {'lat': 36.0978, 'lon': -79.9373, 'name': 'Greensboro'},
    'KILM': {'lat': 34.2704, 'lon': -77.9026, 'name': 'Wilmington'},
    'KHSE': {'lat': 35.2327, 'lon': -75.6178, 'name': 'Cape Hatteras'}
}

# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_era5_year(year, variables, area, output_dir='era5_downloads'):
    """
    Download ERA5 data for a single year
    
    Args:
        year: Year to download (e.g., 2022)
        variables: List of variable names
        area: Bounding box [North, West, South, East]
        output_dir: Directory to save files
    
    Returns:
        Path to downloaded file
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    output_file = f"{output_dir}/era5_{year}_t2m_d2m_nc.nc"
    
    print(f"\n{'='*80}")
    print(f"Downloading ERA5 data for {year}")
    print(f"Variables: {', '.join(variables)}")
    print(f"Area: North Carolina ({area})")
    print('='*80)
    
    c = cdsapi.Client()
    
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variables,
                'year': str(year),
                'month': [
                    '01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12',
                ],
                'day': [
                    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
                ],
                'area': area,  # [North, West, South, East]
            },
            output_file
        )
        
        print(f"\n✓ Successfully downloaded: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"\n✗ Error downloading {year}: {e}")
        return None


def download_all_years(years=YEARS, variables=VARIABLES, area=NC_AREA):
    """
    Download ERA5 data for multiple years
    
    Args:
        years: List of years to download
        variables: List of variable names
        area: Bounding box
    
    Returns:
        List of downloaded file paths
    """
    downloaded_files = []
    
    print("="*80)
    print("ERA5 DATA DOWNLOAD - 2022-2024")
    print("="*80)
    print(f"\nYears: {years}")
    print(f"Variables: {variables}")
    print(f"Area: {area} (North Carolina)")
    print("\nThis will download ~3 files (one per year)")
    print("Each file is approximately 500MB-1GB")
    print("Total download time: 10-30 minutes (depends on CDS queue)")
    
    for year in years:
        file_path = download_era5_year(year, variables, area)
        if file_path:
            downloaded_files.append(file_path)
        
        # Check file size
        if file_path and Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            print(f"  File size: {size_mb:.1f} MB")
    
    print("\n" + "="*80)
    print(f"DOWNLOAD COMPLETE: {len(downloaded_files)}/{len(years)} files")
    print("="*80)
    
    return downloaded_files


# ============================================================================
# HEAT INDEX CALCULATION
# ============================================================================

def calculate_relative_humidity_era5(t2m_k, d2m_k):
    """
    Calculate relative humidity from temperature and dewpoint (ERA5 format)
    
    Args:
        t2m_k: Temperature in Kelvin
        d2m_k: Dewpoint temperature in Kelvin
    
    Returns:
        Relative humidity as percentage (0-100)
    """
    # Convert to Celsius
    t2m_c = t2m_k - 273.15
    d2m_c = d2m_k - 273.15
    
    # Calculate saturation vapor pressure using Magnus formula
    def vapor_pressure(t):
        return 6.112 * np.exp((17.67 * t) / (t + 243.5))
    
    es = vapor_pressure(t2m_c)  # Saturation vapor pressure
    e = vapor_pressure(d2m_c)   # Actual vapor pressure
    
    rh = (e / es) * 100
    
    return np.clip(rh, 0, 100)


def calculate_heat_index_era5(t2m_k, rh):
    """
    Calculate heat index from temperature (Kelvin) and relative humidity
    
    Args:
        t2m_k: Temperature in Kelvin
        rh: Relative humidity (0-100)
    
    Returns:
        Heat index in Kelvin (to match ERA5 units)
    """
    # Convert to Fahrenheit for calculation
    t_f = (t2m_k - 273.15) * 9/5 + 32
    
    # For temperatures below 80°F, heat index equals temperature
    if isinstance(t_f, np.ndarray):
        hi_f = np.where(t_f < 80, t_f, np.nan)
        
        # Apply full formula where temp >= 80°F
        mask = t_f >= 80
        if mask.any():
            T = t_f[mask]
            R = rh[mask] if isinstance(rh, np.ndarray) else rh
            
            # Simple estimate
            HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (R * 0.094))
            
            # Full Rothfusz regression for high heat index
            needs_full = (HI + T) / 2 >= 80
            if needs_full.any():
                T_full = T[needs_full]
                R_full = R[needs_full] if isinstance(R, np.ndarray) else R
                
                HI_full = (-42.379 + 
                          2.04901523 * T_full + 
                          10.14333127 * R_full - 
                          0.22475541 * T_full * R_full - 
                          6.83783e-3 * T_full**2 - 
                          5.481717e-2 * R_full**2 + 
                          1.22874e-3 * T_full**2 * R_full + 
                          8.5282e-4 * T_full * R_full**2 - 
                          1.99e-6 * T_full**2 * R_full**2)
                
                # Adjustments
                low_rh_mask = (R_full < 13) & (T_full >= 80) & (T_full <= 112)
                if low_rh_mask.any():
                    adjustment = ((13 - R_full[low_rh_mask]) / 4) * np.sqrt((17 - abs(T_full[low_rh_mask] - 95)) / 17)
                    HI_full[low_rh_mask] -= adjustment
                
                high_rh_mask = (R_full > 85) & (T_full >= 80) & (T_full <= 87)
                if high_rh_mask.any():
                    adjustment = ((R_full[high_rh_mask] - 85) / 10) * ((87 - T_full[high_rh_mask]) / 5)
                    HI_full[high_rh_mask] += adjustment
                
                HI[needs_full] = HI_full
            
            hi_f[mask] = HI
    else:
        # Scalar case
        if t_f < 80:
            hi_f = t_f
        else:
            T = t_f
            R = rh
            HI = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (R * 0.094))
            
            if (HI + T) / 2 >= 80:
                HI = (-42.379 + 
                      2.04901523 * T + 
                      10.14333127 * R - 
                      0.22475541 * T * R - 
                      6.83783e-3 * T**2 - 
                      5.481717e-2 * R**2 + 
                      1.22874e-3 * T**2 * R + 
                      8.5282e-4 * T * R**2 - 
                      1.99e-6 * T**2 * R**2)
            
            hi_f = HI
    
    # Convert back to Kelvin
    hi_k = (hi_f - 32) * 5/9 + 273.15
    
    return hi_k


# ============================================================================
# PROCESS ERA5 DATA
# ============================================================================

def process_era5_to_heat_index(nc_file, output_file=None):
    """
    Process ERA5 NetCDF file to calculate heat index
    
    Args:
        nc_file: Path to ERA5 NetCDF file
        output_file: Optional output path (default: adds '_with_hi' to input name)
    
    Returns:
        xarray Dataset with heat index added
    """
    print(f"\n{'='*80}")
    print(f"Processing: {nc_file}")
    print('='*80)
    
    # Load data
    print("Loading ERA5 data...")
    ds = xr.open_dataset(nc_file)
    
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.dims)}")
    print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Calculate relative humidity
    print("\nCalculating relative humidity...")
    rh = calculate_relative_humidity_era5(ds['t2m'], ds['d2m'])
    
    # Calculate heat index
    print("Calculating heat index...")
    hi = calculate_heat_index_era5(ds['t2m'], rh)
    
    # Add to dataset
    ds['rh'] = rh
    ds['rh'].attrs['units'] = '%'
    ds['rh'].attrs['long_name'] = 'Relative Humidity'
    
    ds['heat_index'] = hi
    ds['heat_index'].attrs['units'] = 'K'
    ds['heat_index'].attrs['long_name'] = 'Heat Index (Apparent Temperature)'
    
    # Save
    if output_file is None:
        output_file = nc_file.replace('.nc', '_with_hi.nc')
    
    print(f"\nSaving to: {output_file}")
    ds.to_netcdf(output_file)
    
    print("✓ Processing complete")
    
    return ds


# ============================================================================
# EXTRACT STATION DATA
# ============================================================================

def extract_station_data(ds, stations=STATIONS):
    """
    Extract time series for each station location from gridded ERA5 data
    
    Args:
        ds: xarray Dataset with heat index
        stations: Dictionary of station info
    
    Returns:
        Dictionary of DataFrames (one per station)
    """
    print(f"\n{'='*80}")
    print("EXTRACTING STATION DATA FROM ERA5 GRID")
    print('='*80)
    
    station_data = {}
    
    for station_code, info in stations.items():
        print(f"\n{info['name']} ({station_code}):")
        print(f"  Coordinates: {info['lat']:.4f}°N, {info['lon']:.4f}°W")
        
        # Select nearest grid point
        station_ds = ds.sel(
            latitude=info['lat'], 
            longitude=info['lon'], 
            method='nearest'
        )
        
        # Convert to DataFrame
        df = station_ds.to_dataframe().reset_index()
        
        # Keep only needed columns
        df = df[['time', 't2m', 'd2m', 'rh', 'heat_index']]
        df.columns = ['datetime', 't2m', 'd2m', 'rh', 'heat_index']
        
        # Convert temperatures from Kelvin to Fahrenheit
        df['temp_f'] = (df['t2m'] - 273.15) * 9/5 + 32
        df['dwpt_f'] = (df['d2m'] - 273.15) * 9/5 + 32
        df['heat_index_f'] = (df['heat_index'] - 273.15) * 9/5 + 32
        
        station_data[station_code] = df
        
        print(f"  ✓ Extracted {len(df)} hourly records")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return station_data


def aggregate_to_daily(station_data):
    """
    Aggregate hourly data to daily max/min heat index
    
    Args:
        station_data: Dictionary of hourly DataFrames
    
    Returns:
        Dictionary of daily DataFrames
    """
    print(f"\n{'='*80}")
    print("AGGREGATING TO DAILY MAX/MIN HEAT INDEX")
    print('='*80)
    
    daily_data = {}
    
    for station_code, df in station_data.items():
        print(f"\n{station_code}:")
        
        # Group by date
        df['date'] = pd.to_datetime(df['datetime']).dt.date
        
        daily = df.groupby('date').agg({
            'heat_index_f': ['max', 'min']
        }).reset_index()
        
        daily.columns = ['datetime', 'heatindexmax2m', 'heatindexmin2m']
        daily['datetime'] = pd.to_datetime(daily['datetime'])
        
        daily_data[station_code] = daily
        
        print(f"  ✓ {len(daily)} days")
        print(f"  Max heat index: {daily['heatindexmax2m'].max():.1f}°F")
        print(f"  Min heat index: {daily['heatindexmin2m'].min():.1f}°F")
        
        # Save to Excel
        output_file = f"{station_code}heatindex20222024_era5.xlsx"
        daily.to_excel(output_file, index=False)
        print(f"  ✓ Saved: {output_file}")
    
    return daily_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ERA5 DATA DOWNLOAD AND PROCESSING FOR NORTH CAROLINA (2022-2024)")
    print("="*80)
    print("\nThis script will:")
    print("1. Download ERA5 temperature and dewpoint data (2022-2024)")
    print("2. Calculate heat index for the entire grid")
    print("3. Extract data at station locations")
    print("4. Aggregate to daily max/min heat index")
    print("5. Save as Excel files matching your existing format")
    
    print("\n⚠️  IMPORTANT:")
    print("- You need a CDS API account: https://cds.climate.copernicus.eu/")
    print("- Install cdsapi: pip install cdsapi")
    print("- Setup ~/.cdsapirc with your API credentials")
    print("- Total download size: ~1.5-3 GB")
    print("- Download time: 10-30 minutes (depends on CDS queue)")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Step 1: Download ERA5 data
    print("\n" + "="*80)
    print("STEP 1: DOWNLOADING ERA5 DATA")
    print("="*80)
    
    downloaded_files = download_all_years()
    
    if not downloaded_files:
        print("\n✗ No files downloaded. Exiting.")
        exit(1)
    
    # Step 2: Process each file
    print("\n" + "="*80)
    print("STEP 2: CALCULATING HEAT INDEX")
    print("="*80)
    
    processed_files = []
    for nc_file in downloaded_files:
        ds = process_era5_to_heat_index(nc_file)
        processed_files.append(nc_file.replace('.nc', '_with_hi.nc'))
    
    # Step 3: Extract station data and aggregate
    print("\n" + "="*80)
    print("STEP 3: EXTRACTING STATION DATA")
    print("="*80)
    
    all_station_data = {}
    
    for nc_file in processed_files:
        print(f"\nProcessing: {nc_file}")
        ds = xr.open_dataset(nc_file)
        station_data = extract_station_data(ds)
        
        # Merge with existing data
        for station_code, df in station_data.items():
            if station_code not in all_station_data:
                all_station_data[station_code] = []
            all_station_data[station_code].append(df)
    
    # Combine all years for each station
    print("\n" + "="*80)
    print("STEP 4: COMBINING ALL YEARS")
    print("="*80)
    
    for station_code in all_station_data.keys():
        combined = pd.concat(all_station_data[station_code], ignore_index=True)
        all_station_data[station_code] = combined
        print(f"{station_code}: {len(combined)} hours total")
    
    # Step 4: Aggregate to daily
    daily_data = aggregate_to_daily(all_station_data)
    
    # Summary
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nFiles created:")
    for station_code in STATIONS.keys():
        print(f"  ✓ {station_code}heatindex20222024_era5.xlsx")
    
    print("\nNext steps:")
    print("1. Review the Excel files")
    print("2. Run merge_historical_data_era5.py to combine with 1971-2021 data")
    print("3. Update your dashboard")
