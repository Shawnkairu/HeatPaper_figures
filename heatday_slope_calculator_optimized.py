"""
Optimized Heat Day Slope Calculator for ERA5 Gridded Data
Using vectorized xarray operations for efficiency
"""

import streamlit as st
import numpy as np
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects


def calculate_heatday_trends_optimized(era5_file, percentile=0.98, months=[3,4,5,6,7,8,9,10]):
    """
    Optimized calculation of heat day trends using vectorized xarray operations
    
    This is MUCH faster than looping through each date and grid cell individually.
    
    Args:
        era5_file: path to ERA5 NetCDF file
        percentile: percentile threshold (0.98 for 98th percentile)
        months: list of months to include
    
    Returns:
        lats, lons, slopes, r_squared, p_values
    """
    
    st.info("📂 Loading ERA5 data...")
    
    # Load data
    ds = xr.open_dataset(era5_file)
    temp = ds['t2m']
    temp_c = temp - 273.15  # Convert K to C
    
    # Determine time dimension
    time_dim = 'valid_time' if 'valid_time' in temp_c.dims else 'time'
    
    # Add time coordinates
    temp_c = temp_c.assign_coords({
        'month': (time_dim, temp_c[time_dim].dt.month.values),
        'year': (time_dim, temp_c[time_dim].dt.year.values),
        'dayofyear': (time_dim, temp_c[time_dim].dt.dayofyear.values)
    })
    
    st.info(f"✅ Loaded {len(temp_c.latitude)} x {len(temp_c.longitude)} grid, {len(temp_c[time_dim])} time steps")
    
    # Filter to summer months
    st.info("🌞 Filtering to March-October...")
    summer_mask = temp_c['month'].isin(months)
    temp_summer = temp_c.where(summer_mask, drop=True)
    
    # Calculate date-specific percentiles using groupby
    st.info("📊 Calculating date-specific 98th percentiles for all grid cells...")
    
    # Group by day-of-year and calculate percentile
    # This is vectorized across all grid cells simultaneously - much faster!
    percentile_thresholds = temp_summer.groupby('dayofyear').quantile(percentile, dim=time_dim)
    
    st.success(f"✅ Calculated thresholds for {len(percentile_thresholds.dayofyear)} unique days")
    
    # For each day, check if it exceeds its threshold
    st.info("🔥 Identifying extreme heat days...")
    
    # Broadcast and compare
    # This creates a boolean mask where True = heat day
    is_heatday = temp_summer.groupby('dayofyear') > percentile_thresholds
    
    # Count heat days per year for each grid cell
    st.info("📅 Counting heat days per year for each grid cell...")
    heatdays_per_year = is_heatday.groupby('year').sum(dim=time_dim)
    
    st.success(f"✅ Calculated heat days for {len(heatdays_per_year.year)} years")
    
    # Calculate linear trends
    st.info("📈 Calculating trends (this may take a minute)...")
    
    years = heatdays_per_year.year.values
    lats = heatdays_per_year.latitude.values
    lons = heatdays_per_year.longitude.values
    
    # Initialize output arrays
    slopes = np.full((len(lats), len(lons)), np.nan)
    r_squared = np.full((len(lats), len(lons)), np.nan)
    p_values = np.full((len(lats), len(lons)), np.nan)
    
    # Calculate trends for each grid cell
    progress_bar = st.progress(0)
    total = len(lats) * len(lons)
    count = 0
    
    for i in range(len(lats)):
        for j in range(len(lons)):
            heat_series = heatdays_per_year.isel(latitude=i, longitude=j).values
            
            # Only calculate if we have data
            if not np.any(np.isnan(heat_series)) and len(heat_series) > 2:
                try:
                    slope, intercept, r_val, p_val, std_err = stats.linregress(years, heat_series)
                    slopes[i, j] = slope * 10  # Convert to days per decade
                    r_squared[i, j] = r_val ** 2
                    p_values[i, j] = p_val
                except:
                    pass
            
            count += 1
            if count % 50 == 0:
                progress_bar.progress(count / total)
    
    progress_bar.progress(1.0)
    st.success("✅ Trend calculation complete!")
    
    # Close dataset
    ds.close()
    
    return lats, lons, slopes, r_squared, p_values, heatdays_per_year


def plot_heatday_slopes(lats, lons, slopes, p_values=None, significance_threshold=0.05):
    """
    Create heat day slope heatmap
    """
    
    # Create meshgrid
    lons_mesh, lats_mesh = np.meshgrid(lons, lats)
    
    # If showing only significant, mask non-significant values
    if p_values is not None:
        slopes_plot = slopes.copy()
        slopes_plot[p_values >= significance_threshold] = np.nan
        title = f"Significant Heat Day Trends (p<{significance_threshold})"
    else:
        slopes_plot = slopes
        title = "Heat Day Trends (1974-2024)"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Diverging colormap centered at zero
    cmap = LinearSegmentedColormap.from_list(
        'heat_trend',
        ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
         '#f7f7f7',
         '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f']
    )
    
    # Symmetric scale
    max_abs = np.nanmax(np.abs(slopes_plot))
    vmin, vmax = -max_abs, max_abs
    
    # Plot
    contour = ax.contourf(lons_mesh, lats_mesh, slopes_plot, 
                          levels=40, cmap=cmap, vmin=vmin, vmax=vmax,
                          extend='both')
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', 
                        pad=0.02, fraction=0.046, shrink=0.8)
    cbar.set_label('Heat Day Trend (days/decade)', fontsize=13, weight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Add state borders
    try:
        import json
        import urllib.request
        
        url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
        with urllib.request.urlopen(url, timeout=10) as response:
            states = json.load(response)
        
        target_states = ['North Carolina', 'South Carolina', 'Virginia', 
                        'Tennessee', 'Georgia', 'West Virginia']
        
        for feature in states['features']:
            if feature['properties']['name'] in target_states:
                geom = feature['geometry']
                
                if geom['type'] == 'Polygon':
                    coords = geom['coordinates'][0]
                    xs, ys = zip(*coords)
                    
                    # Highlight NC in thicker line
                    if feature['properties']['name'] == 'North Carolina':
                        ax.plot(xs, ys, 'k-', linewidth=2.5, alpha=0.9, zorder=3)
                    else:
                        ax.plot(xs, ys, 'k-', linewidth=1.2, alpha=0.6, zorder=3)
                        
                elif geom['type'] == 'MultiPolygon':
                    for polygon in geom['coordinates']:
                        coords = polygon[0]
                        xs, ys = zip(*coords)
                        
                        if feature['properties']['name'] == 'North Carolina':
                            ax.plot(xs, ys, 'k-', linewidth=2.5, alpha=0.9, zorder=3)
                        else:
                            ax.plot(xs, ys, 'k-', linewidth=1.2, alpha=0.6, zorder=3)
    except:
        pass
    
    # Add NC stations
    stations = {
        'KAVL': (35.4352, -82.5415, 'Asheville'),
        'KGSO': (36.0978, -79.9373, 'Greensboro'),
        'KHSE': (35.2680, -75.6177, 'Cape Hatteras'),
        'KILM': (34.2704, -77.9026, 'Wilmington'),
        'KCLT': (35.2140, -80.9431, 'Charlotte'),
        'KRDU': (35.8774, -78.7875, 'Raleigh-Durham'),
    }
    
    for code, (lat, lon, name) in stations.items():
        ax.plot(lon, lat, 'ko', markersize=10, markeredgecolor='white', 
                markeredgewidth=2.5, zorder=4)
        txt = ax.text(lon, lat + 0.18, name, fontsize=10, ha='center', 
                     weight='bold', zorder=5, color='black')
        txt.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                              path_effects.Normal()])
    
    # Format
    ax.set_xlabel('Longitude', fontsize=13, weight='bold')
    ax.set_ylabel('Latitude', fontsize=13, weight='bold')
    ax.set_title(title + '\nDate-Specific 98th Percentile Method (Mar-Oct)', 
                 fontsize=15, weight='bold', pad=20)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    return fig


def main():
    """
    Main Streamlit app
    """
    
    st.set_page_config(page_title="Heat Day Trends - NC", layout="wide")
    
    st.title("🌡️ Spatial Heat Day Trend Analysis")
    st.markdown("### North Carolina & Surrounding Region (1974-2024)")
    st.markdown("---")
    
    with st.expander("ℹ️ About this Analysis", expanded=True):
        st.markdown("""
        **This tool calculates spatially-explicit trends in extreme heat days across North Carolina.**
        
        **Methodology:**
        - Uses ERA5 gridded climate reanalysis data (0.25° resolution)
        - Applies date-specific 98th percentile thresholds for each calendar day (March-October)
        - Counts extreme heat days each year at each grid cell
        - Calculates linear trend (slope) from 1974-2024
        
        **Interpretation:**
        - **Positive slopes (red)**: Increasing heat days over time
        - **Negative slopes (blue)**: Decreasing heat days over time
        - **Values**: Change in heat days per decade
        
        **Advantages over station-based analysis:**
        - Full spatial coverage (no interpolation needed)
        - Captures fine-scale geographic variation
        - More robust than sparse station network
        """)
    
    st.markdown("---")
    
    # File input
    st.subheader("📁 Data Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        file_path = st.text_input(
            "Path to ERA5 NetCDF file:",
            value="era5_temperature_nc_1974_2024.nc",
            help="File should contain 't2m' variable (2m temperature)"
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
        if os.path.exists(file_path):
            st.success("✅ File found")
        else:
            st.error("❌ File not found")
    
    # Processing options
    st.subheader("⚙️ Analysis Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        percentile = st.slider(
            "Percentile Threshold",
            min_value=90,
            max_value=99,
            value=98,
            help="98th percentile = top 2% of days"
        ) / 100
    
    with col2:
        show_significant_only = st.checkbox(
            "Show only statistically significant trends (p<0.05)",
            value=False
        )
    
    # Run analysis
    if st.button("🚀 Calculate Heat Day Trends", type="primary"):
        
        if not os.path.exists(file_path):
            st.error("❌ Please provide a valid file path")
            return
        
        st.markdown("---")
        
        try:
            # Calculate trends
            lats, lons, slopes, r_squared, p_values, heatdays_per_year = \
                calculate_heatday_trends_optimized(file_path, percentile=percentile)
            
            # Summary statistics
            st.markdown("---")
            st.subheader("📊 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mean_slope = np.nanmean(slopes)
                st.metric("Mean Trend", f"{mean_slope:.2f} days/decade",
                         delta=f"{'Increasing' if mean_slope > 0 else 'Decreasing'}")
            
            with col2:
                median_slope = np.nanmedian(slopes)
                st.metric("Median Trend", f"{median_slope:.2f} days/decade")
            
            with col3:
                pct_increasing = np.sum(slopes > 0) / np.sum(~np.isnan(slopes)) * 100
                st.metric("% Grid Cells Increasing", f"{pct_increasing:.1f}%")
            
            with col4:
                pct_sig = np.sum(p_values < 0.05) / np.sum(~np.isnan(p_values)) * 100
                st.metric("% Statistically Significant", f"{pct_sig:.1f}%")
            
            # Distribution of slopes
            st.markdown("---")
            st.subheader("📈 Distribution of Trends")
            
            import plotly.graph_objects as go
            
            slopes_flat = slopes[~np.isnan(slopes)].flatten()
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=slopes_flat,
                nbinsx=50,
                marker_color='steelblue',
                opacity=0.7
            ))
            fig_hist.update_layout(
                title="Distribution of Heat Day Slopes Across Grid Cells",
                xaxis_title="Slope (days/decade)",
                yaxis_title="Number of Grid Cells",
                height=400
            )
            
            # Add mean line
            fig_hist.add_vline(x=mean_slope, line_dash="dash", line_color="red",
                              annotation_text=f"Mean: {mean_slope:.2f}")
            fig_hist.add_vline(x=0, line_dash="dot", line_color="gray")
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Create map
            st.markdown("---")
            st.subheader("🗺️ Spatial Pattern of Heat Day Trends")
            
            if show_significant_only:
                fig_map = plot_heatday_slopes(lats, lons, slopes, p_values, 0.05)
            else:
                fig_map = plot_heatday_slopes(lats, lons, slopes)
            
            st.pyplot(fig_map)
            
            # Regional analysis
            st.markdown("---")
            st.subheader("🏔️ Regional Breakdown")
            
            st.markdown("""
            Compare trends across different regions of North Carolina:
            """)
            
            # Define regions (approximate lat/lon boxes)
            regions = {
                'Mountains (West)': {'lat': (35, 36.5), 'lon': (-84.5, -81)},
                'Piedmont (Central)': {'lat': (35, 36.5), 'lon': (-81, -78.5)},
                'Coastal Plain (East)': {'lat': (34, 36.5), 'lon': (-78.5, -75.5)}
            }
            
            region_stats = []
            
            for region_name, bounds in regions.items():
                lat_mask = (lats >= bounds['lat'][0]) & (lats <= bounds['lat'][1])
                lon_mask = (lons >= bounds['lon'][0]) & (lons <= bounds['lon'][1])
                
                # Get slopes for this region
                region_slopes = []
                for i, lat in enumerate(lats):
                    if lat_mask[i]:
                        for j, lon in enumerate(lons):
                            if lon_mask[j] and not np.isnan(slopes[i, j]):
                                region_slopes.append(slopes[i, j])
                
                if len(region_slopes) > 0:
                    region_stats.append({
                        'Region': region_name,
                        'Mean Trend (days/dec)': f"{np.mean(region_slopes):.2f}",
                        'Median Trend (days/dec)': f"{np.median(region_slopes):.2f}",
                        'Grid Cells': len(region_slopes)
                    })
            
            if region_stats:
                import pandas as pd
                df_regions = pd.DataFrame(region_stats)
                st.dataframe(df_regions, use_container_width=True, hide_index=True)
            
            # Export
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            if st.button("Prepare Data for Download"):
                import pandas as pd
                
                export_data = []
                for i in range(len(lats)):
                    for j in range(len(lons)):
                        export_data.append({
                            'latitude': lats[i],
                            'longitude': lons[j],
                            'slope_days_per_decade': slopes[i, j],
                            'r_squared': r_squared[i, j],
                            'p_value': p_values[i, j],
                            'significant': p_values[i, j] < 0.05
                        })
                
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name=f"heatday_slopes_nc_{int(percentile*100)}pct.csv",
                    mime="text/csv"
                )
                
                st.success(f"✅ {len(df_export)} grid cells ready for download")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.code(f"Full error:\n{__import__('traceback').format_exc()}")


if __name__ == "__main__":
    import os
    main()
