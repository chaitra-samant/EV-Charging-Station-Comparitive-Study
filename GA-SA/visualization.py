"""
Visualization on Geographic Map of Delhi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import os

# Delhi City boundary coordinates 
DELHI_BOUNDARY = [
    [28.883079, 77.089492],  
    [28.885109, 77.345581], 
    [28.418121, 77.298889],  
    [28.404794, 76.837349],  
    [28.883079, 77.089492]   
]

def plot_delhi_map_enhanced(selected_stations_df, all_stations_df, p, 
                            output_path='results/maps'):

    delhi_center = [28.6139, 77.2090]
    
    # Create map
    m = folium.Map(
        location=delhi_center,
        zoom_start=11,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Delhi Boundary Polygon
    folium.Polygon(
        locations=DELHI_BOUNDARY,
        color='#FF4500',
        weight=3,
        fill=True,
        fillColor='#FFD700',
        fillOpacity=0.1,
        popup='Delhi NCT Boundary',
        tooltip='Delhi Region'
    ).add_to(m)
    
    # Heatmap (all candidate stations showing density)
    heat_data = [[row['lattitude'], row['longitude']] 
                 for _, row in all_stations_df.iterrows()]
    
    HeatMap(
        heat_data,
        min_opacity=0.2,
        max_opacity=0.6,
        radius=15,
        blur=20,
        gradient={
            0.0: 'blue',
            0.4: 'lime',
            0.6: 'yellow',
            0.8: 'orange',
            1.0: 'red'
        }
    ).add_to(m)
    
    # All candidate stations (small gray dots)
    for _, station in all_stations_df.iterrows():
        folium.CircleMarker(
            location=[station['lattitude'], station['longitude']],
            radius=3,
            popup=f"<b>Candidate</b><br>{station['name']}",
            color='gray',
            fill=True,
            fillColor='lightgray',
            fillOpacity=0.4,
            weight=1
        ).add_to(m)
    
    # Add selected stations (prominent green markers)
    for _, station in selected_stations_df.iterrows():
        # Add coverage circle (5km radius)
        folium.Circle(
            location=[station['lattitude'], station['longitude']],
            radius=2500,  # 5 km in meters
            color='#2E8B57',
            fill=True,
            fillColor='#90EE90',
            fillOpacity=0.15,
            weight=2,
            popup=f"2.5km coverage from {station['name']}"
        ).add_to(m)
        
        # Add marker
        folium.Marker(
            location=[station['lattitude'], station['longitude']],
            popup=folium.Popup(
                f"<b>SELECTED STATION</b><br>"
                f"<b>{station['name']}</b><br>"
                f"{station['city']}<br>"
                f"{station['address']}<br>"
                f"<i>Coordinates: ({station['lattitude']:.4f}, {station['longitude']:.4f})</i>",
                max_width=300
            ),
            tooltip=f"⚡ {station['name']}",
            icon=folium.Icon(color='green', icon='bolt', prefix='fa')
        ).add_to(m)
    
    # 5. Enhanced Legend
    legend_html = f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 240px; 
                background-color: white; border:3px solid #FF4500; z-index:9999; 
                padding: 12px; font-size:13px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <p style="margin:0; font-weight:bold; font-size:15px; color:#FF4500;">
            📍 Delhi EV Optimization
        </p>
        <hr style="margin: 8px 0; border: 1px solid #eee;">
        <p style="margin:5px 0;">
            <i class="fa fa-bolt" style="color:green;"></i> 
            <b>Selected Stations:</b> {len(selected_stations_df)}
        </p>
        <p style="margin:5px 0;">
            <span style="color:gray;">●</span> 
            Total Candidates: {len(all_stations_df)}
        </p>
        <p style="margin:5px 0;">
            <span style="color:#90EE90;">○</span> 
            Coverage: 5 km radius
        </p>
        <p style="margin:5px 0;">
            <span style="background: linear-gradient(to right, blue, red); 
                         padding: 2px 8px; border-radius: 3px; color: white;">
                Heatmap
            </span> Station Density
        </p>
        <p style="margin:5px 0;">
            <span style="color:#FF4500; font-weight:bold;">―</span> 
            Delhi Boundary
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 6. Title
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50px; width: 420px; 
                background-color: white; border:3px solid #2E8B57; z-index:9999; 
                padding: 12px; font-weight:bold; font-size:17px; border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        ⚡ EV Charging Station Optimization - Delhi (p={p}) - Hybrid GA-SA
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save
    os.makedirs(output_path, exist_ok=True)
    map_file = os.path.join(output_path, f'delhi_p{p}_map.html')
    m.save(map_file)
    print(f"Enhanced map saved: {map_file}")
    return map_file

def plot_convergence_single(convergence_data, p, output_path='results/plots'):
    """
    Create convergence plot (same as before).
    """
    plt.figure(figsize=(10, 6))
    
    generations = range(len(convergence_data))
    plt.plot(generations, convergence_data, 'b-', linewidth=2)
    
    plt.xlabel('Generation', fontsize=12, fontweight='bold')
    plt.ylabel('Best Fitness (km)', fontsize=12, fontweight='bold')
    plt.title(f'Hybrid GA-SA Convergence - p={p} Stations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Annotate final value
    final = convergence_data[-1]
    plt.annotate(f'Final: {final:,.0f} km',
                xy=(len(convergence_data)-1, final),
                xytext=(len(convergence_data)*0.6, final*1.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    os.makedirs(output_path, exist_ok=True)
    plot_file = os.path.join(output_path, f'convergence_p{p}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Convergence plot saved: {plot_file}")
    return plot_file

def plot_cost_comparison(results_df, output_path='results/plots'):
    """
    Create cost comparison chart (same as before).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort by p value
    results_df = results_df.sort_values('num_stations')
    
    # Plot 1: Total Cost
    ax1.plot(results_df['num_stations'], results_df['total_cost_km'], 
             'o-', linewidth=2, markersize=8, color='#2E8B57')
    ax1.set_xlabel('Number of Stations (p)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Cost (km)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Distance vs Stations (Hybrid GA-SA)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Annotate values
    for _, row in results_df.iterrows():
        ax1.annotate(f"{row['total_cost_km']:,.0f}", 
                    xy=(row['num_stations'], row['total_cost_km']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=10)
    
    # Plot 2: Average Distance per EV
    ax2.plot(results_df['num_stations'], results_df['avg_distance_km'], 
             's-', linewidth=2, markersize=8, color='#4169E1')
    ax2.set_xlabel('Number of Stations (p)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Avg Distance per EV (km)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Customer Distance vs Stations (Hybrid GA-SA)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Annotate values
    for _, row in results_df.iterrows():
        ax2.annotate(f"{row['avg_distance_km']:.3f}", 
                    xy=(row['num_stations'], row['avg_distance_km']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs(output_path, exist_ok=True)
    plot_file = os.path.join(output_path, 'cost_comparison.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Cost comparison saved: {plot_file}")
    return plot_file

def create_visualizations(results_csv='results/ga_results.csv',
                         stations_csv='../dataset/delhi_ev_charging.csv'):
    """
    Generate all visualizations with enhanced maps.
    """
    print("Creating Visualizations")
    
    
    # Load data
    all_stations = pd.read_csv(stations_csv)
    
    if not os.path.exists(results_csv):
        print("Results not found. Run genetic_algorithm.py first.")
        return
    
    results = pd.read_csv(results_csv)
    
    print(f"\nGenerating outputs for p = {results['num_stations'].tolist()}")
    
    # Individual convergence plots and enhanced maps
    for _, row in results.iterrows():
        p = int(row['num_stations'])
        print(f"\nProcessing p={p}...")
        
        # Convergence plot
        conv_file = f'results/data/convergence_p{p}.csv'
        if os.path.exists(conv_file):
            conv_data = pd.read_csv(conv_file)
            plot_convergence_single(conv_data['best_cost'].values, p)
        
        # Enhanced map with heatmap + boundary
        selected_file = f'results/data/selected_stations_p{p}.csv'
        if os.path.exists(selected_file):
            selected_stations = pd.read_csv(selected_file)
            plot_delhi_map_enhanced(selected_stations, all_stations, p)
    
    # 2. Cost comparison chart
    print(f"\nCreating cost comparison...")
    plot_cost_comparison(results)
    
if __name__ == "__main__":
    create_visualizations()
