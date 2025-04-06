import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Set page config
st.set_page_config(layout="wide", page_title="Bowler Injury Prediction Dashboard")

# Title and description
st.title("Bowler Injury Prediction Analysis")
st.markdown("This dashboard visualizes fatigue metrics across matches and dates for three bowlers.")

# File uploader for player data
st.sidebar.header("Upload Player Data")
player1_file = st.sidebar.file_uploader("Upload Player 1 Data (CSV)", type=['csv'])
player2_file = st.sidebar.file_uploader("Upload Player 2 Data (CSV)", type=['csv'])
player3_file = st.sidebar.file_uploader("Upload Player 3 Data (CSV)", type=['csv'])

# Load data function
def load_player_data(file, player_name):
    if file is not None:
        df = pd.read_csv(file)
        
        # Convert date column if it exists
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()]
        if date_columns:
            for date_col in date_columns:
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                except:
                    pass
                    
        # Add player identifier
        df['Player'] = player_name
        return df
    return None

# Process data
player1_data = load_player_data(player1_file, "Player 1")
player2_data = load_player_data(player2_file, "Player 2")
player3_data = load_player_data(player3_file, "Player 3")

# Combine available data
player_dfs = [df for df in [player1_data, player2_data, player3_data] if df is not None]

if player_dfs:
    all_data = pd.concat(player_dfs, ignore_index=True)
    
    # Process the data to ensure we have fatigue metrics
    def ensure_fatigue_metrics(df):
        # If these metrics already exist, keep them
        # Otherwise, calculate them based on available data
        
        if 'Bowling_Intensity' not in df.columns and all(col in df.columns for col in ['Overs', 'Runs', 'Wick']):
            df['Bowling_Intensity'] = (
                (df['Overs'] * 6) +
                (df['Runs'] * 2) +
                (df['Wick'] * 20)
            )
        
        # Check for match date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'day' in col.lower():
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_col = col
                    break
        
        # Calculate days since last match if date column exists
        if date_col and 'Days_Since_Last_Match' not in df.columns:
            df = df.sort_values(by=date_col)
            df['Days_Since_Last_Match'] = df.groupby('Player')[date_col].diff().dt.days.fillna(0)
        
        # Calculate cumulative workload if not already present
        if 'Cumulative_Workload' not in df.columns and 'Bowling_Intensity' in df.columns:
            df['Cumulative_Workload'] = df.groupby('Player')['Bowling_Intensity'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
        
        # Calculate workload variance if not already present
        if 'Workload_Variance' not in df.columns and 'Bowling_Intensity' in df.columns:
            df['Workload_Variance'] = df.groupby('Player')['Bowling_Intensity'].rolling(window=3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
        
        return df
    
    all_data = ensure_fatigue_metrics(all_data)
    
    # Data is loaded, show controls
    st.sidebar.header("Dashboard Controls")
    
    available_players = all_data['Player'].unique().tolist()
    selected_players = st.sidebar.multiselect(
        "Select Players to Display",
        options=available_players,
        default=available_players
    )
    
    # Identify fatigue metrics
    fatigue_metrics = []
    for col in all_data.columns:
        if any(term in col.lower() for term in ['fatigue', 'workload', 'intensity', 'cumulative', 'variance', 'risk']):
            fatigue_metrics.append(col)
    
    # If no fatigue metrics found, suggest alternatives
    if not fatigue_metrics:
        numeric_cols = all_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        for col in numeric_cols:
            if col not in ['Player'] and not any(dim in col.lower() for dim in ['index', 'id']):
                fatigue_metrics.append(col)
    
    selected_metrics = st.sidebar.multiselect(
        "Select Fatigue Metrics to Visualize",
        options=fatigue_metrics,
        default=fatigue_metrics[:3] if len(fatigue_metrics) >= 3 else fatigue_metrics
    )
    
    # Identify date/match columns
    date_columns = []
    match_columns = []
    
    for col in all_data.columns:
        if any(term in col.lower() for term in ['date', 'day']):
            if pd.api.types.is_datetime64_any_dtype(all_data[col]):
                date_columns.append(col)
        if any(term in col.lower() for term in ['match', 'game', 'session']):
            match_columns.append(col)
    
    # If no match column found but we have index, use that
    if not match_columns and 'index' in all_data.columns:
        match_columns.append('index')
    # If still no match columns, create a simple match index
    if not match_columns:
        all_data['Match_Index'] = all_data.groupby('Player').cumcount() + 1
        match_columns.append('Match_Index')
    
    x_axis_options = date_columns + match_columns
    x_axis = st.sidebar.selectbox(
        "Select X-Axis (Date or Match)",
        options=x_axis_options,
        index=0 if date_columns else 0
    )
    
    # Filter data based on selection
    filtered_data = all_data[all_data['Player'].isin(selected_players)]
    
    # Main content - Fatigue vs Match/Date visualization
    st.header(f"Fatigue Metrics vs {x_axis}")
    
    # Line charts for selected metrics
    for metric in selected_metrics:
        if metric in filtered_data.columns:
            fig = px.line(filtered_data, x=x_axis, y=metric, color='Player',
                         title=f"{metric} Over {x_axis} by Player",
                         markers=True)
            
            # Add a horizontal line for average if numeric
            if pd.api.types.is_numeric_dtype(filtered_data[metric]):
                avg_value = filtered_data[metric].mean()
                fig.add_shape(
                    type="line",
                    line=dict(dash="dash", color="gray"),
                    y0=avg_value, y1=avg_value,
                    x0=0, x1=1,
                    xref="paper"
                )
                
            st.plotly_chart(fig, use_container_width=True)
    
    # Combined view of all metrics for comparison
    if len(selected_metrics) >= 2:
        st.header("Combined Metrics Comparison")
        
        # Create subplots - one row per player
        fig = make_subplots(
            rows=len(selected_players),
            cols=1,
            subplot_titles=[f"{player} - Fatigue Metrics" for player in selected_players]
        )
        
        for i, player in enumerate(selected_players):
            player_data = filtered_data[filtered_data['Player'] == player]
            
            for j, metric in enumerate(selected_metrics):
                if metric in player_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=player_data[x_axis],
                            y=player_data[metric],
                            name=f"{player} - {metric}",
                            mode='lines+markers',
                            line=dict(dash='solid' if j % 2 == 0 else 'dash')
                        ),
                        row=i+1,
                        col=1
                    )
        
        fig.update_layout(height=300 * len(selected_players), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap visualization for workload/fatigue patterns
    st.header("Fatigue Patterns Heatmap")
    
    if len(selected_metrics) >= 2:
        # Create a pivot table-like structure for the heatmap
        for player in selected_players:
            player_data = filtered_data[filtered_data['Player'] == player]
            
            # Convert datetime to string for heatmap if needed
            if pd.api.types.is_datetime64_any_dtype(player_data[x_axis]):
                x_values = player_data[x_axis].dt.strftime('%Y-%m-%d').values
            else:
                x_values = player_data[x_axis].values
                
            # Create heatmap data
            heatmap_data = []
            for i, metric in enumerate(selected_metrics):
                if metric in player_data.columns:
                    for j, x_val in enumerate(x_values):
                        heatmap_data.append({
                            'X': x_val,
                            'Metric': metric,
                            'Value': player_data.iloc[j][metric]
                        })
            
            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data)
                
                # Create heatmap
                fig = px.density_heatmap(
                    heatmap_df,
                    x='X',
                    y='Metric',
                    z='Value',
                    color_continuous_scale='Blues',
                    title=f"Fatigue Metrics Heatmap for {player}"
                )
                
                fig.update_layout(
                    xaxis_title=x_axis,
                    yaxis_title="Metric"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Show data table with key metrics
    st.header("Player Data")
    display_cols = ['Player', x_axis] + selected_metrics
    display_cols = [col for col in display_cols if col in filtered_data.columns]
    st.dataframe(filtered_data[display_cols])

else:
    st.info("Please upload data files for at least one player to begin visualization.")