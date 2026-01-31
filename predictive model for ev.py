import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# --- 1. SETUP FILE PATHS ---
input_file = r"C:\Users\Intel\Downloads\ev_final_predictive_model_o.csv"
output_file = r"C:\Users\Intel\Downloads\ev_final_predictive_model_v5.csv"

# --- 2. LOAD DATA ---
if not os.path.exists(input_file):
    print(f"Error: Could not find the input file at {input_file}")
    exit()

df = pd.read_csv(input_file)

# Policy Map
policy_map = {
    'Maharashtra': 1.5, 'Delhi': 1.8, 'Karnataka': 1.6, 'Tamil Nadu': 1.4,
    'Gujarat': 1.3, 'Uttar Pradesh': 1.2, 'Kerala': 1.5, 'Telangana': 1.4
}

def enrich_historical_data(row):
    state, year, stations = row['state'], row['year'], row['charging_stations']
    policy = policy_map.get(state, 1.0)
    ev_share = (0.01 if year < 2020 else 0.04) * (stations / 400) * policy
    ev_share = min(ev_share, 0.18)
    total_market = 600000 * (1.03 ** (year - 2016))
    ev_sales = int(total_market * ev_share)
    ice_sales = int(total_market - ev_sales)
    return pd.Series([ev_sales, ice_sales, policy])

print("Enriching historical data...")
df[['ev_sales', 'ice_sales', 'policy_score']] = df.apply(enrich_historical_data, axis=1)
df['data_type'] = 'Historical'

# --- 3. PREDICTIVE MODEL ---
print("Running predictive models...")
forecast_data = []
states = df['state'].unique()

for state in states:
    state_df = df[df['state'] == state].sort_values('year')
    X = state_df[['year']].values
    
    # Train models
    model_ev = LinearRegression().fit(X, state_df['ev_sales'].values)
    model_ice = LinearRegression().fit(X, state_df['ice_sales'].values)
    model_fast = LinearRegression().fit(X, state_df['fast_charger_pct'].values)
    model_urban = LinearRegression().fit(X, state_df['urban_coverage_pct'].values)
    
    for future_year in [2025, 2026, 2027]:
        X_f = np.array([[future_year]])
        
        # Calculate infra growth trend
        infra_growth = state_df['charging_stations'].diff().mean()
        f_stations = int(state_df['charging_stations'].iloc[-1] + (infra_growth * (future_year - 2024)))
        
        # Predict values
        p_fast = np.clip(model_fast.predict(X_f)[0], state_df['fast_charger_pct'].min(), 1.0)
        p_urban = np.clip(model_urban.predict(X_f)[0], state_df['urban_coverage_pct'].min(), 1.0)
        p_ev = max(int(model_ev.predict(X_f)[0]), int(state_df['ev_sales'].max() * 1.05))
        p_ice = max(int(model_ice.predict(X_f)[0]), 1000)
        
        forecast_data.append({
            'state': state, 'year': future_year, 'charging_stations': max(f_stations, 0),
            'fast_charger_pct': round(p_fast, 4), 'urban_coverage_pct': round(p_urban, 4),
            'ev_sales': p_ev, 'ice_sales': p_ice, 'policy_score': state_df['policy_score'].iloc[-1],
            'data_type': 'Forecast'
        })

# Combine Historical and Forecast
final_df = pd.concat([df, pd.DataFrame(forecast_data)], ignore_index=True)

# --- 4. CALCULATE MISSING COLUMNS (Winning Points) ---
print("Calculating Readiness Scores and Categories...")

# A. Sort for Growth calculation
final_df = final_df.sort_values(['state', 'year'])

# B. Normalized Charging Stations (Global Scale)
final_df['normalized_charging_stations'] = final_df['charging_stations'] / final_df['charging_stations'].max()

# C. EV Readiness Score
# Formula: 50% Infra + 30% Fast Charging + 20% Urban Reach
final_df['ev_readiness_score'] = (
    (0.5 * final_df['normalized_charging_stations']) + 
    (0.3 * final_df['fast_charger_pct']) + 
    (0.2 * final_df['urban_coverage_pct'])
)

# D. Charging Growth %
final_df['charging_growth_pct'] = final_df.groupby('state')['charging_stations'].pct_change().fillna(0)

# E. EV Transition Category
final_df['ev_transition_category'] = pd.cut(
    final_df['ev_readiness_score'],
    bins=[0, 0.3, 0.6, 1.1],
    labels=['Emerging (Low)', 'Developing (Mid)', 'Leader (High)']
)

# F. Derived Sales Metrics
final_df['ev_2w'] = (final_df['ev_sales'] * 0.75).astype(int)
final_df['ev_3w'] = (final_df['ev_sales'] * 0.15).astype(int)
final_df['ev_4w'] = (final_df['ev_sales'] - final_df['ev_2w'] - final_df['ev_3w']).astype(int)
final_df['ev_penetration_pct'] = (final_df['ev_sales'] / (final_df['ev_sales'] + final_df['ice_sales'])) * 100

# --- 5. EXPORT ---
try:
    final_df.to_csv(output_file, index=False)
    print(f"\nSUCCESS! Complete data saved to: {output_file}")
except PermissionError:
    print(f"\n!!! ERROR: Access Denied !!!")
    print(f"Please CLOSE the file '{output_file}' in Tableau/Excel and run this again.")