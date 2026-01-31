import pandas as pd

df = pd.read_csv(r"C:\Users\Intel\Downloads\ev_final_predictive_model_V3.csv")
df.head()

df['normalized_charging_stations'] = (
    df['charging_stations'] / df['charging_stations'].max()
)

df['fast_charger_pct'] = df['fast_charger_pct'] / 100
df['urban_coverage_pct'] = df['urban_coverage_pct'] / 100

df['ev_readiness_score'] = (
    0.5 * df['normalized_charging_stations'] +
    0.3 * df['fast_charger_pct'] +
    0.2 * df['urban_coverage_pct']
)

df = df.sort_values(['state', 'year'])

df['charging_growth_pct'] = (
    df.groupby('state')['charging_stations']
      .pct_change()
)


df['ev_transition_category'] = pd.cut(
    df['ev_readiness_score'],
    bins=[0, 0.5, 0.75, 1],
    labels=['Low', 'Medium', 'High']
)

df.to_excel("ev_dashboard_ready_data.xlsx", index=False)