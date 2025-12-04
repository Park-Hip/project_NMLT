import pandas as pd
import numpy as np

df = pd.read_csv("data/flood_data.csv")

def inject_full_stealth_physics(df):
    df_c = df.copy()

    # --- COMPONENT 1: THE MONSOON (Sigmoid / S-Curve) ---
    # As discussed: Safe -> Tipping Point -> Disaster
    monsoon_risk = 1 / (1 + np.exp(-(df_c['MonsoonIntensity'] - 10)))

    # --- COMPONENT 2: RIVER MANAGEMENT (Diminishing Returns) ---
    # Logic: Management reduces risk, but effectiveness slows down (Log curve).
    # We subtract this from the risk.
    # (Input 0-20 -> Output 0-3.0)
    management_defense = np.log(df_c['RiverManagement'] + 1) * 0.8

    # --- COMPONENT 3: DEFORESTATION (Accelerating Damage) ---
    # Logic: Nature can heal small cuts, but large cuts cause collapse.
    # We use Power 1.5. (Input 0-20 -> Output 0-4.0 after scaling)
    deforest_risk = (df_c['Deforestation'] ** 1.5) / 20.0

    # --- COMPONENT 4: DRAINAGE (Threshold Step) ---
    # Logic: If Drainage is < 5, the system FAILS.
    # This adds a flat "Penalty" block of risk.
    drainage_fail = (df_c['DrainageSystems'] < 5).astype(int) * 0.25

    # --- COMPONENT 5: POLITICS (The Multiplier) ---
    # Logic: Bad Politics amplifies all other risks by up to 50%.
    # 'PoliticalFactors' 0 (Good) -> Multiplier 1.0
    # 'PoliticalFactors' 20 (Bad) -> Multiplier 1.5
    corruption_multiplier = 1 + (df_c['PoliticalFactors'] / 40.0)

    # --- FINAL FORMULA ---
    # Base Risk = Monsoon + Deforestation - Defense + Broken Drainage
    base_risk = (monsoon_risk * 2.0) + deforest_risk - management_defense + drainage_fail

    # Apply the Political Multiplier
    total_risk = base_risk * corruption_multiplier

    # --- NORMALIZE & NOISE ---
    # Scale to 0-1
    df_c['FloodProbability'] = (total_risk - total_risk.min()) / (total_risk.max() - total_risk.min())

    # Add real-world noise (The "Chaos Factor")
    noise = np.random.normal(0, 0.04, size=len(df_c))
    df_c['FloodProbability'] = df_c['FloodProbability'] + noise
    df_c['FloodProbability'] = df_c['FloodProbability'].clip(0, 1)

    return df_c

df_final = inject_full_stealth_physics(df)
df_final.to_csv("data/flood_dataset.csv", index=False)
