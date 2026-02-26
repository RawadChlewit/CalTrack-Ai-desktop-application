import os
import sys
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings

warnings.filterwarnings("ignore")

# ================== PATHS ==================
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "weight_change_dataset_with_daily_change.csv")
SYNTHETIC_PATH = os.path.join(BASE_DIR, "synthetic_weight_dataset_clean.csv")
AUGMENTED_PATH = os.path.join(BASE_DIR, "augmented_weight_dataset_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "trained_weight_model.pkl")

print("Base directory:", BASE_DIR)
print("Loading REAL dataset from:", DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise SystemExit(
        f"ERROR: '{DATA_PATH}' not found.\n"
        "Make sure 'weight_change_dataset_with_daily_change.csv' is in the same folder."
    )

df_real = pd.read_csv(DATA_PATH)

# ================== BASIC COLUMN CHECK ==================
required_cols = [
    "Age",
    "Gender",
    "Current Weight (lbs)",
    "BMR (Calories)",
    "Daily Calories Consumed",
    "Physical Activity Level",
    "Sleep Quality",
    "Stress Level",
    "Daily_Weight_Change",
]

missing = [c for c in required_cols if c not in df_real.columns]
if missing:
    raise SystemExit(f"Your dataset is missing required columns: {missing}")

# Keep only rows with all needed fields
df_real = df_real.dropna(subset=required_cols).reset_index(drop=True)

# ================== HELPER: ACTIVITY MULTIPLIER ==================
def activity_multiplier(level: str) -> float:
    level = (str(level) or "").lower()
    if "very" in level:
        return 1.725
    if "moderately" in level:
        return 1.55
    if "light" in level:
        return 1.375
    if "sedentary" in level:
        return 1.2
    return 1.3

# ================== SYNTHETIC GENERATOR (LOGICAL) ==================
def generate_logical_synthetic(df_orig: pd.DataFrame, n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic users with PHYSIOLOGICALLY LOGICAL relationships:
    - Calories > TDEE -> weight gain
    - Calories < TDEE -> weight loss
    - Daily_Weight_Change ≈ (caloric_surplus / 3500) lbs/day
    """
    np.random.seed(seed)
    random.seed(seed)

    df = df_orig.copy().reset_index(drop=True)

    # Use only rows that have key info
    df = df.dropna(subset=[
        "Age", "Gender", "Current Weight (lbs)", "BMR (Calories)",
        "Physical Activity Level", "Sleep Quality", "Stress Level"
    ])

    if df.empty:
        raise ValueError("No valid rows to base synthetic data on.")

    activity_values = df["Physical Activity Level"].dropna().unique().tolist()
    if not activity_values:
        activity_values = ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]

    sleep_values = df["Sleep Quality"].dropna().unique().tolist()
    if not sleep_values:
        sleep_values = ["Excellent", "Good", "Fair", "Poor"]

    gender_values = df["Gender"].dropna().unique().tolist()
    if not gender_values:
        gender_values = ["M", "F"]

    records = []

    for _ in range(n_samples):
        base = df.sample(1).iloc[0]

        # Age: around base, clipped
        age = int(np.clip(np.random.normal(base["Age"], 5), 18, 70))

        # Gender: keep base or sample
        gender = str(base["Gender"])
        if gender not in ["M", "F"]:
            gender = random.choice(["M", "F"])

        # Current weight (lbs)
        cw = float(np.clip(np.random.normal(base["Current Weight (lbs)"], 10), 90, 280))

        # Recompute BMR for consistency (height fixed ~170 cm)
        height_cm = 170.0
        if gender.upper() == "M":
            bmr = 10 * (cw * 0.453592) + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * (cw * 0.453592) + 6.25 * height_cm - 5 * age - 161

        # Activity & TDEE
        pal = random.choice(activity_values)
        mult = activity_multiplier(pal)
        tdee = bmr * mult

        # Scenario: loss / gain / maintenance-ish
        scenario = random.random()
        if scenario < 0.33:
            # weight loss scenario
            delta = -random.uniform(300, 900)  # deficit
        elif scenario < 0.66:
            # weight gain scenario
            delta = random.uniform(300, 900)   # surplus
        else:
            # near maintenance
            delta = random.uniform(-200, 200)

        daily_cals = np.clip(tdee + delta, 1200, 4000)
        # adjust delta to reflect final clipped cals
        delta = daily_cals - tdee

        # Daily weight change (lbs/day) ~ caloric surplus / 3500
        daily_w_change = delta / 3500.0  # positive: gain, negative: loss

        # Duration & total weight change
        duration_weeks = random.randint(4, 12)
        total_w_change = daily_w_change * 7 * duration_weeks  # lbs
        final_weight = cw + total_w_change

        # Sleep & stress
        sleep = random.choice(sleep_values)
        stress = int(np.clip(round(np.random.normal(6, 3)), 1, 12))

        rec = {
            "Age": age,
            "Gender": gender,
            "Current Weight (lbs)": round(cw, 1),
            "BMR (Calories)": round(bmr, 1),
            "Daily Calories Consumed": round(daily_cals, 1),
            "Daily Caloric Surplus/Deficit": round(delta, 1),
            "Weight Change (lbs)": round(total_w_change, 3),
            "Duration (weeks)": duration_weeks,
            "Physical Activity Level": pal,
            "Sleep Quality": sleep,
            "Stress Level": stress,
            "Final Weight (lbs)": round(final_weight, 1),
            "Daily_Weight_Change": daily_w_change,
        }
        records.append(rec)

    synthetic_df = pd.DataFrame(records)
    return synthetic_df


def main():
    print("Generating logically consistent synthetic dataset...")

    SYNTH_ROWS = 1000
    synthetic_df = generate_logical_synthetic(df_real, n_samples=SYNTH_ROWS, seed=42)

    synthetic_df.to_csv(SYNTHETIC_PATH, index=False)
    print(f"Synthetic data saved to: {SYNTHETIC_PATH} ({len(synthetic_df)} rows)")

    # Combine REAL + SYNTHETIC
    augmented_df = pd.concat(
        [df_real.reset_index(drop=True), synthetic_df.reset_index(drop=True)],
        ignore_index=True
    )
    augmented_df.to_csv(AUGMENTED_PATH, index=False)
    print(f"Augmented dataset saved to: {AUGMENTED_PATH} (total rows: {len(augmented_df)})")

    # ============== ENCODING ==============
    categorical_cols = ["Gender", "Physical Activity Level", "Sleep Quality"]
    encoders = {}

    for c in categorical_cols:
        if c in augmented_df.columns:
            le = LabelEncoder()
            augmented_df[c] = augmented_df[c].fillna("Unknown").astype(str)
            augmented_df[c] = le.fit_transform(augmented_df[c])
            encoders[c] = le

    TARGET = "Daily_Weight_Change"
    if TARGET not in augmented_df.columns:
        raise KeyError(
            f"Target column '{TARGET}' not found in dataset. "
            "Make sure your CSV has this column."
        )

    features = [
        "Age", "Gender", "Current Weight (lbs)", "BMR (Calories)",
        "Daily Calories Consumed", "Physical Activity Level",
        "Sleep Quality", "Stress Level"
    ]
    features_available = [f for f in features if f in augmented_df.columns]

    X = augmented_df[features_available]
    y = augmented_df[TARGET].astype(float)

    mask = X.notnull().all(axis=1) & y.notnull()
    X = X.loc[mask]
    y = y.loc[mask]

    print("Training GradientBoostingRegressor on augmented dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Training finished. R²: {r2:.4f}, MAE: {mae:.4f}")

    bundle = {
        "model": model,
        "encoders": encoders,
        "features": features_available,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Trained model and encoders saved to: {MODEL_PATH}")
    print("Done. Now you can run 'final_project.py' to use the GUI app.")


if __name__ == "__main__":
    main()
