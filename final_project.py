import os
import sys
import math
import joblib
import random
import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

warnings.filterwarnings("ignore")

# ================== SETTINGS ==================
MAINTENANCE_DIFF_THRESHOLD = 100  # kcal/day; within this => "on track"
MEALPLAN_TOLERANCE = 150  # kcal tolerance for meal-plan totals (template-based)

# ETA simulation settings (lbs/day model output assumed)
MAX_SIM_DAYS = 3650  # safety cap (~10 years)
EPS_CHANGE = 1e-6  # tiny-change threshold

# ================== CUSTOMTKINTER THEME ==================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ================== PATHS ==================
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FOOD_PATH = os.path.join(BASE_DIR, "calories.csv")
MODEL_PATH = os.path.join(BASE_DIR, "trained_weight_model.pkl")

if not os.path.exists(FOOD_PATH):
    raise SystemExit(f"Food dataset not found at: {FOOD_PATH}")

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model file not found at: {MODEL_PATH}\nRun 'train_model.py' first.")

# ================== LOAD DATA (tracker only) ==================
food_df = pd.read_csv(FOOD_PATH)
expected_cols = {"FoodCategory", "FoodItem", "per100grams", "Cals_per100grams", "KJ_per100grams"}
missing = expected_cols - set(food_df.columns)
if missing:
    raise SystemExit(f"Missing column(s) in calories.csv: {missing}")

food_df["Cals_per100grams"] = (
    food_df["Cals_per100grams"]
    .astype(str)
    .str.replace(" cal", "", regex=False)
    .astype(float)
)

# ================== LOAD MODEL ==================
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
encoders = bundle.get("encoders", {})
features = bundle.get("features", [])

# ================== USER PROFILE (set from Goal Page) ==================
user_profile = {
    "gender": "M",  # "M" or "F"
    "activity": "Moderately Active",
    "goal_dir": 0,
}

# ================== ENERGY HELPERS ==================
def compute_activity_multiplier(activity_level: str) -> float:
    level = (activity_level or "").lower()
    if "very" in level:
        return 1.725
    if "moderately" in level:
        return 1.55
    if "light" in level:
        return 1.375
    if "sedentary" in level:
        return 1.2
    return 1.3


def compute_safe_calorie_recommendation(
    current_weight_lbs: float,
    goal_weight_lbs: float,
    gender: str,
    avg_calories: float,
    bmr: float,
    activity_level: str,
    height_cm: float = 170.0,
):
    gender = (gender or "").upper()
    act_mult = compute_activity_multiplier(activity_level)
    maintenance = bmr * act_mult

    height_m = height_cm / 100.0
    bmi = (current_weight_lbs * 0.453592) / (height_m**2)

    is_loss_goal = goal_weight_lbs < current_weight_lbs
    is_gain_goal = goal_weight_lbs > current_weight_lbs

    clinical_min = 1500 if gender == "M" else 1200

    if is_loss_goal:
        deficit_pct = 0.20 if bmi >= 30 else (0.18 if bmi >= 25 else 0.15)
        raw_target = maintenance * (1 - deficit_pct)
        recommended = max(raw_target, clinical_min, bmr * 0.75)
    elif is_gain_goal:
        surplus_pct = 0.15 if bmi < 18.5 else 0.12
        raw_target = maintenance * (1 + surplus_pct)
        recommended = max(raw_target, bmr + 100, clinical_min)
    else:
        recommended = maintenance

    return {
        "maintenance": maintenance,
        "bmi": bmi,
        "recommended_calories": recommended,
        "change_vs_avg": recommended - avg_calories,
        "is_loss_goal": is_loss_goal,
        "is_gain_goal": is_gain_goal,
    }

# ================== MODEL-BASED CALORIE SEARCH ==================
def ml_calorie_search(model, base_features: dict, goal_dir: int, avg_calories: float, features_list):
    if "Daily Calories Consumed" not in features_list:
        return None, None

    min_kcal, max_kcal, step = 1200, 4000, 50
    candidates = []

    for cals in range(min_kcal, max_kcal + 1, step):
        row = dict(base_features)
        row["Daily Calories Consumed"] = float(cals)
        df_row = pd.DataFrame([row], columns=features_list)
        pred = model.predict(df_row)[0]

        if goal_dir == 1 and pred > 0:
            candidates.append((cals, pred))
        elif goal_dir == -1 and pred < 0:
            candidates.append((cals, pred))

    if not candidates:
        return None, None

    best_cals, best_pred = min(candidates, key=lambda cp: abs(cp[0] - avg_calories))
    return float(best_cals), float(best_pred)


def ml_find_maintenance_calories(model, base_features: dict, features_list, avg_calories: float):
    if "Daily Calories Consumed" not in features_list:
        return None, None

    best_cals = None
    best_pred = None
    best_score = None

    for cals in range(1200, 4001, 25):
        row = dict(base_features)
        row["Daily Calories Consumed"] = float(cals)
        df_row = pd.DataFrame([row], columns=features_list)
        pred = float(model.predict(df_row)[0])

        score = abs(pred)
        if best_score is None:
            best_score = score
            best_cals = float(cals)
            best_pred = pred
            continue

        if (score < best_score) or (
            abs(score - best_score) < 1e-12 and abs(cals - avg_calories) < abs(best_cals - avg_calories)
        ):
            best_score = score
            best_cals = float(cals)
            best_pred = pred

    return best_cals, best_pred

# ================== GOAL ETA (DAYS TO REACH GOAL) ==================
def _bmr_from_weight_lbs(weight_lbs: float, age: int, gender: str, height_cm: float = 170.0) -> float:
    wkg = float(weight_lbs) * 0.453592
    g = (gender or "M").upper()
    if g == "M":
        return 10 * wkg + 6.25 * height_cm - 5 * age + 5
    return 10 * wkg + 6.25 * height_cm - 5 * age - 161


def estimate_days_to_goal(
    model,
    base_features: dict,
    features_list: list,
    current_weight_lbs: float,
    goal_weight_lbs: float,
    daily_calories: float,
    age: int,
    gender: str,
    max_days: int = MAX_SIM_DAYS,
):
    """
    Assumes model predicts weight change in lbs/day.
    Simulates day-by-day until goal is reached (or fails).
    Returns dict: {"reached": bool, "days": int|None}
    """
    cw = float(current_weight_lbs)
    gw = float(goal_weight_lbs)

    if abs(gw - cw) < 1e-9:
        return {"reached": True, "days": 0}

    want_dir = 1 if gw > cw else -1
    w = cw

    for day in range(1, max_days + 1):
        row = dict(base_features)
        row["Current Weight (lbs)"] = float(w)
        row["BMR (Calories)"] = float(_bmr_from_weight_lbs(w, age, gender, height_cm=170.0))
        row["Daily Calories Consumed"] = float(daily_calories)

        df_row = pd.DataFrame([row], columns=features_list)
        change = float(model.predict(df_row)[0])  # lbs/day

        # stagnation or wrong direction => can't estimate reliably
        if abs(change) < EPS_CHANGE or (change * want_dir <= 0):
            return {"reached": False, "days": None}

        w_next = w + change
        if (want_dir == 1 and w_next >= gw) or (want_dir == -1 and w_next <= gw):
            return {"reached": True, "days": day}

        w = w_next

    return {"reached": False, "days": None}


def fmt_days(days):
    if days is None:
        return "N/A"
    if days < 14:
        return f"~{days:.0f} days"
    return f"~{days:.0f} days (~{days/7:.1f} weeks)"

# ============================================================
# =============== PROFESSIONAL MEAL PLAN ENGINE ===============
# ============================================================
ACTIVITY_LEVELS = ["Very Active", "Moderately Active", "Lightly Active", "Sedentary"]


def _is_workout_day(day_name: str, activity_level: str) -> bool:
    """
    Simple workout assumption:
      - Moderately/Very Active => workouts Mon/Wed/Fri
      - Lightly Active => workouts Tue/Thu
      - Sedentary => no workout emphasis
    """
    lvl = (activity_level or "").lower()
    if "very" in lvl or "moderately" in lvl:
        return day_name in ("Monday", "Wednesday", "Friday")
    if "light" in lvl:
        return day_name in ("Tuesday", "Thursday")
    return False


def macro_targets(calories: float, diet_type: str, activity_level: str, gender: str, workout_day: bool):
    """
    Returns gram targets for protein/carbs/fats + fiber
    Uses EXACT 4 activity levels + gender.
    """
    diet = (diet_type or "Healthy Balanced").lower()
    act = (activity_level or "Moderately Active").strip()
    gender = (gender or "M").upper()

    # --- base macro splits by diet ---
    if "keto" in diet:
        base = {"p": 0.30, "c": 0.06, "f": 0.64}
        carb_cap_g = 30
    elif "high protein" in diet:
        base = {"p": 0.36, "c": 0.32, "f": 0.32}
        carb_cap_g = None
    elif "vegan" in diet:
        base = {"p": 0.25, "c": 0.50, "f": 0.25}
        carb_cap_g = None
    else:
        # Healthy Balanced
        base = {"p": 0.26, "c": 0.45, "f": 0.29}
        carb_cap_g = None

    # --- activity adjustments (4 types) ---
    # More active => more carbs + slightly more protein
    # Sedentary => fewer carbs + more protein, fat slightly higher
    if act == "Very Active":
        if carb_cap_g is None:
            base["c"] = min(0.55, base["c"] + 0.07)
            base["p"] = min(0.34, base["p"] + 0.02)
            base["f"] = 1.0 - (base["p"] + base["c"])
        else:
            base["p"] = min(0.35, base["p"] + 0.02)
            base["f"] = 1.0 - (base["p"] + base["c"])

    elif act == "Moderately Active":
        if carb_cap_g is None:
            base["c"] = min(0.52, base["c"] + 0.04)
            base["p"] = min(0.33, base["p"] + 0.01)
            base["f"] = 1.0 - (base["p"] + base["c"])

    elif act == "Lightly Active":
        if carb_cap_g is None:
            base["c"] = min(0.49, base["c"] + 0.01)
            base["p"] = min(0.32, base["p"] + 0.01)
            base["f"] = 1.0 - (base["p"] + base["c"])

    elif act == "Sedentary":
        if carb_cap_g is None:
            base["c"] = max(0.30, base["c"] - 0.07)
            base["p"] = min(0.36, base["p"] + 0.03)
            base["f"] = 1.0 - (base["p"] + base["c"])
        else:
            base["p"] = min(0.36, base["p"] + 0.02)
            base["f"] = 1.0 - (base["p"] + base["c"])
    else:
        base["p"] = min(0.36, base["p"] + 0.02)
        base["f"] = 1.0 - (base["p"] + base["c"])

    # --- workout-day carbs emphasis (not keto) ---
    if workout_day and carb_cap_g is None:
        base["c"] = min(0.58, base["c"] + 0.06)
        base["f"] = max(0.20, base["f"] - 0.04)
        base["p"] = 1.0 - (base["c"] + base["f"])

    # --- gender adjustment ---
    if gender == "M":
        base["p"] = min(0.38, base["p"] + 0.01)
        if carb_cap_g is None:
            base["c"] = max(0.28, base["c"] - 0.005)
        base["f"] = 1.0 - (base["p"] + base["c"])

    # fiber targets
    base_fiber = 28 if gender == "F" else 25
    if "vegan" in diet:
        fiber_g = max(base_fiber + 6, 32)
    elif "keto" in diet:
        fiber_g = max(base_fiber - 2, 22)
    else:
        fiber_g = base_fiber + (2 if workout_day else 0)

    # convert to grams
    protein_g = (calories * base["p"]) / 4.0
    carbs_g = (calories * base["c"]) / 4.0
    fat_g = (calories * base["f"]) / 9.0

    # keto carb cap + reallocate to fat
    if carb_cap_g is not None:
        carbs_g = min(carbs_g, float(carb_cap_g))
        used = protein_g * 4 + carbs_g * 4
        fat_g = max(0.0, (calories - used) / 9.0)

    return {
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
        "fiber_g": float(fiber_g),
    }


def _scale_meal(meal: dict, target_kcal: float):
    """Scale template macros linearly to hit target kcal."""
    base_kcal = float(meal["kcal"])
    if base_kcal <= 0:
        base_kcal = 1.0
    factor = target_kcal / base_kcal

    return {
        "name": meal["name"],
        "kcal": base_kcal * factor,
        "p": float(meal["p"]) * factor,
        "c": float(meal["c"]) * factor,
        "f": float(meal["f"]) * factor,
        "fiber": float(meal.get("fiber", 0)) * factor,
        "note": meal.get("note", ""),
    }


def _pick(pool, prefer=None):
    """pool: list[dict]  prefer: callable(meal)->bool"""
    if not pool:
        return None
    if prefer:
        preferred = [m for m in pool if prefer(m)]
        if preferred:
            return random.choice(preferred)
    return random.choice(pool)

# ---- PROFESSIONAL TEMPLATE LIBRARY (not from CSV) ----
# Macros are approximate per “standard serving”.
MEAL_LIBRARY = {
    "Healthy Balanced": {
        "Breakfast": [
            {"name": "Greek yogurt bowl (berries + oats + chia)", "kcal": 420, "p": 32, "c": 50, "f": 12, "fiber": 10},
            {"name": "Eggs + whole-grain toast + spinach + fruit", "kcal": 460, "p": 28, "c": 45, "f": 20, "fiber": 8},
            {"name": "Oatmeal + banana + peanut butter", "kcal": 480, "p": 18, "c": 65, "f": 16, "fiber": 10},
        ],
        "Lunch": [
            {"name": "Chicken quinoa bowl (veg + olive oil)", "kcal": 620, "p": 45, "c": 60, "f": 20, "fiber": 12},
            {"name": "Tuna rice bowl (veg + avocado)", "kcal": 640, "p": 42, "c": 65, "f": 22, "fiber": 10},
            {"name": "Turkey wrap (whole grain) + salad + fruit", "kcal": 600, "p": 40, "c": 65, "f": 18, "fiber": 12},
        ],
        "Dinner": [
            {"name": "Salmon + sweet potato + broccoli", "kcal": 680, "p": 42, "c": 55, "f": 28, "fiber": 12},
            {"name": "Lean beef stir-fry + rice + veggies", "kcal": 700, "p": 45, "c": 70, "f": 22, "fiber": 10},
            {"name": "Chicken pasta (whole-grain) + side salad", "kcal": 720, "p": 45, "c": 85, "f": 18, "fiber": 12},
        ],
        "SnackCore": [
            {"name": "Apple + nuts", "kcal": 220, "p": 6, "c": 22, "f": 14, "fiber": 5},
            {"name": "Protein shake + banana", "kcal": 260, "p": 28, "c": 30, "f": 3, "fiber": 3},
            {"name": "Cottage cheese + berries", "kcal": 220, "p": 24, "c": 18, "f": 4, "fiber": 4},
            {"name": "Hummus + carrots + whole-grain crackers", "kcal": 280, "p": 10, "c": 35, "f": 12, "fiber": 8},
        ],
        "Sweet": [
            {"name": "Dark chocolate (small portion) + strawberries", "kcal": 200, "p": 3, "c": 20, "f": 12, "fiber": 5},
            {"name": "Greek yogurt dessert (honey + cinnamon)", "kcal": 220, "p": 16, "c": 28, "f": 4, "fiber": 0},
        ],
        "Alcohol": [
            {"name": "1 glass dry wine", "kcal": 120, "p": 0, "c": 4, "f": 0, "fiber": 0},
            {"name": "1 light beer", "kcal": 110, "p": 1, "c": 10, "f": 0, "fiber": 0},
        ],
    },
    "High Protein": {
        "Breakfast": [
            {"name": "Egg-white omelet (veg) + toast + fruit", "kcal": 420, "p": 40, "c": 40, "f": 10, "fiber": 7},
            {"name": "Protein oatmeal (whey + oats + berries)", "kcal": 470, "p": 38, "c": 55, "f": 12, "fiber": 10},
            {"name": "Skyr + granola + berries", "kcal": 430, "p": 35, "c": 45, "f": 10, "fiber": 7},
        ],
        "Lunch": [
            {"name": "Chicken + brown rice + vegetables", "kcal": 650, "p": 55, "c": 65, "f": 15, "fiber": 10},
            {"name": "Lean turkey chili + side salad", "kcal": 620, "p": 50, "c": 55, "f": 18, "fiber": 14},
            {"name": "Salmon salad bowl + quinoa", "kcal": 660, "p": 48, "c": 55, "f": 24, "fiber": 10},
        ],
        "Dinner": [
            {"name": "Steak + roasted potatoes + asparagus", "kcal": 720, "p": 55, "c": 60, "f": 28, "fiber": 8},
            {"name": "Chicken stir-fry + rice + veggies", "kcal": 700, "p": 55, "c": 70, "f": 18, "fiber": 10},
            {"name": "White fish + couscous + vegetables", "kcal": 660, "p": 50, "c": 70, "f": 14, "fiber": 10},
        ],
        "SnackCore": [
            {"name": "Protein shake (whey) + fruit", "kcal": 280, "p": 32, "c": 30, "f": 4, "fiber": 4},
            {"name": "Cottage cheese + pineapple", "kcal": 250, "p": 26, "c": 28, "f": 4, "fiber": 2},
            {"name": "Beef jerky + apple", "kcal": 260, "p": 22, "c": 28, "f": 6, "fiber": 5},
        ],
        "Sweet": [
            {"name": "Protein pudding (high protein dessert)", "kcal": 200, "p": 20, "c": 15, "f": 6, "fiber": 2},
        ],
        "Alcohol": [
            {"name": "1 glass dry wine", "kcal": 120, "p": 0, "c": 4, "f": 0, "fiber": 0},
            {"name": "1 light beer", "kcal": 110, "p": 1, "c": 10, "f": 0, "fiber": 0},
        ],
    },
    "Keto": {
        "Breakfast": [
            {"name": "Eggs + avocado + sautéed spinach", "kcal": 520, "p": 24, "c": 10, "f": 40, "fiber": 10},
            {"name": "Keto yogurt bowl (unsweetened) + nuts + berries", "kcal": 480, "p": 20, "c": 12, "f": 38, "fiber": 8},
            {"name": "Omelet (cheese + mushrooms + peppers)", "kcal": 540, "p": 28, "c": 12, "f": 40, "fiber": 6},
        ],
        "Lunch": [
            {"name": "Chicken salad (olive oil) + almonds", "kcal": 650, "p": 45, "c": 12, "f": 45, "fiber": 8},
            {"name": "Salmon + green salad + avocado", "kcal": 680, "p": 42, "c": 10, "f": 50, "fiber": 10},
            {"name": "Bunless burger + cheese + side salad", "kcal": 700, "p": 45, "c": 10, "f": 52, "fiber": 6},
        ],
        "Dinner": [
            {"name": "Steak + roasted zucchini + butter", "kcal": 720, "p": 48, "c": 10, "f": 55, "fiber": 6},
            {"name": "Chicken thighs + broccoli + olive oil", "kcal": 700, "p": 45, "c": 12, "f": 50, "fiber": 8},
            {"name": "Shrimp + cauliflower rice + salad", "kcal": 640, "p": 45, "c": 14, "f": 40, "fiber": 10},
        ],
        "SnackCore": [
            {"name": "Cheese + walnuts", "kcal": 280, "p": 10, "c": 4, "f": 26, "fiber": 2},
            {"name": "Celery + peanut butter", "kcal": 250, "p": 8, "c": 10, "f": 20, "fiber": 5},
            {"name": "Hard-boiled eggs + olives", "kcal": 220, "p": 14, "c": 3, "f": 16, "fiber": 1},
        ],
        "Sweet": [
            {"name": "Keto dessert (chia pudding, no sugar)", "kcal": 220, "p": 8, "c": 10, "f": 16, "fiber": 10},
        ],
        "Alcohol": [
            {"name": "1 glass dry wine", "kcal": 120, "p": 0, "c": 4, "f": 0, "fiber": 0},
            {"name": "1 shot spirits + soda water", "kcal": 100, "p": 0, "c": 0, "f": 0, "fiber": 0},
        ],
    },
    "Vegan": {
        "Breakfast": [
            {"name": "Tofu scramble + whole-grain toast + fruit", "kcal": 480, "p": 28, "c": 55, "f": 16, "fiber": 12},
            {"name": "Overnight oats + chia + berries", "kcal": 500, "p": 18, "c": 75, "f": 14, "fiber": 16},
            {"name": "Smoothie (soy milk + banana + peanut butter)", "kcal": 520, "p": 20, "c": 60, "f": 22, "fiber": 10},
        ],
        "Lunch": [
            {"name": "Lentil bowl + brown rice + veggies", "kcal": 650, "p": 28, "c": 95, "f": 14, "fiber": 22},
            {"name": "Chickpea salad wrap + fruit", "kcal": 620, "p": 22, "c": 85, "f": 18, "fiber": 18},
            {"name": "Tofu quinoa bowl + mixed veggies", "kcal": 660, "p": 28, "c": 80, "f": 20, "fiber": 14},
        ],
        "Dinner": [
            {"name": "Bean chili + avocado + side salad", "kcal": 700, "p": 26, "c": 95, "f": 24, "fiber": 24},
            {"name": "Tempeh stir-fry + rice + veggies", "kcal": 720, "p": 30, "c": 95, "f": 22, "fiber": 16},
            {"name": "Tofu curry + rice + vegetables", "kcal": 740, "p": 28, "c": 100, "f": 22, "fiber": 14},
        ],
        "SnackCore": [
            {"name": "Hummus + carrots + crackers", "kcal": 300, "p": 10, "c": 40, "f": 12, "fiber": 10},
            {"name": "Edamame + fruit", "kcal": 280, "p": 16, "c": 30, "f": 10, "fiber": 8},
            {"name": "Trail mix (nuts + dried fruit)", "kcal": 320, "p": 8, "c": 35, "f": 18, "fiber": 6},
        ],
        "Sweet": [
            {"name": "Vegan dessert (dark chocolate + berries)", "kcal": 220, "p": 3, "c": 24, "f": 14, "fiber": 6},
        ],
        "Alcohol": [
            {"name": "1 glass dry wine", "kcal": 120, "p": 0, "c": 4, "f": 0, "fiber": 0},
            {"name": "1 light beer", "kcal": 110, "p": 1, "c": 10, "f": 0, "fiber": 0},
        ],
    },
}


def build_professional_day_plan(
    day_name: str,
    target_kcal: float,
    diet_type: str,
    gender: str,
    activity_level: str,
    allow_sweets: bool,
    allow_alcohol: bool,
):
    """
    Builds a day plan with professional meal structure + macro/fiber focus.
    Returns (meals, totals_dict, targets_dict, flags_dict)
    """
    workout_day = _is_workout_day(day_name, activity_level)
    is_weekend = day_name in ("Friday", "Saturday", "Sunday")
    alcohol_today = allow_alcohol and is_weekend

    # macro/fiber targets (activity + gender + workout day)
    targets = macro_targets(
        calories=float(target_kcal),
        diet_type=diet_type,
        activity_level=activity_level,
        gender=gender,
        workout_day=workout_day,
    )

    # meal calorie distribution
    if workout_day:
        shares = {"Breakfast": 0.24, "Lunch": 0.38, "Dinner": 0.28, "Snacks": 0.10}
    else:
        shares = {"Breakfast": 0.26, "Lunch": 0.35, "Dinner": 0.29, "Snacks": 0.10}

    lib = MEAL_LIBRARY.get(diet_type, MEAL_LIBRARY["Healthy Balanced"])

    meals_out = []
    totals = {"kcal": 0.0, "p": 0.0, "c": 0.0, "f": 0.0, "fiber": 0.0}

    def add_scaled(meal_name, template_meal, kcal_target):
        nonlocal totals
        scaled = _scale_meal(template_meal, kcal_target)
        meals_out.append((meal_name, scaled))
        totals["kcal"] += scaled["kcal"]
        totals["p"] += scaled["p"]
        totals["c"] += scaled["c"]
        totals["f"] += scaled["f"]
        totals["fiber"] += scaled["fiber"]

    # core meals
    for meal_name in ("Breakfast", "Lunch", "Dinner"):
        pool = lib.get(meal_name, [])
        kcal_target = float(target_kcal) * shares[meal_name]

        if diet_type != "Keto":
            if activity_level in ("Very Active", "Moderately Active") and (
                meal_name in ("Lunch", "Dinner") or workout_day
            ):
                chosen = _pick(pool, prefer=lambda m: m["c"] >= 60)
            elif activity_level == "Sedentary":
                chosen = _pick(pool, prefer=lambda m: m["c"] <= 65)
            else:
                chosen = _pick(pool)
        else:
            chosen = _pick(pool)

        if not chosen:
            chosen = {"name": "Simple balanced plate", "kcal": 600, "p": 30, "c": 60, "f": 20, "fiber": 8}

        add_scaled(meal_name, chosen, kcal_target)

    # snacks: base snack + optional sweet + optional alcohol (weekends)
    snack_kcal_total = float(target_kcal) * shares["Snacks"]
    snack_pool = lib.get("SnackCore", [])
    sweet_pool = lib.get("Sweet", [])
    alcohol_pool = lib.get("Alcohol", [])

    sweet_kcal = 0.0
    alcohol_kcal = 0.0

    if allow_sweets and sweet_pool:
        sweet_kcal = min(220.0, snack_kcal_total * 0.55)

    if alcohol_today and alcohol_pool:
        alcohol_kcal = min(140.0, snack_kcal_total * 0.6)

    core_snack_kcal = max(80.0, snack_kcal_total - sweet_kcal - alcohol_kcal)

    if diet_type != "Keto":
        if activity_level == "Sedentary":
            core_snack = _pick(snack_pool, prefer=lambda m: m["p"] >= 18)
        elif workout_day:
            core_snack = _pick(snack_pool, prefer=lambda m: m["c"] >= 25)
        else:
            core_snack = _pick(snack_pool)
    else:
        core_snack = _pick(snack_pool)

    if core_snack:
        add_scaled("Snack", core_snack, core_snack_kcal)
    if sweet_kcal > 0 and sweet_pool:
        add_scaled("Sweet", _pick(sweet_pool), sweet_kcal)
    if alcohol_kcal > 0 and alcohol_pool:
        add_scaled("Alcohol", _pick(alcohol_pool), alcohol_kcal)

    flags = {
        "workout_day": workout_day,
        "alcohol_today": alcohol_today,
    }

    return meals_out, totals, targets, flags

# ================== APP + PAGES ==================
app = ctk.CTk()
app.title("🍎 Weekly Food Calorie Tracker")

try:
    app.state("zoomed")
except Exception:
    app.geometry("1150x800")

tracker_page = ctk.CTkFrame(app)
goal_page = ctk.CTkFrame(app)
mealplan_page = ctk.CTkFrame(app)

current_page = "tracker"
mealplan_target_calories = None
page_stack = []


def _switch_page(target: str, push_history: bool = True):
    global current_page

    if push_history and current_page != target:
        page_stack.append(current_page)

    tracker_page.pack_forget()
    goal_page.pack_forget()
    mealplan_page.pack_forget()

    if target == "tracker":
        tracker_page.pack(fill="both", expand=True)
    elif target == "goal":
        goal_page.pack(fill="both", expand=True)
    elif target == "mealplan":
        mealplan_page.pack(fill="both", expand=True)
    else:
        tracker_page.pack(fill="both", expand=True)
        target = "tracker"

    current_page = target


def show_tracker_page(push_history: bool = True):
    _switch_page("tracker", push_history=push_history)


def show_goal_page(push_history: bool = True):
    _switch_page("goal", push_history=push_history)


def show_mealplan_page(push_history: bool = True):
    _switch_page("mealplan", push_history=push_history)


def go_back():
    if not page_stack:
        show_tracker_page(push_history=False)
        return
    prev = page_stack.pop()
    _switch_page(prev, push_history=False)

# ================== TRACKER PAGE ==================
title_label = ctk.CTkLabel(tracker_page, text="Weekly Food Calorie Tracker", font=("Segoe UI", 26, "bold"))
title_label.pack(pady=15)

top_frame = ctk.CTkFrame(tracker_page, corner_radius=15)
top_frame.pack(fill="x", padx=20, pady=5)

bottom_frame = ctk.CTkFrame(tracker_page, corner_radius=15)
bottom_frame.pack(fill="both", expand=True, padx=20, pady=10)

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_var = ctk.StringVar(value=days[0])

day_meals = {d: [] for d in days}
daily_totals = {d: 0.0 for d in days}

ctk.CTkLabel(top_frame, text="Select Day:", font=("Segoe UI", 14, "bold")).grid(
    row=0, column=0, padx=10, pady=10, sticky="w"
)
day_menu = ctk.CTkComboBox(top_frame, variable=day_var, values=days, width=150, font=("Segoe UI", 13))
day_menu.grid(row=0, column=1, padx=5, pady=10)

daily_total_var = ctk.StringVar(value="0.0")
average_label_var = ctk.StringVar(value="Average Calorie Intake per Day: -")

ctk.CTkLabel(top_frame, text="Daily Total:", font=("Segoe UI", 14, "bold")).grid(
    row=0, column=2, padx=20, pady=10, sticky="e"
)
daily_total_label = ctk.CTkLabel(top_frame, textvariable=daily_total_var, font=("Segoe UI", 14))
daily_total_label.grid(row=0, column=3, padx=5, pady=10, sticky="w")

avg_label = ctk.CTkLabel(top_frame, textvariable=average_label_var, font=("Segoe UI", 14))
avg_label.grid(row=0, column=4, padx=20, pady=10, sticky="w")

meal_frame = ctk.CTkFrame(bottom_frame, corner_radius=15)
meal_frame.pack(fill="x", padx=15, pady=10)

ctk.CTkLabel(meal_frame, text="Add Meal", font=("Segoe UI", 18, "bold")).grid(
    row=0, column=0, columnspan=4, padx=10, pady=(10, 5), sticky="w"
)

ctk.CTkLabel(meal_frame, text="Food Category:", font=("Segoe UI", 13)).grid(
    row=1, column=0, sticky="e", padx=5, pady=5
)
category_var = ctk.StringVar()
category_combo = ctk.CTkComboBox(
    meal_frame,
    variable=category_var,
    values=sorted(food_df["FoodCategory"].unique()),
    width=220,
    font=("Segoe UI", 13),
)
category_combo.grid(row=1, column=1, padx=5, pady=5)

ctk.CTkLabel(meal_frame, text="Food Item:", font=("Segoe UI", 13)).grid(
    row=1, column=2, sticky="e", padx=5, pady=5
)
food_var = ctk.StringVar()
food_combo = ctk.CTkComboBox(meal_frame, variable=food_var, values=[], width=220, font=("Segoe UI", 13))
food_combo.grid(row=1, column=3, padx=5, pady=5)

ctk.CTkLabel(meal_frame, text="Weight (grams/ml):", font=("Segoe UI", 13)).grid(
    row=2, column=0, sticky="e", padx=5, pady=5
)
weight_entry = ctk.CTkEntry(meal_frame, width=220, placeholder_text="e.g., 150", font=("Segoe UI", 13))
weight_entry.grid(row=2, column=1, padx=5, pady=5)


def refresh_day_table():
    meal_table.delete(*meal_table.get_children())
    d = day_var.get()
    for meal in day_meals[d]:
        meal_table.insert("", "end", values=meal)
    daily_total_var.set(f"{daily_totals[d]:.2f}")


def update_food_items(*_args):
    cat = category_var.get()
    df_cat = food_df[food_df["FoodCategory"] == cat]
    items = sorted(df_cat["FoodItem"].unique().tolist())
    if items:
        food_combo.configure(values=items)
        food_var.set(items[0])
    else:
        food_combo.configure(values=["No items"])
        food_var.set("No items")


def add_meal():
    cat = category_var.get()
    food = food_var.get()
    weight = weight_entry.get()

    if not cat or not food or not weight:
        messagebox.showwarning("Missing Data", "Please fill all fields before adding a meal.")
        return

    try:
        weight_val = float(weight)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a numeric weight value.")
        return

    row = food_df[(food_df["FoodCategory"] == cat) & (food_df["FoodItem"] == food)]
    if row.empty:
        meal = (food, f"{weight_val:.1f}", "0.00")
        cals = 0.0
    else:
        cals_per_100g = row["Cals_per100grams"].values[0]
        cals = (cals_per_100g * weight_val) / 100.0
        meal = (food, f"{weight_val:.1f}", f"{cals:.2f}")

    d = day_var.get()
    day_meals[d].append(meal)
    daily_totals[d] += cals

    refresh_day_table()
    weight_entry.delete(0, "end")


def clear_selected_meal():
    sel = meal_table.selection()
    if not sel:
        messagebox.showwarning("No Selection", "Please select a meal to delete.")
        return

    d = day_var.get()
    for i in sel:
        vals = meal_table.item(i, "values")
        try:
            daily_totals[d] -= float(vals[2])
        except Exception:
            pass

        day_meals[d] = [m for m in day_meals[d] if m != vals]
        meal_table.delete(i)

    daily_total_var.set(f"{daily_totals[d]:.2f}")


def show_weekly_calorie_chart():
    total_week = sum(daily_totals.values())
    if total_week == 0:
        messagebox.showinfo("Info", "No meals recorded yet.")
        return

    chart_win = ctk.CTkToplevel(app)
    chart_win.title("Weekly Calorie Intake Chart")
    chart_win.geometry("900x500")
    chart_win.transient(app)
    chart_win.grab_set()
    chart_win.focus()

    fig = Figure(figsize=(8, 4), dpi=100)
    ax = fig.add_subplot(111)

    days_list = list(daily_totals.keys())
    calories_list = [daily_totals[d] for d in days_list]

    ax.bar(days_list, calories_list)
    ax.set_title("Weekly Calorie Intake", fontsize=16)
    ax.set_xlabel("Day of Week", fontsize=12)
    ax.set_ylabel("Calories (kcal)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    canvas = FigureCanvasTkAgg(fig, master=chart_win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    ctk.CTkButton(chart_win, text="Close", command=chart_win.destroy).pack(pady=10)


def finish_week():
    total_week = sum(daily_totals.values())
    if total_week == 0:
        messagebox.showinfo("Info", "No meals recorded yet.")
        return

    avg_day = round(total_week / 7, 2)
    average_label_var.set(f"Average Calorie Intake per Day: {avg_day} cal")

    messagebox.showinfo(
        "Week Finished",
        f"Weekly Total Calories: {total_week:.2f}\n"
        f"Average Calories Per Day: {avg_day:.2f}\n\n"
        "Showing your weekly calorie chart.",
    )
    show_weekly_calorie_chart()


def reset_app():
    for d in days:
        day_meals[d] = []
        daily_totals[d] = 0.0
    day_var.set(days[0])
    refresh_day_table()
    average_label_var.set("Average Calorie Intake per Day: -")


def on_day_change(_choice):
    refresh_day_table()


day_menu.configure(command=on_day_change)
category_combo.configure(command=update_food_items)

table_frame = ctk.CTkFrame(bottom_frame, corner_radius=15)
table_frame.pack(fill="both", expand=True, padx=15, pady=10)

style = ttk.Style()
style.configure("Treeview", font=("Segoe UI", 13))
style.configure("Treeview.Heading", font=("Segoe UI", 14, "bold"))

columns = ("Food", "Weight (g)", "Calories")
meal_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=10, style="Treeview")
for c in columns:
    meal_table.heading(c, text=c)
    meal_table.column(c, anchor="center", width=260)

meal_table.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)

scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=meal_table.yview)
meal_table.configure(yscroll=scrollbar.set)
scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=10)

buttons_frame = ctk.CTkFrame(bottom_frame, corner_radius=15)
buttons_frame.pack(fill="x", padx=15, pady=5)

ctk.CTkButton(
    buttons_frame,
    text="➕ Add Meal",
    width=150,
    font=("Segoe UI", 13, "bold"),
    command=add_meal,
).grid(row=0, column=0, padx=10, pady=10)

ctk.CTkButton(
    buttons_frame,
    text="🗑️ Clear Selected Meal",
    fg_color="#b71c1c",
    hover_color="#d32f2f",
    width=190,
    font=("Segoe UI", 13, "bold"),
    command=clear_selected_meal,
).grid(row=0, column=1, padx=10, pady=10)

ctk.CTkButton(
    buttons_frame,
    text="✅ Finish Week & Show Chart",
    fg_color="#2e7d32",
    hover_color="#388e3c",
    width=230,
    font=("Segoe UI", 13, "bold"),
    command=finish_week,
).grid(row=0, column=2, padx=10, pady=10)

ctk.CTkButton(
    buttons_frame,
    text="🎯 Goal Weight Prediction",
    fg_color="#f9a825",
    hover_color="#fbc02d",
    text_color="black",
    width=230,
    font=("Segoe UI", 13, "bold"),
    command=lambda: show_goal_page(),
).grid(row=0, column=3, padx=10, pady=10)

# ================== GOAL PAGE ==================
goal_title = ctk.CTkLabel(goal_page, text="Goal Weight Prediction", font=("Segoe UI", 26, "bold"))
goal_title.pack(pady=15)

goal_form_frame = ctk.CTkFrame(goal_page, corner_radius=15)
goal_form_frame.pack(fill="x", padx=20, pady=10)

ctk.CTkLabel(goal_form_frame, text="Age:", font=("Segoe UI", 13)).grid(
    row=0, column=0, padx=10, pady=8, sticky="w"
)
age_entry = ctk.CTkEntry(goal_form_frame, width=180, placeholder_text="e.g., 30", font=("Segoe UI", 13))
age_entry.grid(row=0, column=1, padx=10, pady=8)

ctk.CTkLabel(goal_form_frame, text="Gender:", font=("Segoe UI", 13)).grid(
    row=1, column=0, padx=10, pady=8, sticky="w"
)
gender_var = ctk.StringVar()
gender_combo = ctk.CTkComboBox(goal_form_frame, variable=gender_var, values=["M", "F"], width=180, font=("Segoe UI", 13))
gender_combo.grid(row=1, column=1, padx=10, pady=8)

ctk.CTkLabel(goal_form_frame, text="Current Weight (lbs):", font=("Segoe UI", 13)).grid(
    row=2, column=0, padx=10, pady=8, sticky="w"
)
current_weight_entry = ctk.CTkEntry(goal_form_frame, width=180, placeholder_text="e.g., 180", font=("Segoe UI", 13))
current_weight_entry.grid(row=2, column=1, padx=10, pady=8)

ctk.CTkLabel(goal_form_frame, text="Goal Weight (lbs):", font=("Segoe UI", 13)).grid(
    row=3, column=0, padx=10, pady=8, sticky="w"
)
goal_weight_entry = ctk.CTkEntry(goal_form_frame, width=180, placeholder_text="e.g., 165", font=("Segoe UI", 13))
goal_weight_entry.grid(row=3, column=1, padx=10, pady=8)

ctk.CTkLabel(goal_form_frame, text="Physical Activity Level:", font=("Segoe UI", 13)).grid(
    row=4, column=0, padx=10, pady=8, sticky="w"
)
activity_var = ctk.StringVar()
activity_combo = ctk.CTkComboBox(
    goal_form_frame,
    variable=activity_var,
    values=["Very Active", "Moderately Active", "Lightly Active", "Sedentary"],
    width=180,
    font=("Segoe UI", 13),
)
activity_combo.set("Moderately Active")
activity_combo.grid(row=4, column=1, padx=10, pady=8)

ctk.CTkLabel(goal_form_frame, text="Sleep Quality:", font=("Segoe UI", 13)).grid(
    row=5, column=0, padx=10, pady=8, sticky="w"
)
sleep_var = ctk.StringVar()
sleep_combo = ctk.CTkComboBox(
    goal_form_frame,
    variable=sleep_var,
    values=["Excellent", "Good", "Fair", "Poor"],
    width=180,
    font=("Segoe UI", 13),
)
sleep_combo.set("Good")
sleep_combo.grid(row=5, column=1, padx=10, pady=8)

ctk.CTkLabel(goal_form_frame, text="Stress Level (1–12):", font=("Segoe UI", 13)).grid(
    row=6, column=0, padx=10, pady=8, sticky="w"
)
stress_var = ctk.StringVar()
stress_combo = ctk.CTkComboBox(
    goal_form_frame,
    variable=stress_var,
    values=[str(i) for i in range(1, 13)],
    width=180,
    font=("Segoe UI", 13),
)
stress_combo.set("4")
stress_combo.grid(row=6, column=1, padx=10, pady=8)

avg_from_week_label = ctk.CTkLabel(goal_page, text="Average daily calories from your week: -", font=("Segoe UI", 13))
avg_from_week_label.pack(pady=8)

# Result sentence (no CTkToplevel popups for goal status)
result_var = ctk.StringVar(value="")
result_label = ctk.CTkLabel(goal_page, textvariable=result_var, font=("Segoe UI", 13), justify="left", wraplength=900)
result_label.pack(pady=(0, 10))


def get_weekly_average():
    total_week = sum(daily_totals.values())
    if total_week == 0:
        return 0.0
    return total_week / 7.0


def predict_and_suggest():
    global mealplan_target_calories, user_profile

    result_var.set("")
    avg_calories = get_weekly_average()

    if avg_calories == 0:
        messagebox.showwarning("No Data", "Please record meals first before using the predictor.")
        return

    avg_from_week_label.configure(text=f"Average daily calories from your week: {avg_calories:.2f} kcal")

    try:
        age = int(age_entry.get())
        gender = gender_var.get()
        current_weight = float(current_weight_entry.get())
        goal_weight = float(goal_weight_entry.get())
        activity = activity_var.get()
        sleep = sleep_var.get()
        stress = int(stress_var.get())
    except Exception:
        messagebox.showerror("Error", "Please fill all fields correctly.")
        return

    # store for meal plan
    user_profile["gender"] = gender
    user_profile["activity"] = activity

    if avg_calories < 600 or avg_calories > 5000:
        messagebox.showwarning("Unrealistic Intake", "Your intake looks unrealistic. Week will be cleared.")
        reset_app()
        show_tracker_page(push_history=False)
        return

    # simple BMR estimate (current)
    if gender == "M":
        bmr = 10 * (current_weight * 0.453592) + 6.25 * 170 - 5 * age + 5
    else:
        bmr = 10 * (current_weight * 0.453592) + 6.25 * 170 - 5 * age - 161

    if abs(goal_weight - current_weight) < 1e-6:
        goal_dir = 0
    elif goal_weight > current_weight:
        goal_dir = 1
    else:
        goal_dir = -1

    user_profile["goal_dir"] = goal_dir

    base_row = {f: 0.0 for f in features}
    base_row["Age"] = age

    if "Gender" in encoders:
        try:
            base_row["Gender"] = encoders["Gender"].transform([gender])[0]
        except Exception:
            base_row["Gender"] = 0
    else:
        base_row["Gender"] = 1 if gender == "M" else 0

    base_row["Current Weight (lbs)"] = current_weight
    base_row["BMR (Calories)"] = bmr
    base_row["Daily Calories Consumed"] = avg_calories

    if "Physical Activity Level" in encoders and "Physical Activity Level" in features:
        try:
            base_row["Physical Activity Level"] = encoders["Physical Activity Level"].transform([activity])[0]
        except Exception:
            base_row["Physical Activity Level"] = 0

    if "Sleep Quality" in encoders and "Sleep Quality" in features:
        try:
            base_row["Sleep Quality"] = encoders["Sleep Quality"].transform([sleep])[0]
        except Exception:
            base_row["Sleep Quality"] = 0

    if "Stress Level" in features:
        base_row["Stress Level"] = stress

    df_input = pd.DataFrame([base_row], columns=features)
    pred_change = float(model.predict(df_input)[0])

    # ETA using current avg calories (if goal is loss/gain)
    eta_avg = None
    if goal_dir != 0:
        eta_avg = estimate_days_to_goal(
            model=model,
            base_features=base_row,
            features_list=features,
            current_weight_lbs=current_weight,
            goal_weight_lbs=goal_weight,
            daily_calories=avg_calories,
            age=age,
            gender=gender,
        )

    # MAINTENANCE (no popups)
    if goal_dir == 0:
        maint_cals, _ = ml_find_maintenance_calories(model, base_row, features, avg_calories)
        if maint_cals is None:
            maint_cals = float(bmr * compute_activity_multiplier(activity))
        maint_cals = max(1200.0, float(maint_cals))
        mealplan_target_calories = float(maint_cals)

        result_var.set(f"Goal: MAINTAIN ✅ | Estimated maintenance: {maint_cals:.0f} kcal/day.")
        return

    # OFFTRACK suggestion / or on-track
    ml_dir = 0 if abs(pred_change) < 1e-6 else (1 if pred_change > 0 else -1)

    # ON TRACK (no popups)
    if ml_dir == goal_dir and ml_dir != 0:
        mealplan_target_calories = float(avg_calories)
        days_txt = fmt_days(eta_avg["days"] if eta_avg else None)
        result_var.set(
            f"Status: ON TRACK ✅ | Current avg: {avg_calories:.0f} kcal/day | "
            f"Estimated days to reach {goal_weight:.0f} lbs: {days_txt}."
        )
        return

    # otherwise suggest calorie target
    best_cals, _ = ml_calorie_search(model, base_row, goal_dir, avg_calories, features)
    if best_cals is None:
        rec = compute_safe_calorie_recommendation(current_weight, goal_weight, gender, avg_calories, bmr, activity)
        best_cals = float(rec["recommended_calories"])

    # keep your ±300 logic
    plan_target = best_cals - 300 if goal_dir == -1 else best_cals + 300
    plan_target = max(1200.0, float(plan_target))
    mealplan_target_calories = plan_target

    eta_plan = estimate_days_to_goal(
        model=model,
        base_features=base_row,
        features_list=features,
        current_weight_lbs=current_weight,
        goal_weight_lbs=goal_weight,
        daily_calories=plan_target,
        age=age,
        gender=gender,
    )

    days_txt = fmt_days(eta_plan["days"] if eta_plan else None)
    result_var.set(
        f"Status: OFF TRACK ❌ | Current avg: {avg_calories:.0f} kcal/day | "
        f"Suggested target: {best_cals:.0f} kcal/day | Meal plan target (±300): {plan_target:.0f} kcal/day | "
        f"Estimated days to reach {goal_weight:.0f} lbs: {days_txt}."
    )
    return


def build_and_show_mealplan():
    global mealplan_target_calories

    if mealplan_target_calories is None or mealplan_target_calories <= 0:
        messagebox.showerror("No Target", "Please click Predict / Suggest first to set a calorie target.")
        return

    # preload
    target_kcal_var.set(f"{mealplan_target_calories:.0f}")
    mealplan_subtitle_var.set(f"Target: ~{mealplan_target_calories:.0f} kcal/day")

    generate_week_plan(mealplan_target_calories)
    show_mealplan_page()

# Buttons on Goal page
ctk.CTkButton(
    goal_page,
    text="Predict / Suggest",
    font=("Segoe UI", 15, "bold"),
    width=220,
    height=40,
    command=predict_and_suggest,
).pack(pady=(10, 8))

ctk.CTkButton(
    goal_page,
    text="Open Meal Plan",
    font=("Segoe UI", 15, "bold"),
    width=220,
    height=40,
    command=build_and_show_mealplan,
).pack(pady=(0, 14))

ctk.CTkButton(
    goal_page,
    text="⬅ Back to Tracker",
    fg_color="#444444",
    hover_color="#555555",
    width=180,
    height=35,
    command=lambda: show_tracker_page(push_history=False),
).pack(pady=10)

# ================== MEAL PLAN PAGE ==================
mealplan_title = ctk.CTkLabel(mealplan_page, text="Weekly Meal Plan", font=("Segoe UI", 26, "bold"))
mealplan_title.pack(pady=10)

mealplan_subtitle_var = ctk.StringVar(value="Target: - kcal/day")
mealplan_subtitle = ctk.CTkLabel(mealplan_page, textvariable=mealplan_subtitle_var, font=("Segoe UI", 14))
mealplan_subtitle.pack(pady=(0, 10))

controls_frame = ctk.CTkFrame(mealplan_page, corner_radius=12)
controls_frame.pack(fill="x", padx=20, pady=10)

ctk.CTkLabel(controls_frame, text="Diet Type:", font=("Segoe UI", 14, "bold")).grid(
    row=0, column=0, padx=10, pady=10, sticky="w"
)
diet_var = ctk.StringVar(value="Healthy Balanced")
diet_combo = ctk.CTkComboBox(
    controls_frame,
    values=["Healthy Balanced", "High Protein", "Keto", "Vegan"],
    variable=diet_var,
    width=180,
    font=("Segoe UI", 13),
)
diet_combo.grid(row=0, column=1, padx=10, pady=10)

sweets_var = ctk.BooleanVar(value=False)
sweets_check = ctk.CTkCheckBox(controls_frame, text="Allow Sweets", variable=sweets_var, font=("Segoe UI", 13))
sweets_check.grid(row=0, column=2, padx=10, pady=10)

alcohol_var = ctk.BooleanVar(value=False)
alcohol_check = ctk.CTkCheckBox(
    controls_frame, text="Allow Alcohol (weekends)", variable=alcohol_var, font=("Segoe UI", 13)
)
alcohol_check.grid(row=0, column=3, padx=10, pady=10)

target_kcal_var = ctk.StringVar(value="")
ctk.CTkLabel(controls_frame, text="Target kcal/day:", font=("Segoe UI", 14, "bold")).grid(
    row=0, column=4, padx=10, pady=10, sticky="w"
)
target_entry = ctk.CTkEntry(controls_frame, width=120, textvariable=target_kcal_var, font=("Segoe UI", 13))
target_entry.grid(row=0, column=5, padx=10, pady=10)

scroll_frame = ctk.CTkScrollableFrame(mealplan_page, width=900, height=520, corner_radius=12)
scroll_frame.pack(padx=20, pady=10, fill="both", expand=True)

# store day cards for single-day regeneration
day_card_widgets = {}


def _clear_plan_ui():
    for w in scroll_frame.winfo_children():
        w.destroy()
    day_card_widgets.clear()


def _render_day_card(day_name: str, target_calories: float):
    """
    Creates a day card and returns it.
    Includes "Regenerate Day" button.
    """
    diet = diet_var.get()
    allow_sweets = bool(sweets_var.get())
    allow_alcohol = bool(alcohol_var.get())
    gender = user_profile.get("gender", "M")
    activity = user_profile.get("activity", "Moderately Active")

    meals, totals, targets, flags = build_professional_day_plan(
        day_name=day_name,
        target_kcal=float(target_calories),
        diet_type=diet,
        gender=gender,
        activity_level=activity,
        allow_sweets=allow_sweets,
        allow_alcohol=allow_alcohol,
    )

    card = ctk.CTkFrame(scroll_frame, corner_radius=12)
    card.pack(fill="x", padx=5, pady=6)

    header = ctk.CTkFrame(card, corner_radius=10)
    header.pack(fill="x", padx=10, pady=(10, 6))

    workout_badge = "🏋️ Workout day" if flags["workout_day"] else "🧘 Rest day"
    alcohol_badge = " • 🍷 Alcohol" if flags["alcohol_today"] else ""

    title = ctk.CTkLabel(
        header,
        text=f"{day_name} — {diet} ({workout_badge}{alcohol_badge})",
        font=("Segoe UI", 16, "bold"),
    )
    title.pack(side="left", padx=10, pady=8)

    def regen_this_day():
        new_card = _render_day_card(day_name, target_calories)
        old = day_card_widgets.get(day_name)
        if old is not None:
            old.destroy()
        day_card_widgets[day_name] = new_card
        new_card.pack(fill="x", padx=5, pady=6)

    regen_btn = ctk.CTkButton(header, text="♻ Regenerate Day", width=160, command=regen_this_day)
    regen_btn.pack(side="right", padx=10, pady=8)

    summary = (
        f"Target: {target_calories:.0f} kcal | "
        f"Plan: {totals['kcal']:.0f} kcal\n"
        f"Macros (Plan): P {totals['p']:.0f}g | C {totals['c']:.0f}g | F {totals['f']:.0f}g | Fiber {totals['fiber']:.0f}g\n"
        f"Targets: P {targets['protein_g']:.0f}g | C {targets['carbs_g']:.0f}g | F {targets['fat_g']:.0f}g | Fiber {targets['fiber_g']:.0f}g\n"
        f"Profile: Gender={gender}, Activity={activity}"
    )
    ctk.CTkLabel(card, text=summary, font=("Segoe UI", 12), justify="left").pack(anchor="w", padx=18, pady=(0, 8))

    for meal_name, m in meals:
        mtitle = ctk.CTkLabel(card, text=f"{meal_name}:", font=("Segoe UI", 14, "bold"))
        mtitle.pack(anchor="w", padx=18, pady=(4, 0))

        line = (
            f"• {m['name']} "
            f"(~{m['kcal']:.0f} kcal | P {m['p']:.0f}g, C {m['c']:.0f}g, F {m['f']:.0f}g, Fiber {m['fiber']:.0f}g)"
        )
        ctk.CTkLabel(card, text=line, font=("Segoe UI", 12), justify="left", wraplength=1050).pack(
            anchor="w", padx=28, pady=(0, 2)
        )

    return card


def generate_week_plan(target_calories: float):
    _clear_plan_ui()
    days_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in days_names:
        card = _render_day_card(day, target_calories)
        day_card_widgets[day] = card


def on_generate_plan_clicked():
    global mealplan_target_calories
    try:
        mealplan_target_calories = float(target_kcal_var.get())
    except Exception:
        messagebox.showerror("Invalid Target", "Please enter a valid target calorie number.")
        return

    mealplan_subtitle_var.set(f"Target: ~{mealplan_target_calories:.0f} kcal/day")
    generate_week_plan(mealplan_target_calories)

# Buttons row (Generate + Back side-by-side)
btn_row = ctk.CTkFrame(mealplan_page, corner_radius=12)
btn_row.pack(pady=10)

ctk.CTkButton(
    btn_row,
    text="Generate Weekly Meal Plan",
    font=("Segoe UI", 15, "bold"),
    width=260,
    height=40,
    command=on_generate_plan_clicked,
).grid(row=0, column=0, padx=(0, 10), pady=5)

ctk.CTkButton(
    btn_row,
    text="⬅ Back",
    font=("Segoe UI", 15, "bold"),
    width=160,
    height=40,
    fg_color="#444444",
    hover_color="#555555",
    command=go_back,
).grid(row=0, column=1, padx=(10, 0), pady=5)

ctk.CTkButton(
    mealplan_page,
    text="♻ Regenerate Whole Week",
    font=("Segoe UI", 14, "bold"),
    width=260,
    height=38,
    command=lambda: (
        messagebox.showinfo("Regenerate", "Generating a fresh weekly plan..."),
        generate_week_plan(float(target_kcal_var.get() or (mealplan_target_calories or 2000))),
    ),
).pack(pady=(0, 10))

# ================== START APP ==================
show_tracker_page(push_history=False)
refresh_day_table()
app.mainloop()
