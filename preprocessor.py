import json, os, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load(file_path, meta_path):
    df = pd.read_csv(file_path)
    with open(meta_path) as f: meta = json.load(f)
    return df, meta

def detect_time_col(df):
    for c in df.columns:
        try:
            pd.to_datetime(df[c].iloc[:50])  # quick probe
            return c
        except: pass
    raise ValueError("No time-like column found")

def align_time(df, time_col):
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)
    freq = pd.infer_freq(df.index) or "H"   # default hourly if unknown
    df = df.asfreq(freq)
    return df, freq

def impute_missing(df):
    num = df.select_dtypes(include=[np.number]).columns
    cat = df.select_dtypes(exclude=[np.number]).columns
    if len(num): df[num] = df[num].interpolate(limit_direction="both")
    if len(num): df[num] = df[num].fillna(df[num].median())
    if len(cat): df[cat] = df[cat].ffill().bfill()
    return df

def clip_outliers(df, z=3.0):
    num = df.select_dtypes(include=[np.number]).columns
    if not len(num): return df
    mu, sigma = df[num].mean(), df[num].std().replace(0, 1)
    df[num] = df[num].clip(lower=mu - z*sigma, upper=mu + z*sigma, axis=1)
    return df

def scale_train_only(df, target_col, split_idx):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num: num.remove(target_col)
    scaler = StandardScaler()
    train = df.iloc[:split_idx]
    scaler.fit(train[num])
    df[num] = scaler.transform(df[num])
    return df, scaler

def time_split(df, train=0.7, val=0.15):
    n = len(df)
    i_train = int(n*train)
    i_val = int(n*(train+val))
    return {"train_end": i_train, "val_end": i_val, "n": n}

def run(file_path, meta_path, out_dir="App/static/output"):
    os.makedirs(out_dir, exist_ok=True)
    df, meta = load(file_path, meta_path)
    target = meta.get("target_variable")
    time_col = next((c for c in df.columns if c.lower() in ("time","date","timestamp")), None) or detect_time_col(df)

    report = {"file": file_path, "target": target, "steps":[]}

    # 1) Align time
    df, freq = align_time(df, time_col); report["freq"] = freq; report["steps"].append("align_time")

    # 2) Impute missing
    df = impute_missing(df); report["steps"].append("impute_missing")

    # 3) Outliers (gate via metadata keywords)
    issues_raw = meta.get("known_data_quality_issues")
    if isinstance(issues_raw, str):
        issues = issues_raw.lower()
    elif isinstance(issues_raw, (list, tuple, set)):
        issues = " ".join(map(str, issues_raw)).lower()
    elif isinstance(issues_raw, dict):
        issues = " ".join(map(str, issues_raw.values())).lower()
    elif issues_raw is None:
        issues = ""
    else:
        issues = str(issues_raw).lower()
    if "outlier" in issues or "spike" in issues:
        df = clip_outliers(df); report["steps"].append("clip_outliers")

    # 4) Split & scale (fit scaler on train only)
    split = time_split(df); report["split"] = split
    df_scaled, scaler = scale_train_only(df.copy(), target, split["train_end"]); report["steps"].append("scale_zscore")

    # 5) Save artifacts
    clean_csv = os.path.join(out_dir, os.path.basename(file_path).replace(".csv", "_clean.csv"))
    df_scaled.reset_index().to_csv(clean_csv, index=False)
    with open(os.path.join(out_dir, "split_indices.json"), "w") as f: json.dump(split, f, indent=2)
    with open(os.path.join(out_dir, "preprocess_report.json"), "w") as f: json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f: pickle.dump(scaler, f)

    print(f"✅ Saved: {clean_csv}")
    return clean_csv

if __name__ == "__main__":
    # example:
    # python App/backend/preprocess_pipeline.py
    run("App/static/input/ETTh1.csv", "App/backend/output.json")
