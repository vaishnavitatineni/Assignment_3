import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, json, os
os.makedirs("results", exist_ok=True)

df = pd.read_csv("diabetes.csv")
sample = df.sample(n=25, random_state=72)

def g_stats(data, s):
    r = {
        "pop_mean": data["Glucose"].mean(),
        "samp_mean": s["Glucose"].mean(),
        "pop_max": data["Glucose"].max(),
        "samp_max": s["Glucose"].max()
    }
    plt.figure(figsize=(8, 4))
    sns.histplot(data["Glucose"], kde=True, color="skyblue", label="Full")
    sns.histplot(s["Glucose"], kde=True, color="orange", label="Sample")
    plt.axvline(r["pop_mean"], color="blue", linestyle="--")
    plt.axvline(r["samp_mean"], color="red", linestyle="--")
    plt.legend(); plt.tight_layout(); plt.savefig("glucose_distribution.png"); plt.close()
    return r

def bmi_pct(data, s, p=98):
    pop = np.percentile(data["BMI"], p)
    samp = np.percentile(s["BMI"], p)
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data["BMI"], fill=True, label="Full")
    sns.kdeplot(s["BMI"], fill=True, color="orange", label="Sample")
    plt.axvline(pop, color="blue", linestyle="--")
    plt.axvline(samp, color="red", linestyle="--")
    plt.legend(); plt.tight_layout(); plt.savefig("bmi_percentile.png"); plt.close()
    return {"pop_pct": pop, "samp_pct": samp}

def bp_boot(data, col="BloodPressure", n=150, reps=500):
    m, s, p = [], [], []
    for _ in range(reps):
        b = data.sample(n=n, replace=True)
        m.append(b[col].mean()); s.append(b[col].std()); p.append(np.percentile(b[col], 98))
    plt.figure(figsize=(10, 4))
    sns.histplot(m, kde=True, color="purple", label="Bootstraps")
    plt.axvline(np.mean(m), color="black", linestyle="--")
    plt.legend(); plt.tight_layout(); plt.savefig("bloodpressure_bootstrap.png"); plt.close()
    return {
        "boot_mean": np.mean(m),
        "pop_mean": data[col].mean(),
        "boot_std": np.mean(s),
        "pop_std": data[col].std(),
        "boot_98th": np.mean(p),
        "pop_98th": np.percentile(data[col], 98)
    }

def to_json(obj): return obj.item() if isinstance(obj, (np.integer, np.floating)) else obj

res = {
    "glucose": g_stats(df, sample),
    "bmi": bmi_pct(df, sample),
    "blood_pressure": bp_boot(df)
}
with open("results.json", "w") as f: json.dump(res, f, indent=2, default=to_json)