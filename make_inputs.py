#!/usr/bin/env python3
"""課題ごとのInputデータ（業務らしく作った合成データ）を生成する。

公開データの取得に失敗したときの合成代替と、公開データを使えない社内実務問題のための合成データである。
各ノートは `input/*.csv` を読み込む（欠品時はノート内蔵の生成関数へ切替）。
生成ロジックは各ノートの関数と同一であり、シードも同一のため、
CSV経由と内蔵生成で同一のデータになる。

使い方:
    python3 make_inputs.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def w(df: pd.DataFrame, *parts: str) -> None:
    p = ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"Wrote {p} ({len(df)} rows)")


def credit(n: int, seed: int = 42) -> pd.DataFrame:
    """German Credit と同じ列（A1〜A20）の合成代替。値の分布は実データの近似、貸倒との関係は仮置き。"""
    r = np.random.default_rng(seed)

    def cat(codes, p):
        p = np.asarray(p, dtype=float)
        return r.choice(codes, size=n, p=p / p.sum())

    d = pd.DataFrame({
        "A1": cat(["A11", "A12", "A13", "A14"], [27, 27, 6, 40]),
        "A2": np.clip(np.round(r.gamma(3.0, 7.0, size=n)), 4, 72).astype(int),
        "A3": cat(["A30", "A31", "A32", "A33", "A34"], [4, 5, 53, 9, 29]),
        "A4": cat(["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"],
                  [23, 10, 18, 28, 1, 2, 5, 1, 10, 1]),
        "A5": np.clip(np.round(np.exp(r.normal(7.8, 0.75, size=n))), 250, 18424).astype(int),
        "A6": cat(["A61", "A62", "A63", "A64", "A65"], [60, 10, 6, 5, 18]),
        "A7": cat(["A71", "A72", "A73", "A74", "A75"], [6, 17, 34, 17, 25]),
        "A8": cat([1, 2, 3, 4], [14, 23, 16, 47]),
        "A9": cat(["A91", "A92", "A93", "A94"], [5, 31, 55, 9]),
        "A10": cat(["A101", "A102", "A103"], [91, 4, 5]),
        "A11": cat([1, 2, 3, 4], [13, 31, 15, 41]),
        "A12": cat(["A121", "A122", "A123", "A124"], [28, 23, 33, 15]),
        "A13": np.clip(np.round(r.gamma(2.5, 6.6, size=n) + 19), 19, 75).astype(int),
        "A14": cat(["A141", "A142", "A143"], [14, 5, 81]),
        "A15": cat(["A151", "A152", "A153"], [18, 71, 11]),
        "A16": cat([1, 2, 3, 4], [63, 33, 3, 1]),
        "A17": cat(["A171", "A172", "A173", "A174"], [2, 20, 63, 15]),
        "A18": cat([1, 2], [85, 15]),
        "A19": cat(["A191", "A192"], [60, 40]),
        "A20": cat(["A201", "A202"], [96, 4]),
    })
    # 貸倒の起こりやすさ（仮置き）：口座状態・期間・返済履歴・金額・貯蓄・資産・他ローンで決まる
    logit = (
        -1.05
        + d["A1"].map({"A11": 1.0, "A12": 0.5, "A13": -0.2, "A14": -1.2})
        + 0.03 * (d["A2"] - 21)
        + d["A3"].map({"A30": 1.0, "A31": 0.9, "A32": 0.0, "A33": -0.2, "A34": -0.7})
        + 0.00008 * (d["A5"] - 3271)
        + d["A6"].map({"A61": 0.3, "A62": 0.0, "A63": -0.3, "A64": -0.5, "A65": -0.4})
        + 0.4 * (d["A12"] == "A124") + 0.4 * (d["A14"] == "A141")
        + 0.15 * (d["A8"] - 3)
    )
    d["default"] = (r.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return d


def fraud(n: int, seed: int = 7) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    amount = np.clip(r.lognormal(mean=3.0, sigma=1.0, size=n), 1.0, 500.0)
    hour = r.integers(0, 24, size=n)
    night = ((hour <= 4) | (hour >= 23)).astype(float)
    intl = (r.uniform(size=n) < 0.08).astype(int)
    card_age = np.clip(r.exponential(scale=24.0, size=n), 0.0, 120.0)
    new_card = (card_age < 3.0).astype(float)
    online = (r.uniform(size=n) < 0.60).astype(int)
    merchant = r.choice(["retail", "food", "travel", "digital", "other"], size=n,
                        p=[0.35, 0.25, 0.10, 0.15, 0.15])
    risky_merch = np.isin(merchant, ["travel", "digital"]).astype(float)
    logit = (-5.2 + 0.030 * amount * night + 3.0 * intl * new_card
             + 2.2 * risky_merch * online + 1.6 * new_card * night
             + 0.008 * amount * intl - 0.02 * card_age)
    prob = 1.0 / (1.0 + np.exp(-logit))
    fraud = (r.uniform(size=n) < prob).astype(int)
    return pd.DataFrame({
        "amount": amount, "hour": hour, "intl": intl, "card_age": card_age,
        "online": online, "merchant": merchant, "fraud": fraud,
    })


def fraud_modus(n: int, seed: int = 7) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    amount = np.clip(r.lognormal(mean=3.0, sigma=1.0, size=n), 1.0, 500.0)
    hour = r.integers(0, 24, size=n)
    night = ((hour <= 4) | (hour >= 23)).astype(float)
    intl = (r.uniform(size=n) < 0.08).astype(int)
    card_age = np.clip(r.exponential(scale=24.0, size=n), 0.0, 120.0)
    new_card = (card_age < 3.0).astype(float)
    online = (r.uniform(size=n) < 0.60).astype(int)
    merchant = r.choice(["retail", "food", "travel", "digital", "other"], size=n,
                        p=[0.35, 0.25, 0.10, 0.15, 0.15])
    risky_merch = np.isin(merchant, ["travel", "digital"]).astype(float)
    logit_a = -5.4 + 3.0 * intl * new_card + 0.008 * amount * intl - 0.02 * card_age
    logit_b = -5.4 + 0.030 * amount * night * risky_merch * online + 1.6 * new_card * night
    pa = 1.0 / (1.0 + np.exp(-logit_a))
    pb = 1.0 / (1.0 + np.exp(-logit_b))
    is_a = (r.uniform(size=n) < pa).astype(int)
    is_b = ((1 - is_a) * (r.uniform(size=n) < pb)).astype(int)
    fraud = np.maximum(is_a, is_b)
    modus = np.where(is_a == 1, "A", np.where(is_b == 1, "B", "-"))
    return pd.DataFrame({
        "amount": amount, "hour": hour, "intl": intl, "card_age": card_age,
        "online": online, "merchant": merchant, "fraud": fraud, "modus": modus,
    })


def bank(n: int, seed: int = 3) -> pd.DataFrame:
    # 公開データと同じ列名・型の合成代替（取得失敗時のみ使う手順確認用）
    r = np.random.default_rng(seed)
    prev = (r.poisson(1.5, size=n) + 1) * (r.uniform(size=n) < 0.18)
    d = pd.DataFrame({
        "age": r.integers(18, 90, size=n).astype(float),
        "job": r.choice(["blue-collar", "management", "technician", "admin.", "services", "retired"], size=n),
        "marital": r.choice(["married", "single", "divorced"], size=n, p=[0.60, 0.28, 0.12]),
        "education": r.choice(["secondary", "tertiary", "primary", "unknown"], size=n, p=[0.51, 0.29, 0.16, 0.04]),
        "default": r.choice(["no", "yes"], size=n, p=[0.98, 0.02]),
        "balance": np.round(r.lognormal(6.0, 1.5, size=n) - 300.0),
        "housing": r.choice(["yes", "no"], size=n, p=[0.56, 0.44]),
        "loan": r.choice(["no", "yes"], size=n, p=[0.84, 0.16]),
        "contact": r.choice(["cellular", "unknown", "telephone"], size=n, p=[0.65, 0.29, 0.06]),
        "day": r.integers(1, 32, size=n).astype(float),
        "month": r.choice(["may", "jul", "aug", "jun", "nov", "apr"], size=n),
        "duration": np.round(r.exponential(250.0, size=n)),
        "campaign": (r.poisson(1.8, size=n) + 1).astype(float),
        "pdays": np.where(prev > 0, r.integers(1, 400, size=n), -1).astype(float),
        "previous": prev.astype(float),
        "poutcome": np.where(prev > 0, r.choice(["failure", "other", "success"], size=n, p=[0.6, 0.15, 0.25]), "unknown"),
    })
    logit = (-3.6 + 0.004 * d["duration"] - 0.15 * d["campaign"] + 1.8 * (d["poutcome"] == "success")
             + 0.5 * (d["housing"] == "no") + 0.4 * (d["contact"] == "cellular"))
    y = (r.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    d["Class"] = np.where(y == 1, 2, 1)
    return d


def demand(n: int = 144, seed: int = 4) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    t = np.arange(n)
    month = t % 12
    seas = 60.0 * np.sin(2 * np.pi * (month - 3) / 12.0) + 30.0 * (month == 6) + 40.0 * (month == 11)
    y = 120.0 + 1.1 * t + seas + r.normal(0, 12.0, size=n)
    dates = pd.date_range("1949-01", periods=n, freq="MS")
    return pd.DataFrame({"date": dates, "value": y})


def sensors(n_normal: int = 5000, n_abnormal: int = 200, seed: int = 5) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    base = r.normal(0, 1, size=(n_normal, 8))
    normal = base + 0.6 * base[:, [0]]
    shift = np.array([2.5, -2.0, 1.5, 0.5, 0.0, 0.0, 1.0, -1.0])
    abnormal = base[:n_abnormal] * 1.5 + shift + r.normal(0, 0.5, size=(n_abnormal, 8))
    X = np.vstack([normal, abnormal])
    y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_abnormal, dtype=int)])
    cols = {f"s{i}": X[:, i] for i in range(8)}
    cols["label"] = y
    return pd.DataFrame(cols)


def serve_mini(n: int = 4000, seed: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = np.clip(rng.normal(450.0, 150.0, size=n), 150.0, 1500.0)
    past_due = np.clip(rng.poisson(0.4, size=n), 0, 5)
    use_rate = rng.uniform(0.0, 1.0, size=n)
    housing = rng.choice(["rent", "own", "company"], size=n, p=[0.55, 0.35, 0.10])
    logit = -3.4 + 0.9 * past_due + 1.2 * use_rate - 0.003 * (income - 450.0) + 0.2 * (housing == "rent")
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"income": income, "past_due": past_due,
                         "use_rate": use_rate, "housing": housing, "default": y})


def ops_data(n: int, income_shift: float, seed: int) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    income = np.clip(r.normal(450.0 + income_shift, 150.0, size=n), 150.0, 1500.0)
    past_due = np.clip(r.poisson(0.4, size=n), 0, 5)
    use_rate = r.uniform(0.0, 1.0, size=n)
    housing = r.choice(["rent", "own", "company"], size=n, p=[0.55, 0.35, 0.10])
    logit = -3.4 + 0.9 * past_due + 1.2 * use_rate - 0.003 * (income - 450.0) + 0.2 * (housing == "rent")
    y = (r.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"income": income, "past_due": past_due,
                         "use_rate": use_rate, "housing": housing, "default": y})


def make_shadow_like(n: int, income_shift: float, seed: int, cut: float) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    income = np.clip(r.normal(450.0 + income_shift, 150.0, size=n), 150.0, 1500.0)
    past_due = np.clip(r.poisson(0.4, size=n), 0, 5)
    use_rate = r.uniform(0.0, 1.0, size=n)
    housing = r.choice(["rent", "own", "company"], size=n, p=[0.55, 0.35, 0.10])
    # 低年収×高利用率の層だけ危険度が跳ね上がる（線形では表しにくい区切り）。区切りの年収は景気で動く
    risky = (income < cut) & (use_rate > 0.6)
    logit = -3.0 + 0.9 * past_due - 0.002 * (income - 450.0) + 0.2 * (housing == "rent") + 2.5 * risky
    y = (r.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"income": income, "past_due": past_due,
                         "use_rate": use_rate, "housing": housing, "default": y})


def main() -> None:
    w(credit(1000), "01_credit", "input", "applications.csv")  # German Credit形式の合成代替
    w(fraud(20000), "02_fraud", "input", "transactions.csv")
    w(fraud_modus(20000), "02_fraud", "input", "transactions_modus.csv")
    w(bank(45211), "03_targeting", "input", "bank.csv")
    w(demand(), "04_demand", "input", "passengers.csv")
    w(sensors(), "05_inspection", "input", "sensors.csv")
    w(serve_mini(), "06_ops", "input", "mini_applications.csv")
    w(ops_data(6000, 0.0, 61), "06_ops", "input", "ops_base.csv")
    w(ops_data(2000, -120.0, 62), "06_ops", "input", "ops_new.csv")
    w(make_shadow_like(6000, 0.0, 70, 300.0), "07_production", "input", "shadow_base.csv")
    w(make_shadow_like(2000, -120.0, 71, 180.0), "07_production", "input", "shadow_new.csv")


if __name__ == "__main__":
    main()
