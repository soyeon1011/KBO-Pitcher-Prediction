"""
KBO 투수/타자 모델 학습 스크립트
=====================================
- 투수: RandomForestRegressor (n=200) → ERA_next 예측
- 타자: Ridge Regression (α=1.0)     → AVG_KBO_next 예측

실행: python model_train.py
결과: models/pitcher_model.pkl, models/batter_model.pkl 생성

※ 투수는 kbo_pitcher_dataset_methodB.csv 사용
※ 타자는 kbo_final_dataset_methodC.csv 사용
   (마스터 CSV가 없으면 kbo_pitcher_master_final_2021_2025.csv로 대체 시도)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 투수: methodB 데이터셋 (P06_predict_2026.py 기준) ─────────────────────────
PITCHER_DATASET = os.path.join(DATA_DIR, "kbo_pitcher_dataset_methodB.csv")
PITCHER_MASTER  = os.path.join(DATA_DIR, "KBO_Pitcher_Master_2021_2025_agefix.csv")

PITCHER_FEATS = [
    'ERA_KBO', 'IP_KBO', 'G_KBO', 'GS', 'W_KBO', 'L_KBO', 'SV', 'HLD',
    'SO_KBO', 'BB_KBO', 'HR_KBO', 'WHIP_KBO', 'K/9', 'BB/9', 'K/BB',
    'FIP', 'WAR', 'TBF', '나이', '데뷔연차', 'Park_Factor',
    '부상발생여부', '부상횟수', '최대결장일수', '최대심각도', '최고신뢰도점수', '대표부상분류',
]
PITCHER_TARGET = 'ERA_next'

# ── 타자: methodC 데이터셋 (Kbo_final_model.py 기준) ─────────────────────────
BATTER_DATASET = os.path.join(DATA_DIR, "kbo_final_dataset_methodB_v2.csv")
BATTER_MASTER  = os.path.join(DATA_DIR, "KBO_Master_Data_2021_2025_final.csv")

BATTER_FEATS = [
    'AVG_KBO', 'PA_KBO', 'AB_KBO', 'H_KBO', 'HR_KBO', 'RBI_KBO',
    'OBP', 'SLG', 'OPS', 'wRC+', 'WAR', 'BABIP', 'ISO', 'wOBA',
    'BB/K', '데뷔연차', 'Age_2026', 'Park_Factor',
    '부상발생여부', '부상횟수', '최대결장일수', '최대심각도', '최고신뢰도점수', '대표부상분류',
]
BATTER_TARGET = 'AVG_KBO_next'


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def encode_injury(df, le=None, fit=True):
    col = '대표부상분류'
    if col not in df.columns:
        return df, le
    df = df.copy()
    if fit:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
    known = set(le.classes_)
    df[col] = le.transform(df[col].astype(str).map(lambda v: v if v in known else le.classes_[0]))
    return df, le


# ── 투수 모델: RandomForest ───────────────────────────────────────────────────
def train_pitcher():
    # 데이터 파일 선택
    csv = PITCHER_DATASET if os.path.exists(PITCHER_DATASET) else PITCHER_MASTER
    if not os.path.exists(csv):
        print(f"  ⚠️  투수 데이터 없음: {csv}")
        return

    print(f"\n[투수] RandomForest — {os.path.basename(csv)}")
    df = pd.read_csv(csv)
    print(f"  데이터: {len(df)}행, {len(df.columns)}컬럼")

    # 타겟 컬럼 확인
    target = PITCHER_TARGET if PITCHER_TARGET in df.columns else find_col(df, ['ERA_KBO', '평균자책점', 'ERA'])
    if not target:
        print("  ⚠️  ERA 관련 컬럼 없음")
        return

    feats = [c for c in PITCHER_FEATS if c in df.columns]
    print(f"  피처: {len(feats)}개, 타겟: {target}")

    train_df = df[df[target].notna()].copy()
    if len(train_df) < 10:
        print(f"  ⚠️  학습 데이터 부족 ({len(train_df)}행)")
        return

    train_df, le = encode_injury(train_df, fit=True)
    X = train_df[feats].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = train_df[target]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    result = {"model": model, "features": feats, "target": target, "le": le, "type": "pitcher"}
    path = os.path.join(MODEL_DIR, "pitcher_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"  ✅ 학습 완료 ({len(train_df)}행) → {path}")


# ── 타자 모델: Ridge ──────────────────────────────────────────────────────────
def train_batter():
    csv = BATTER_DATASET if os.path.exists(BATTER_DATASET) else BATTER_MASTER
    if not os.path.exists(csv):
        print(f"  ⚠️  타자 데이터 없음: {csv}")
        return

    print(f"\n[타자] Ridge (α=1.0) — {os.path.basename(csv)}")
    df = pd.read_csv(csv)
    print(f"  데이터: {len(df)}행, {len(df.columns)}컬럼")

    target = BATTER_TARGET if BATTER_TARGET in df.columns else find_col(df, ['AVG_KBO', '타율', 'AVG'])
    if not target:
        print("  ⚠️  타율 관련 컬럼 없음")
        return

    feats = [c for c in BATTER_FEATS if c in df.columns]
    print(f"  피처: {len(feats)}개, 타겟: {target}")

    train_df = df[df[target].notna()].copy()
    if len(train_df) < 10:
        print(f"  ⚠️  학습 데이터 부족 ({len(train_df)}행)")
        return

    train_df, le = encode_injury(train_df, fit=True)
    X = train_df[feats].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = train_df[target]

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    result = {"model": model, "features": feats, "target": target, "le": le, "type": "batter"}
    path = os.path.join(MODEL_DIR, "batter_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"  ✅ 학습 완료 ({len(train_df)}행) → {path}")


def main():
    print("=" * 55)
    print("KBO 모델 학습 시작")
    print("  투수: RandomForest → ERA 예측")
    print("  타자: Ridge        → 타율 예측")
    print("=" * 55)
    train_pitcher()
    train_batter()
    print("\n" + "=" * 55)
    print("완료! app.py를 실행하세요.")
    print("=" * 55)


if __name__ == "__main__":
    main()
