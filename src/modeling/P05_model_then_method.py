"""
[투수] 1) 모델 8종 비교 → 2) 고른 모델로 부상 집계 방식 A/B/C 비교  (9주차 발표와 같은 순서)
==============================================================================================
1단계 모델 비교 : 부상 집계는 방법 B로 고정하고 8개 모델 비교
2단계 방식 비교 : 1단계에서 고른 모델 하나로 A/B/C만 바꿔 비교

평가 (두 가지 모두 출력)
  - 검증: 2021~2023 기록 → 다음 시즌 ERA로 학습, 2024 기록 → 2025 ERA 예측  (9주차와 같은 분할)
  - 순차 평가: 2023·2024·2025 ERA를 각각 그 이전 자료로 학습해 예측한 3개 연도 평균 (교수님 요청)
선택 기준(미리 정함): 순차 평가 3개 연도 평균 MAE가 가장 낮은 것
  방식 A/B/C는 차이가 0.01 이하이면 타자 논문과 같은 B 사용

출력: P05_1_모델비교.csv, P05_2_부상집계방식비교.csv
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

DATA = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data"
TARGET, Y = 'ERA_next', '시즌'

PITCH = ['ERA_KBO', 'IP_KBO', 'G_KBO', 'GS', 'W_KBO', 'L_KBO', 'SV', 'HLD', 'SO_KBO', 'BB_KBO',
         'HR_KBO', 'WHIP_KBO', 'K/9', 'BB/9', 'K/BB', 'FIP', 'WAR', 'TBF']
CAREER = ['나이', '데뷔연차']
PARK = ['Park_Factor']
INJ = {'A': ['부상발생여부', '부상횟수', '총결장일수', '최대심각도', '최고신뢰도점수', '대표부상분류'],
       'B': ['부상발생여부', '부상횟수', '최대결장일수', '최대심각도', '최고신뢰도점수', '대표부상분류'],
       'C': ['부상발생여부', '부상횟수', '총결장일수', '평균심각도', '최고신뢰도점수', '대표부상분류']}


def models():
    m = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.001, max_iter=50000),
        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=50000),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'SVR': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0)),
    }
    if XGBRegressor:
        m['XGBoost'] = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42, verbosity=0)
    if LGBMRegressor:
        m['LightGBM'] = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
    return m


def make_x(d, feats, le=None):
    X = d[feats].copy()
    if '대표부상분류' in feats:
        if le is None:
            le = LabelEncoder().fit(X['대표부상분류'].astype(str))      # 학습 세트로만 맞춤
        known = set(le.classes_)
        X['대표부상분류'] = le.transform(X['대표부상분류'].astype(str).map(lambda v: v if v in known else le.classes_[0]))
    return X.apply(pd.to_numeric, errors='coerce').fillna(0), le


def evaluate(method, model_name):
    df = pd.read_csv(f"{DATA}/kbo_pitcher_dataset_method{method}.csv")
    feats = [c for c in PITCH + CAREER + PARK + INJ[method] if c in df.columns]
    lab = df[df[TARGET].notna()]
    out = {}
    for vy in [2022, 2023, 2024]:
        tr, va = lab[lab[Y] < vy], lab[lab[Y] == vy]
        Xtr, le = make_x(tr, feats)
        Xva, _ = make_x(va, feats, le)
        model = models()[model_name]
        model.fit(Xtr, tr[TARGET])
        p = model.predict(Xva)
        out[vy + 1] = (mean_absolute_error(va[TARGET], p), np.sqrt(mean_squared_error(va[TARGET], p)),
                       r2_score(va[TARGET], p), len(va))
    v = out[2025]
    return {'검증 MAE': v[0], '검증 RMSE': v[1], '검증 R²': v[2], '검증 n': v[3],
            'MAE 2023': out[2023][0], 'MAE 2024': out[2024][0], 'MAE 2025': out[2025][0],
            '3개 연도 평균 MAE': np.mean([out[k][0] for k in out])}


# ── 1단계: 모델 비교 (방법 B 고정) ──────────────────────────────────────
t1 = pd.DataFrame([{'모델': n, **evaluate('B', n)} for n in models()]).sort_values('3개 연도 평균 MAE')
t1 = t1.round(4)
t1.to_csv(f"{DATA}/P05_1_모델비교.csv", index=False, encoding='utf-8-sig')
print("[1단계] 모델 8종 비교 (부상 집계 방법 B, 낮을수록 좋음 / R²는 높을수록 좋음)")
print(t1.to_string(index=False))
best_model = t1.iloc[0]['모델']
print(f"\n→ 3개 연도 평균 MAE 기준 최종 모델: {best_model}\n")

# ── 2단계: 고른 모델로 부상 집계 방식 비교 ───────────────────────────────
t2 = pd.DataFrame([{'방식': m, **evaluate(m, best_model)} for m in 'ABC']).round(4)
t2.to_csv(f"{DATA}/P05_2_부상집계방식비교.csv", index=False, encoding='utf-8-sig')
print(f"[2단계] 부상 집계 방식 비교 (모델: {best_model})")
print(t2.to_string(index=False))
gap = t2['3개 연도 평균 MAE'].max() - t2['3개 연도 평균 MAE'].min()
best_m = t2.sort_values('3개 연도 평균 MAE').iloc[0]['방식']
choice = best_m if gap > 0.01 else 'B'
print(f"\n방식 간 차이 {gap:.4f} → 선택: 방법 {choice}"
      + ("" if gap > 0.01 else "  (차이 0.01 이하 → 타자 논문과 같은 B)"))
print(f"저장 → {DATA}/P05_1_모델비교.csv, P05_2_부상집계방식비교.csv")