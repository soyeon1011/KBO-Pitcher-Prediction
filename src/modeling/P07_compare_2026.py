"""
[투수] 2026 예측 ERA vs 2026 시즌 실제 ERA 비교
=================================================
입력 : kbo_pitcher_2026_predictions.csv  (P06 결과)
       statiz_pitcher_stats_2026.csv      (시즌 종료 후 스탯티즈 투수 기록)
비교 대상 : 예측한 투수 중 2026년에 25이닝 이상 던진 투수 (학습 기준과 같음)
          같은 이름이 두 명 이상인 경우(동명이인)는 제외
기준선   : 2025 ERA 그대로 / 2025 FIP / 2025 리그 평균 ERA  (예측 시점에 알 수 있는 값)

출력 : kbo_pitcher_2026_validation.csv  (순위, 선수명, 2026_예측ERA, 실제ERA, 이닝, 오차, 절대오차)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data"
PRED = f"{DATA}/kbo_pitcher_2026_predictions.csv"
STATIZ = f"{DATA}/statiz_pitcher_stats_2026.csv"
MASTER = f"{DATA}/kbo_pitcher_dataset_methodB.csv"
OUT = f"{DATA}/kbo_pitcher_2026_validation.csv"
MIN_IP = 25


def ip_to_float(v):
    """스탯티즈 이닝 표기 169.2 → 169과 2/3"""
    whole, _, frac = str(v).partition('.')
    return int(whole) + (int(frac) / 3 if frac else 0)


pred = pd.read_csv(PRED)
stz = pd.read_csv(STATIZ)
stz['Name'] = stz['Name'].astype(str).str.strip()
stz['이닝'] = stz['IP'].apply(ip_to_float)

dup = stz['Name'][stz['Name'].duplicated(keep=False)].unique()
if len(dup):
    print(f"동명이인으로 제외: {sorted(dup)}")
stz = stz[~stz['Name'].isin(dup)]

df = pred.merge(stz[['Name', 'ERA', '이닝']].rename(columns={'Name': '선수명', 'ERA': '실제ERA'}),
                on='선수명', how='inner')
n_match = len(df)
df = df[df['이닝'] >= MIN_IP].copy()
print(f"예측 {len(pred)}명 중 2026 기록이 있는 {n_match}명, 그중 {MIN_IP}이닝 이상 {len(df)}명 비교")

# 기준선
m = pd.read_csv(MASTER)
m25 = m[m['시즌'] == 2025]
lg25 = m25['ER_KBO'].sum() * 9 / m25['IP_KBO'].sum()
y = df['실제ERA'].values
cands = {'모델 (RandomForest)': df['2026_예측ERA'].values,
         '2025 ERA 그대로': df['2025_ERA'].values,
         '2025 FIP': df['2025_FIP'].values,
         f'리그 평균 ERA (2025, {lg25:.2f})': np.full(len(df), lg25)}
rows = [{'예측 방법': k, 'MAE': mean_absolute_error(y, p), 'RMSE': np.sqrt(mean_squared_error(y, p)),
         'R²': r2_score(y, p)} for k, p in cands.items()]
print("\n[2026 실제 ERA와 비교] (MAE·RMSE 낮을수록, R² 높을수록 좋음)")
print(pd.DataFrame(rows).round(3).to_string(index=False))

df['오차'] = (df['2026_예측ERA'] - df['실제ERA']).round(2)
df['절대오차'] = df['오차'].abs()
e = df['절대오차']
print(f"\n절대오차 0.5 이내 {(e <= 0.5).sum()}명 / 1.0 이내 {(e <= 1.0).sum()}명 / 2.0 초과 {(e > 2.0).sum()}명 "
      f"(전체 {len(df)}명), 평균 편향(예측−실제) {df['오차'].mean():+.2f}")

out = df[['선수명', '2026_예측ERA', '실제ERA', '이닝', '오차', '절대오차']].copy()
out['이닝'] = out['이닝'].round(1)
out = out.sort_values('절대오차').reset_index(drop=True)
out.index += 1
out.to_csv(OUT, index_label='순위', encoding='utf-8-sig')
print(f"\n저장 → {OUT}")
print(out.head(10).to_string())