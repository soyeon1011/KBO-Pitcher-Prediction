"""
[투수] 최종 모델로 2026 ERA 예측
==================================
모델 : RandomForest (트리 200개, random_state=42) — P05 1단계에서 선택
부상 : 방법 B (최장 결장 1건 + 최대 심각도) — P05 2단계에서 선택
학습 : 2021~2024 기록 → 각 다음 시즌 ERA (정답이 있는 행 전체, 은퇴 선수 포함)
예측 : 2025 기록 → 2026 ERA, 2026 등록 명단에 있는 투수만 (2026현역 == 1)

출력 : kbo_pitcher_2026_predictions.csv  (예측 ERA 낮은 순)
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

DATA = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data"
IN_FILE = f"{DATA}/kbo_pitcher_dataset_methodB.csv"
OUT_FILE = f"{DATA}/kbo_pitcher_2026_predictions.csv"
TARGET, Y = 'ERA_next', '시즌'

FEATS = ['ERA_KBO', 'IP_KBO', 'G_KBO', 'GS', 'W_KBO', 'L_KBO', 'SV', 'HLD', 'SO_KBO', 'BB_KBO',
         'HR_KBO', 'WHIP_KBO', 'K/9', 'BB/9', 'K/BB', 'FIP', 'WAR', 'TBF',
         '나이', '데뷔연차', 'Park_Factor',
         '부상발생여부', '부상횟수', '최대결장일수', '최대심각도', '최고신뢰도점수', '대표부상분류']

df = pd.read_csv(IN_FILE)
FEATS = [c for c in FEATS if c in df.columns]

train = df[df[TARGET].notna()].copy()                       # 2021~2024 기록 + 다음 시즌 ERA
# 2026 엔트리 크롤링(kbo_entry_pitchers_2026.csv)에서 빠졌지만 2026 시즌에 실제로 뛴 투수
# (수집 시점에 등록 명단에 없었거나, 같은 이름의 다른 팀 선수와 중복으로 지워진 경우)
EXTRA_ACTIVE = ['오원석', '손동현', '원상현', '이상동', '송영진', '주승우',
                '박명근', '함덕주', '황동하', '홍민규', '홍민기', '우규민']     # 우규민: 2026 출전 여부 확인 후 추가
df.loc[(df[Y] == 2025) & df['이름'].isin(EXTRA_ACTIVE), '2026현역'] = 1
pred = df[(df[Y] == 2025) & (df['2026현역'] == 1)].copy()    # 2025 기록, 2026 현역

le = LabelEncoder().fit(train['대표부상분류'].astype(str))
known = set(le.classes_)


def make_x(d):
    X = d[FEATS].copy()
    X['대표부상분류'] = le.transform(X['대표부상분류'].astype(str).map(lambda v: v if v in known else le.classes_[0]))
    return X.apply(pd.to_numeric, errors='coerce').fillna(0)


model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(make_x(train), train[TARGET])
pred['2026_예측ERA'] = model.predict(make_x(pred)).round(2)

print(f"학습: {len(train)}행 (특징 연도 {sorted(train[Y].unique())}) / 특징 {len(FEATS)}개")
print(f"예측: 2025 기록 {(df[Y] == 2025).sum()}명 중 2026 현역 {len(pred)}명")

out = pred[['이름', '팀명', '나이', 'ERA_KBO', 'IP_KBO', 'FIP', 'WHIP_KBO', 'K/9', 'BB/9',
            '부상발생여부', '최대결장일수', '최대심각도', '2026_예측ERA']].rename(columns={
    '이름': '선수명', '팀명': '2025_팀', '나이': '2025_나이', 'ERA_KBO': '2025_ERA', 'IP_KBO': '2025_이닝',
    'FIP': '2025_FIP', 'WHIP_KBO': '2025_WHIP', 'K/9': '2025_K/9', 'BB/9': '2025_BB/9',
    '부상발생여부': '2025_부상여부', '최대결장일수': '2025_최대결장일수', '최대심각도': '2025_최대심각도'})
out['2025_이닝'] = out['2025_이닝'].round(1)
out = out.sort_values('2026_예측ERA').reset_index(drop=True)
out.index += 1
out.to_csv(OUT_FILE, index_label='순위', encoding='utf-8-sig')

print(f"\n[2026 예측 ERA 상위 15명]")
print(out.head(15).to_string())
print(f"\n예측 ERA 범위: {out['2026_예측ERA'].min()} ~ {out['2026_예측ERA'].max()}  "
      f"(2025 실제 ERA 범위: {out['2025_ERA'].min()} ~ {out['2025_ERA'].max()})")

imp = pd.Series(model.feature_importances_, index=FEATS).sort_values(ascending=False)
print("\n[특징 중요도 상위 10개]")
print(imp.head(10).round(3).to_string())
print(f"\n저장 → {OUT_FILE}")