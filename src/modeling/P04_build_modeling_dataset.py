"""
[투수] 모델링 데이터 만들기 — 파크팩터 + 부상 변수(A/B/C) + 다음 시즌 ERA(정답)
==================================================================================
입력
  KBO_Pitcher_Master_2021_2025_agefix.csv  (나이·데뷔연차 수정본, 은퇴 선수 포함 645행)
  kbo_injury_master_final_all.csv          (부상 기록 319건, 은퇴 선수·장기 결장 포함)
  statiz_pf_yearly_adjusted.csv            (스탯티즈 파크팩터 연도별 보정값, 타자와 같은 파일)
  kbo_entry_pitchers_2026.csv              (2026 등록 명단 → 2026 예측 대상 표시용)

만드는 것 (타자 04단계와 같은 구조)
  1) Park_Factor : 소속팀 홈구장의 스탯티즈 연도별 보정값을 리그 안타 구성비로 가중합
  2) 부상 변수   : 실제부상여부 == 1 인 기록만, [이름 + 팀 + 시즌]으로 연결
       결장일수는 그 시즌 안(10/31까지)에서만 계산 → 비시즌·다음 시즌까지 이어진 기간 제외
       A: 총 결장일수 + 최대 심각도 / B: 최장 결장 1건 + 최대 심각도 / C: 총 결장일수 + 평균 심각도
  3) 정답 ERA_next : 같은 선수(이름 + 출생연도)의 다음 시즌 ERA (동명이인 섞임 방지)
       다음 시즌 기록이 없거나 25이닝 미만이면 NaN (Master가 이미 25이닝 이상만 포함)
  4) 2026현역 : 2026 등록 명단에 있으면 1 → 2026 예측 대상을 고를 때만 사용 (학습 데이터는 그대로)

출력: kbo_pitcher_dataset_methodA.csv / B / C
"""
import pandas as pd
import numpy as np

DATA = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data"
MASTER = f"{DATA}/KBO_Pitcher_Master_2021_2025_agefix.csv"
INJURY = f"{DATA}/kbo_injury_master_final_all.csv"
PF_FILE = f"{DATA}/statiz_pf_yearly_adjusted.csv"
ENTRY = f"{DATA}/kbo_entry_pitchers_2026.csv"

HOME = {'KIA': '광주', 'KT': '수원', 'LG': '잠실', '두산': '잠실', 'NC': '창원',
        'SSG': '문학', '롯데': '사직', '삼성': '대구', '키움': '고척'}
WEIGHTS = {2021: (0.723, 0.174, 0.013, 0.089), 2022: (0.732, 0.171, 0.016, 0.081),
           2023: (0.745, 0.170, 0.017, 0.068), 2024: (0.717, 0.168, 0.017, 0.098),
           2025: (0.722, 0.175, 0.018, 0.086)}
CONF_SCORE = {'High': 4, 'Mid': 3, 'Mid(LLM)': 3, 'Low': 1}

df = pd.read_csv(MASTER)
Y = '시즌'

# ── 1) 파크팩터 ──────────────────────────────────────────────────────────
pf = pd.read_csv(PF_FILE)
pf['Park_Factor'] = [round(sum(w * r[c] for w, c in zip(WEIGHTS[r['연도']], ['단타', '2루타', '3루타', '홈런'])), 1)
                     for _, r in pf.iterrows()]
df['구장'] = [('대전(신)' if y >= 2025 else '대전(구)') if t == '한화' else HOME.get(t)
             for t, y in zip(df['팀명'], df[Y])]
df = df.drop(columns=['Park_Factor'], errors='ignore').merge(
    pf[['구장', '연도', 'Park_Factor']].rename(columns={'연도': Y}), on=['구장', Y], how='left')
print(f"파크팩터 부여: {df['Park_Factor'].notna().sum()}/{len(df)}행")

# ── 2) 부상 변수 ─────────────────────────────────────────────────────────
inj = pd.read_csv(INJURY)
n_all = len(inj)
inj = inj[inj['실제부상여부'] == 1].copy()
print(f"부상 기록 {n_all}건 중 실제 부상 {len(inj)}건 사용 (비부상 {n_all - len(inj)}건 제외)")
inj['선수'] = inj['선수'].astype(str).str.split('(').str[0].str.strip()
inj[Y] = pd.to_datetime(inj['부상시작일']).dt.year
# 결장일수는 해당 시즌 안에서만 센다 (시즌 종료일 10/31까지)
#  - 복귀 기록이 없으면(시즌 아웃) 시즌 종료일까지
#  - 다음 시즌에 복귀했으면 비시즌 기간이 들어가지 않도록 시즌 종료일에서 자름
start = pd.to_datetime(inj['부상시작일'])
ret = pd.to_datetime(inj['실제복귀일'], errors='coerce')
season_end = pd.to_datetime(start.dt.year.astype(str) + '-10-31')
inj['결장'] = (ret.where(ret.notna() & (ret <= season_end), season_end) - start).dt.days.clip(lower=0)
inj['신뢰도점수'] = inj['신뢰도'].map(CONF_SCORE).fillna(1)

g = inj.sort_values('결장', ascending=False).groupby(['선수', '팀', Y])
agg = pd.DataFrame({
    '부상횟수': g.size(),
    '총결장일수': g['결장'].sum(),
    '최대결장일수': g['결장'].max(),
    '최대심각도': g['심각도점수'].max(),
    '평균심각도': g['심각도점수'].mean().round(2),
    '최고신뢰도점수': g['신뢰도점수'].max(),
    '대표부상분류': g['부상분류'].first(),          # 결장일수가 가장 긴 부상의 분류
}).reset_index().rename(columns={'선수': '이름', '팀': '팀명'})
agg['부상발생여부'] = 1

df = df.merge(agg, on=['이름', '팀명', Y], how='left')
for c in ['부상발생여부', '부상횟수', '총결장일수', '최대결장일수', '최대심각도', '평균심각도', '최고신뢰도점수']:
    df[c] = df[c].fillna(0)
df['대표부상분류'] = df['대표부상분류'].fillna('정상')
linked = (df['부상발생여부'] == 1).sum()
print(f"부상 있는 선수-시즌: {linked}/{len(df)}행 (부상 집계 {len(agg)}건 중 {linked}건 연결)")

# ── 3) 정답: 다음 시즌 ERA (이름 + 출생연도) ──────────────────────────────
df['선수키'] = df['이름'] + '_' + df['출생연도'].astype('Int64').astype(str)
nxt = (df.groupby(['선수키', Y])[['ER_KBO', 'IP_KBO']].sum().reset_index())
nxt['ERA_next'] = (nxt['ER_KBO'] * 9 / nxt['IP_KBO']).round(2)
nxt['IP_next'] = nxt['IP_KBO'].round(1)
nxt[Y] = nxt[Y] - 1                                     # t+1 시즌 기록을 t 시즌 행에 붙임
df = df.merge(nxt[['선수키', Y, 'ERA_next', 'IP_next']], on=['선수키', Y], how='left')
print(f"정답(다음 시즌 ERA) 있는 행: {df['ERA_next'].notna().sum()}  "
      f"(연도별 {df[df['ERA_next'].notna()].groupby(Y).size().to_dict()})")

# ── 4) 2026 예측 대상 표시 ────────────────────────────────────────────────
active = set(pd.read_csv(ENTRY)['선수'].astype(str).str.split('(').str[0].str.strip())
df['2026현역'] = df['이름'].isin(active).astype(int)
pred = df[(df[Y] == 2025) & (df['2026현역'] == 1)]
print(f"2026 예측 대상: 2025 기록 {(df[Y] == 2025).sum()}명 중 현역 {len(pred)}명")

# ── 방법별 저장 ───────────────────────────────────────────────────────────
keep = {'A': ['총결장일수', '최대심각도'], 'B': ['최대결장일수', '최대심각도'], 'C': ['총결장일수', '평균심각도']}
drop_all = ['총결장일수', '최대결장일수', '최대심각도', '평균심각도']
for m, cols in keep.items():
    out = df.drop(columns=[c for c in drop_all if c not in cols])
    path = f"{DATA}/kbo_pitcher_dataset_method{m}.csv"
    out.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  방법 {m} → {path} ({out.shape[0]}행 × {out.shape[1]}열)")

print("\n[부상 변수 요약 — 방법 B]")
print(df[['부상발생여부', '부상횟수', '최대결장일수', '최대심각도', '최고신뢰도점수']].describe().round(2).to_string())
print(df['대표부상분류'].value_counts().to_string())