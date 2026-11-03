"""
투수 데이터 — 나이·데뷔연차를 '해당 시즌 기준'으로 다시 계산
================================================================
기존: Age_2026 = 2026 − 출생연도 + 1, 데뷔연차 = 2026 − 입단연도 + 1
      → 선수의 모든 시즌 행에 2026년 값이 똑같이 들어감 (과거 행의 나이가 틀림)
수정: 나이 = 시즌 − 출생연도, 데뷔연차 = 시즌 − 입단연도 + 1   (타자 논문과 같은 기준)

크롤링은 다시 하지 않는다. 이미 수집한 출생연도·입단연도로 계산만 다시 한다.
입단 나이가 이상한 선수(동명이인 등으로 잘못 수집됐을 가능성)는 따로 파일로 뽑아 직접 확인한다.

사용법: 모델에 넣는 최종 데이터 파일 경로를 IN_FILE에 넣고 실행
출력 : *_agefix.csv          (나이·데뷔연차 열 추가, 원본은 그대로)
       pitcher_birth_debut_check.csv  (확인이 필요한 선수 목록)
모델 코드에서는 'Age_2026' 대신 '나이'를 특징으로 사용
"""
import os
import pandas as pd

DATA_DIR = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data"
IN_FILE  = os.path.join(DATA_DIR, "KBO_Pitcher_Master_2021_2025.csv")   # ← 최종 데이터 파일로 바꿔도 됨
OUT_FILE = IN_FILE.replace(".csv", "_agefix.csv")
CHECK    = os.path.join(DATA_DIR, "pitcher_birth_debut_check.csv")

# 직접 확인해서 고친 값 (모두 확인 완료)
BIRTH_FIX = {      # (이름, 팀명): 출생연도
    ('이상민', '삼성'): 1990,   # 1983은 다른 이상민의 값
}
DEBUT_FIX = {      # (이름, 출생연도): KBO 입단연도
    ('구승민', 1990): 2013, ('최지민', 2003): 2022, ('최지명', 1995): 2018,
    ('최승용', 2001): 2021, ('조병현', 2002): 2021, ('정수민', 1990): 2016,
    ('전미르', 2005): 2024, ('장원준', 1985): 2004, ('이준영', 1992): 2015,
    ('이우찬', 1992): 2011, ('이상민', 1990): 2013, ('이병헌', 2003): 2022,
    ('한현희', 1993): 2012, ('이민호', 2001): 2020, ('서준원', 2000): 2019,
    ('백승현', 1995): 2015, ('박종훈', 1991): 2010, ('문경찬', 1992): 2015,
    ('김택형', 1996): 2015, ('김종수', 1994): 2013, ('김영현', 2002): 2021,
    ('김민수', 1992): 2015,
    ('김민', 1999): 2018, ('김기훈', 2000): 2019, ('이로운', 2004): 2023,
    ('홍상삼', 1990): 2008, ('김민우', 1995): 2015, ('이승현', 2002): 2021,
}

df = pd.read_csv(IN_FILE)
year_col = '시즌' if '시즌' in df.columns else '연도'

for (name, team), v in BIRTH_FIX.items():
    df.loc[(df['이름'] == name) & (df['팀명'] == team), '출생연도'] = v
df['출생연도'] = pd.to_numeric(df['출생연도'], errors='coerce')
for (name, birth), v in DEBUT_FIX.items():
    df.loc[(df['이름'] == name) & (df['출생연도'] == birth), '입단연도'] = v

df['출생연도'] = pd.to_numeric(df['출생연도'], errors='coerce')
df['입단연도'] = pd.to_numeric(df['입단연도'], errors='coerce')

df['나이'] = df[year_col] - df['출생연도']
df['데뷔연차'] = df[year_col] - df['입단연도'] + 1
bad = (df['데뷔연차'] < 1) | (df['데뷔연차'] > 30)
df.loc[bad, '데뷔연차'] = pd.NA            # 입단연도가 시즌보다 늦은 등 말이 안 되는 값은 비움

df.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
print(f"저장 → {OUT_FILE}")
print(df[['이름', '팀명', year_col, '출생연도', '입단연도', '나이', '데뷔연차']].head(8).to_string(index=False))

# ── 확인이 필요한 선수 ────────────────────────────────────────────────
p = (df.groupby(['이름', '출생연도', '입단연도'], dropna=False)
       .agg(팀목록=('팀명', lambda s: '/'.join(dict.fromkeys(s))), 첫시즌=(year_col, 'min'))
       .reset_index())
p['입단나이'] = p['입단연도'] - p['출생연도']
reason = pd.Series('', index=p.index)
reason[p['출생연도'].isna()] += '출생연도 없음 '
reason[p['입단연도'].isna()] += '입단연도 없음 '
reason[(p['입단나이'] < 17) | (p['입단나이'] > 24)] += '입단 나이 이상 '
reason[p['입단연도'] > p['첫시즌']] += '입단연도가 첫 시즌보다 늦음 '
p['확인 이유'] = reason.str.strip()
chk = p[p['확인 이유'] != ''].sort_values('확인 이유')
chk.to_csv(CHECK, index=False, encoding='utf-8-sig')
print(f"\n확인이 필요한 선수 {len(chk)}명 → {CHECK}")
print(chk.head(20).to_string(index=False))