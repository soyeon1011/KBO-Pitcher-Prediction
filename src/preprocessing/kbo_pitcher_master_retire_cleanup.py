"""
2026 현역 여부 표시 (은퇴 선수 기록은 지우지 않음)
====================================================
기존: 2026 명단에 없는 선수의 모든 시즌 기록을 삭제 → 은퇴 선수의 과거 기록이 학습에서 빠짐
      (2026년 정보로 과거 학습 데이터를 고르는 셈이라 미래 정보 문제)
수정: 기록은 전부 남기고 '2026현역' 열(1/0)만 추가
      → 학습에는 전체 기록 사용, 2026 예측 대상을 고를 때만 2026현역 == 1 사용

출력 파일 이름은 기존과 같게 두어 다음 단계 코드(add_park_factor.py 등)를 그대로 쓸 수 있게 함
"""
import pandas as pd

DATA = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data"
master = pd.read_csv(f"{DATA}/KBO_Pitcher_Master_2021_2025_agefix.csv")    # 나이·데뷔연차 수정본
entry = pd.read_csv(f"{DATA}/kbo_entry_pitchers_2026.csv")

active_names = set(entry['선수'].astype(str).str.split('(').str[0].str.strip())
master['2026현역'] = master['이름'].isin(active_names).astype(int)

year_col = '시즌' if '시즌' in master.columns else '연도'
last = master[master[year_col] == 2025]
print(f"전체 기록: {len(master)}행, 선수 {master['이름'].nunique()}명 (모두 학습에 사용)")
print(f"2025 기록이 있는 선수 {last['이름'].nunique()}명 중 2026 현역 {last['2026현역'].sum()}명 → 2026 예측 대상")
print(f"2026 명단에 없는 선수 {master.loc[master['2026현역'] == 0, '이름'].nunique()}명 (기록은 유지, 예측 대상에서만 제외)")

master.to_csv(f"{DATA}/kbo_pitcher_master_final_2021_2025.csv", index=False, encoding='utf-8-sig')
print("저장 → kbo_pitcher_master_final_2021_2025.csv")