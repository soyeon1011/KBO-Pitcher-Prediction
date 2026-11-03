"""
[복원 6단계] 기존 부상 분류 결과(213건) + 새로 분류한 빠진 기록(106건) 합치기
=============================================================================
출력: kbo_injury_master_final_all.csv  ← 앞으로 부상 변수는 이 파일로 만든다
(기존 kbo_injury_master_final.csv는 그대로 둠)
"""
import pandas as pd

DATA = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data"
old = pd.read_csv(f"{DATA}/kbo_injury_master_final.csv")
new = pd.read_csv(f"{DATA}/kbo_injury_missing_master_final.csv")

old['2군확인'] = '확인'
allf = pd.concat([old, new], ignore_index=True)
allf['부상시작일'] = pd.to_datetime(allf['부상시작일']).dt.strftime('%Y-%m-%d')
before = len(allf)
allf = allf.drop_duplicates(subset=['선수', '팀', '부상시작일'], keep='first')
allf = allf.sort_values('부상시작일').reset_index(drop=True)

allf.to_csv(f"{DATA}/kbo_injury_master_final_all.csv", index=False, encoding='utf-8-sig')
print(f"기존 {len(old)}건 + 추가 {len(new)}건 = {before}건 → 중복 제거 후 {len(allf)}건")
print(f"선수 {allf['선수'].nunique()}명")
print("\n실제부상여부:", allf['실제부상여부'].value_counts().to_dict())
print("신뢰도:", allf['신뢰도'].value_counts().to_dict())
print("\n연도별:", pd.to_datetime(allf['부상시작일']).dt.year.value_counts().sort_index().to_dict())
print(f"\n저장 → {DATA}/kbo_injury_master_final_all.csv")
