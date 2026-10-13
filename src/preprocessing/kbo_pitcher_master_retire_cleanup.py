import pandas as pd

master = pd.read_csv("/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/KBO_Pitcher_Master_2021_2025.csv")
entry = pd.read_csv("/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_entry_pitchers_2026.csv")

active_names = set(entry['선수'])
master_filtered = master[master['이름'].isin(active_names)]

print(f"제거 전: {master['이름'].nunique()}명")
print(f"제거 후: {master_filtered['이름'].nunique()}명")
print(f"제거된 선수: {master['이름'].nunique() - master_filtered['이름'].nunique()}명")

master_filtered.to_csv("/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_pitcher_master_final_2021_2025.csv", index=False, encoding='utf-8-sig')