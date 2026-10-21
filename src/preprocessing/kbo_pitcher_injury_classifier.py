import pandas as pd

df = pd.read_csv("/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_injury_classified.csv")

def assign_confidence(row):
    news = str(row['뉴스기사제목'])
    분류 = row['부상분류']
    
    if news == '관련 기사 없음':
        return 'Low'
    if 분류 == '기타부상':
        return 'Low'
    if 분류 == '성적부진_조정':
        return 'Mid'
    if 분류 in ['어깨_팔꿈치', '하체_다리', '손_허리', '안면_두부', '사고_충돌']:
        return 'High'
    if 분류 == '내과질환':
        return 'Mid'
    return 'Low'

def assign_real_injury(row):
    if row['부상분류'] == '성적부진_조정':
        return 0
    if row['뉴스기사제목'] == '관련 기사 없음' and row['2군출장상세'] != 'X':
        return 0
    return 1

df['신뢰도'] = df.apply(assign_confidence, axis=1)
df['실제부상여부'] = df.apply(assign_real_injury, axis=1)

print(df['신뢰도'].value_counts())
print(df['실제부상여부'].value_counts())

df.to_csv("/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_injury_classified_final.csv", index=False, encoding='utf-8-sig')