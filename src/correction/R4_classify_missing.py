# [복원 4단계] 기존 kbo_pitcher_injury_classifier.py와 같은 코드 — 입력·출력 파일만 바꿈
import pandas as pd
import re

df = pd.read_csv("/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_injury_missing_classified.csv")

# ── 부상분류 사전 보정 ────────────────────────────────────────────────────────
# injury_feature_extractor.py 키워드 순서 문제로 생기는 오분류를 여기서 수정
def fix_classification(row):
    news = str(row['뉴스기사제목'])
    분류 = row['부상분류']

    # 1. 피로골절 / 스트레스골절 → 성적부진_조정 오분류 수정
    #    "피로" 키워드에 걸렸지만 실제로는 골절 부상
    if 분류 == '성적부진_조정' and re.search(r'피로골절|스트레스\s*골절', news):
        return '기타부상'

    # 2. 볼넷(사구) → 사고_충돌 오분류 수정
    #    "사구가 많았다" 등 볼넷 맥락인데 몸에 맞는 공(HBP) 맥락이 없으면 수정
    if 분류 == '사고_충돌':
        is_bolnnet    = bool(re.search(r'사구(?:가|를|는|도|로|만)', news))
        is_hitbypitch = bool(re.search(r'몸에\s*맞|데드볼|사구\s*맞|맞은\s*공|타구\s*맞', news))
        if is_bolnnet and not is_hitbypitch:
            return '기타부상'

    return 분류

df['부상분류'] = df.apply(fix_classification, axis=1)

# ── 신뢰도 부여 ───────────────────────────────────────────────────────────────
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

# ── 실제부상여부 ──────────────────────────────────────────────────────────────
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

df.to_csv("/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_injury_missing_classified_final.csv", index=False, encoding='utf-8-sig')