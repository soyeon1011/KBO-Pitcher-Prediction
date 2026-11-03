import pandas as pd
import re

# 1. 파일 불러오기
INPUT_FILE  = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_injury_naver_news.csv"
OUTPUT_FILE = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_injury_classified.csv"

df = pd.read_csv(INPUT_FILE)

# 🌟 2. 부상 부위 분류 로직 (투수용 키워드 추가)
def classify_injury(text):
    text = str(text)

    # '등' 오인식 방지: "A 등 B" 형태는 신체 부위가 아님
    back_keyword = '등' in text
    if back_keyword:
        if re.search(r'등[이을은도가와과]', text) or re.search(r'[가-힣] 등 [가-힣]', text):
            back_keyword = False

    # 어깨/팔꿈치 (투수 핵심 부위)
    if any(w in text for w in ['어깨', '팔꿈치', '엘보', '회전근개', '건염', '건초염',
                                '전완부', '전완', '굴곡근', '이두근', '삼두근', '상완',
                                '회전근', '상지', '손톱',
                                '팔뚝']): return '어깨_팔꿈치' 

    # 하체/다리
    if any(w in text for w in ['햄스트링', '허벅지', '종아리', '발목', '무릎', '십자인대',
                                '발가락', '내전근', '고관절', '비복근', '아킬레스',
                                '대퇴', '하지']): return '하체_다리'

    # 손/허리
    if any(w in text for w in ['허리', '손가락', '손등', '손목', '늑골', '갈비뼈', '옆구리',
                                '복사뼈', '내복사근', '복근', '복사근', '척추', '요추',
                                '디스크', '골반', '광배근', '활배근',
                                '장요근']) or back_keyword: return '손_허리'

    # 안면/두부
    if any(w in text for w in ['헤드샷', '안면', '코뼈', '턱', '두부', '뇌진탕',
                                '입술', '목', '경추']): return '안면_두부'

    # 내과질환
    if any(w in text for w in ['감기', '몸살', '코로나', '장염', '봉와직염', '대상포진',
                                '맹장', '충수염', '가래톳', '백신', '질환']): return '내과질환'

    # 사고/충돌
    if any(w in text for w in ['충돌', '사고', '부딪혀', '타구', '맞아', '사구']): return '사고_충돌'

    # 성적부진/조정
    if any(w in text for w in ['조정', '부진', '휴식', '재정비', '컨디션', '피로']): return '성적부진_조정'

    return '기타부상'

# 🌟 3. 심각도 점수 로직
def get_severity(text):
    text = str(text)
    if any(w in text for w in ['수술', '파열', '시즌아웃', '끊어져']): return 5
    if '골절' in text: return 4
    if any(w in text for w in ['손상', '염좌', '미세', '재활', '부분']): return 3
    if any(w in text for w in ['이탈', '통증', '결장', '불편']): return 2
    return 1

print("🔧 [부상 정밀 분석] 뉴스 제목에서 부상 종류를 분류하고 점수를 매깁니다...")

df['부상분류'] = df['뉴스기사제목'].apply(classify_injury)
df['심각도점수'] = df['뉴스기사제목'].apply(get_severity)
df['재검필요'] = df['뉴스기사제목'].apply(lambda x: 1 if '재검' in str(x) else 0)

print(df['부상분류'].value_counts())

df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"✅ 분석 완료! {OUTPUT_FILE}")