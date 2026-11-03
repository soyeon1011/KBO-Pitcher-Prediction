import pandas as pd
import time
import re
import sys
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# ── 경로 설정 ──────────────────────────────────────────────────────────────
INPUT_FILE  = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_injury_futures_result.csv"
OUTPUT_FILE = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_injury_naver_news.csv"

# ── 데이터 로딩 ────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(INPUT_FILE)
    df['부상시작일'] = pd.to_datetime(df['부상시작일'].astype(str).str.replace('.', '-', regex=False))
    print(f"✅ {len(df)}건 로드 완료")
except Exception as e:
    print(f"❌ 로드 실패: {e}"); sys.exit()

# ── 크롬 설정 ──────────────────────────────────────────────────────────────
options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
options.add_argument("--start-maximized")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ── 부상 키워드 ────────────────────────────────────────────────────────────
INJURY_KEYWORDS = [
    '부상', '통증', '말소', '이탈', '결장', '수술', '파열', '재활', '손상',
    '후유증', '소견', '골절', '타박상', '염좌', '불편',
    '어깨', '팔꿈치', '손목', '손가락', '전완', '굴곡근', '광배근',
    '허리', '옆구리', '척추', '요추', '디스크',
    '무릎', '발목', '종아리', '햄스트링', '허벅지',
    '맹장', '장염', '감기', '몸살', '가래톳',
]

SURNAMES = ['김', '이', '박', '최', '정', '강', '조', '윤', '장', '임',
            '한', '오', '서', '신', '권', '황', '안', '송', '류', '홍',
            '전', '고', '문', '양', '손', '배', '백', '허', '유', '남',
            '심', '원', '노', '하', '곽', '성', '차', '구', '우', '진',
            '엄', '채', '변', '나', '공']

FALSE_POSITIVE_WORDS = ['이탈', '이후', '이전', '이상', '이하', '이내', '이외',
                         '전날', '전반', '전문', '전체', '전부', '안정', '안타',
                         '강도', '강점', '고통', '고려', '구간', '구성']

# ── 필터 함수 ──────────────────────────────────────────────────────────────
def simple_filter(txt, player_name):
    """1순위용: 선수명 포함 + 부상 키워드 (이전 코드 방식)"""
    return player_name in txt and any(k in txt for k in INJURY_KEYWORDS) and 10 < len(txt) < 500

def strict_filter(txt, player_name, window=40):
    """2순위용: 거리제한 + 쉼표 + 다른선수이름 필터"""
    idx = txt.find(player_name)
    if idx == -1: return False
    after = txt[idx + len(player_name): idx + len(player_name) + window]
    if not any(k in after for k in INJURY_KEYWORDS): return False
    if ',' in after or '，' in after: return False
    for m in re.finditer(r'[' + ''.join(SURNAMES) + r'][가-힣]{1,3}', after):
        word = m.group()
        if word not in FALSE_POSITIVE_WORDS and word != player_name:
            return False
    return 5 < len(txt) < 400

def soft_filter(txt, player_name, window=40):
    """3순위용: 거리제한만"""
    idx = txt.find(player_name)
    if idx == -1: return False
    after = txt[idx + len(player_name): idx + len(player_name) + window]
    return any(k in after for k in INJURY_KEYWORDS) and 5 < len(txt) < 400

def search_and_collect(query, date_range, player_name, filter_fn):
    """검색 URL 구성 후 기사 수집"""
    start_dt, end_dt = date_range
    start_no_dot = start_dt.replace(".", "")
    end_no_dot   = end_dt.replace(".", "")
    url = (f"https://search.naver.com/search.naver?where=news&query={quote(query)}"
           f"&pd=3&ds={start_dt}&de={end_dt}"
           f"&nso=so%3Ar%2Cp%3Afrom{start_no_dot}to{end_no_dot}%2Ca%3Aall")
    driver.get(url)
    time.sleep(2.5)
    driver.execute_script("window.scrollTo(0, 800);")

    results = []
    elements = driver.find_elements(By.CSS_SELECTOR, "a.news_tit, .api_txt_lines, .news_dsc, .sds-comps-text")
    for el in elements:
        txt = el.text.strip().replace("\n", " ")
        if filter_fn(txt, player_name):
            results.append(txt)
    return results

# ── 메인 크롤링 ────────────────────────────────────────────────────────────
try:
    for index, row in df.iterrows():
        player_name = str(row['선수']).split('(')[0].strip()
        team        = str(row.get('팀', '')).strip()
        injury_date = row['부상시작일']

        # 날짜 범위 두 가지
        range_narrow = ((injury_date - pd.Timedelta(days=2)).strftime("%Y.%m.%d"),
                        (injury_date + pd.Timedelta(days=10)).strftime("%Y.%m.%d"))
        range_wide   = ((injury_date - pd.Timedelta(days=5)).strftime("%Y.%m.%d"),
                        (injury_date + pd.Timedelta(days=14)).strftime("%Y.%m.%d"))

        print(f"[{index+1:3d}/{len(df)}] {player_name}", end="  ", flush=True)

        found_info = []

        # ── 1순위: 팀명+선수명, 좁은 날짜, 단순 필터 (이전 코드 방식) ──────
        query1 = f"{team} {player_name} 부상" if team else f"{player_name} 부상"
        found_info = search_and_collect(query1, range_narrow, player_name, simple_filter)

        # ── 2순위: 선수명만, 넓은 날짜, 엄격 필터 ────────────────────────────
        if not found_info:
            query2 = f"{player_name} 부상"
            found_info = search_and_collect(query2, range_wide, player_name, strict_filter)

        # ── 3순위: 선수명만, 넓은 날짜, 완화 필터 ────────────────────────────
        if not found_info:
            found_info = search_and_collect(query2, range_wide, player_name, soft_filter)

        if found_info:
            combined = " || ".join(list(dict.fromkeys(found_info))[:3])  # 중복 제거 후 최대 3개
            df.at[index, '뉴스기사제목'] = combined
            print(f"✅ {combined[:60]}...")
        else:
            df.at[index, '뉴스기사제목'] = "관련 기사 없음"
            print("❌ 기사 없음")

except KeyboardInterrupt:
    print("\n🛑 중단 - 저장 중...")

finally:
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장 완료: {OUTPUT_FILE} ({len(df)}건)")
    driver.quit()