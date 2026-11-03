"""
[복원 1단계] 최종 부상 파일에서 빠진 투수 부상 기록 찾기 + 2군 출장 확인
=========================================================================
빠진 이유
  1) kbo_futures_game_crawling.py가 '결장 100일 이하' 기록만 결과 파일에 저장 → 100일 초과·시즌 아웃 기록이 사라짐
  2) 투수 명단을 은퇴 선수를 뺀 final 파일에서 가져옴 → 은퇴 선수 기록이 사라짐

여기서는 kbo_injury_master_dataset.csv(부상자 명단 원본) 중
  - Master(나이 수정본) 투수의 [이름 + 팀 + 시즌]과 일치하고
  - 최종 부상 파일(kbo_injury_master_final.csv)에 없는 기록만 고른다.
2군 출장 확인은 기존과 같이 결장 1~100일 기록만 한다 (코드는 기존 크롤러와 동일).
100일 초과·복귀 기록 없음은 장기 결장이므로 확인하지 않고 '2군출장상세' = 'X'로 둔다
(→ 기사가 없어도 비부상으로 판정되지 않음).

출력: kbo_injury_missing_futures.csv  → 다음은 R2_news_missing.py
중간에 멈추면 확인 못 한 건은 '미확인'으로 저장되니, 그 경우 다시 실행하면 됨
"""
import pandas as pd
import time
import io
import sys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager

DATA = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data"
RAW_FILE    = f"{DATA}/kbo_injury_master_dataset.csv"
MASTER_FILE = f"{DATA}/KBO_Pitcher_Master_2021_2025_agefix.csv"
DONE_FILE   = f"{DATA}/kbo_injury_master_final.csv"
OUTPUT_FILE = f"{DATA}/kbo_injury_missing_futures.csv"

raw = pd.read_csv(RAW_FILE)
master = pd.read_csv(MASTER_FILE)
done = pd.read_csv(DONE_FILE)

raw['선수'] = raw['선수'].astype(str).str.split('(').str[0].str.strip()
raw['연도'] = pd.to_datetime(raw['부상시작일']).dt.year
keys = set(zip(master['이름'], master['팀명'], master['시즌']))
raw = raw[[k in keys for k in zip(raw['선수'], raw['팀'], raw['연도'])]]

done_keys = set(zip(done['선수'], pd.to_datetime(done['부상시작일']).dt.strftime('%Y-%m-%d')))
missing = raw[[k not in done_keys for k in zip(raw['선수'], pd.to_datetime(raw['부상시작일']).dt.strftime('%Y-%m-%d'))]].copy()
missing = missing.drop(columns=['연도']).reset_index(drop=True)

short = missing['실제결장일수'].between(1, 100)
missing['2군출장상세'] = 'X'
missing['2군확인'] = '생략(100일 초과·복귀 기록 없음)'
missing.loc[short, '2군확인'] = '확인'
print(f"빠진 기록 {len(missing)}건 (선수 {missing['선수'].nunique()}명) — 2군 확인 대상 {short.sum()}건, 장기 결장 {(~short).sum()}건")

suspects_df = missing[short].copy()
if suspects_df.empty:
    missing.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    sys.exit(f"💾 저장: {OUTPUT_FILE}")

# 2. 셀레니움 설정 (창 띄움)
options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

futures_detail_list = []

try:
    for n, (index, row) in enumerate(suspects_df.iterrows(), 1):
        player_name = row['선수']
        team = row['팀']
        # 괄호 제거 로직 ('김도영(내야수)' -> '김도영')
        clean_name = str(player_name).split('(')[0].strip()
        start_date = pd.to_datetime(row['부상시작일'])
        end_date = pd.to_datetime(row['실제복귀일'])
        year = str(start_date.year)
        
        print(f"🔍 [{n}/{len(suspects_df)}] {team} {clean_name} 확인 중...", end=" ", flush=True)
        
        try:
            # Step 1: 선수 고유 ID 확보
            driver.get(f"https://www.koreabaseball.com/Player/Search.aspx?searchWord={clean_name}")
            time.sleep(1.5)
            
            player_id = ""
            if "Search.aspx" in driver.current_url:
                rows = driver.find_elements(By.CSS_SELECTOR, ".tEx tBody tr")
                for r in rows:
                    if team in r.text:
                        player_id = r.find_element(By.TAG_NAME, "a").get_attribute("href").split("playerId=")[1]
                        break
            else:
                player_id = driver.current_url.split("playerId=")[1].split("&")[0]
            
            if not player_id:
                print("❌ 검색불가"); futures_detail_list.append("검색불가"); continue

            # Step 2: 2군 일자별 기록실 순간이동
            driver.get(f"https://www.koreabaseball.com/Futures/Player/PitcherDaily.aspx?playerId={player_id}")
            time.sleep(1.5)

            # Step 3: 연도 선택
            try:
                select = Select(driver.find_element(By.ID, "cphContents_cphContents_cphContents_ddlYear"))
                select.select_by_value(year)
                time.sleep(2)
            except: pass

            # Step 4: 테이블 분석 및 상세 날짜 추출
            html_source = driver.page_source
            if "데이터가 없습니다" in html_source:
                print("✅ 통과"); futures_detail_list.append("X"); continue

            all_tables = pd.read_html(io.StringIO(html_source))
            valid_tables = []
            for t in all_tables:
                if '상대' in t.columns:
                    new_cols = list(t.columns); new_cols[0] = '날짜'; t.columns = new_cols
                    valid_tables.append(t)
            
            if not valid_tables:
                print("✅ 통과"); futures_detail_list.append("X"); continue

            # 데이터 병합 및 날짜 필터링
            df_futures = pd.concat(valid_tables, ignore_index=True)
            df_futures['날짜_정제'] = year + "." + df_futures['날짜'].astype(str)
            df_futures['날짜_정제'] = pd.to_datetime(df_futures['날짜_정제'], format='%Y.%m.%d', errors='coerce')
            
            played = df_futures[(df_futures['날짜_정제'] >= start_date) & (df_futures['날짜_정제'] <= end_date)]
            
            if not played.empty:
                # 결과 포맷: "n경기 (MM.DD, MM.DD)"
                dates_str = ", ".join(played['날짜_정제'].dt.strftime('%m.%d').tolist())
                detail = f"{len(played)}경기 ({dates_str})"
                print(f"🚨 적발! {detail}")
                futures_detail_list.append(detail)
            else:
                print("✅ 통과"); futures_detail_list.append("X")

        except Exception as e:
            print("⚠️ 에러"); futures_detail_list.append("에러")

except KeyboardInterrupt:
    print("\n🛑 사용자에 의해 조사가 중단되었습니다.")

finally:
    # 확인한 건은 결과를, 확인하지 못하고 중단된 건은 '미확인'으로 채움
    res = futures_detail_list + ['미확인'] * (len(suspects_df) - len(futures_detail_list))
    missing.loc[suspects_df.index, '2군출장상세'] = res
    missing.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장: {OUTPUT_FILE} ({len(missing)}건, 2군 확인 {len(futures_detail_list)}/{len(suspects_df)}건)")
    driver.quit()
