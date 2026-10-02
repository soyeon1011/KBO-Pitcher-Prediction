from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from io import StringIO
import sys

options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

print("KBO 공식 홈페이지 기록실 접속 중...")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://www.koreabaseball.com/Record/Player/PitcherBasic/Basic1.aspx")
time.sleep(3)

# Select(driver.find_element(By.CSS_SELECTOR, "[id$='ddlSeason']")).select_by_value("2021")
# Select(driver.find_element(By.CSS_SELECTOR, "[id$='ddlSeason']")).select_by_value("2022")
# Select(driver.find_element(By.CSS_SELECTOR, "[id$='ddlSeason']")).select_by_value("2023")
# Select(driver.find_element(By.CSS_SELECTOR, "[id$='ddlSeason']")).select_by_value("2024")
Select(driver.find_element(By.CSS_SELECTOR, "[id$='ddlSeason']")).select_by_value("2025")
time.sleep(2)
Select(driver.find_element(By.CSS_SELECTOR, "[id$='ddlSeries']")).select_by_value("0")
time.sleep(3)

all_data = []

for team_idx in range(1, 11):
    team_select = Select(driver.find_element(By.CSS_SELECTOR, "[id$='ddlTeam']"))
    team_name = team_select.options[team_idx].text
    print(f"\n⚾ [{team_name}] 팀 수집 중...")
    team_select.select_by_index(team_idx)
    time.sleep(4)

    try:
        driver.find_element(By.LINK_TEXT, "1").click()
        time.sleep(3)
    except:
        pass

    for page in range(1, 6):
        html = driver.page_source
        if "<table" in html:
            tables = pd.read_html(StringIO(html))
            if len(tables) > 0:
                df = tables[0]
                df.columns = df.columns.str.replace('\n', ' ')
                first_player = df['선수명'].iloc[0] if not df.empty else "없음"
                all_data.append(df)
                print(f"  ✅ {page}페이지 긁기 성공 (ex. {first_player} 등)")

        try:
            next_btn = driver.find_element(By.LINK_TEXT, str(page + 1))
            next_btn.click()
            time.sleep(4)
        except:
            print(f"  👉 {team_name} 팀 끝! (총 {page}페이지)")
            break

driver.quit()

if not all_data:
    print("🚨 수집된 데이터가 없습니다.")
    sys.exit()

final_df = pd.concat(all_data, ignore_index=True)
final_df = final_df.drop_duplicates(subset=['선수명', '팀명'], keep='first')

# save_path = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_pitchers_2021_all.csv"
# save_path = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_pitchers_2022_all.csv"
# save_path = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_pitchers_2023_all.csv"
# save_path = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_pitchers_2024_all.csv"
save_path = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_pitchers_2025_all.csv"
final_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"\n🎉 완료! 총 [{len(final_df)}명]의 투수 데이터가 '{save_path}'에 저장되었습니다.")