import pandas as pd
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)
wait = WebDriverWait(driver, 10)

def scrape_kbo_active_pitchers_2026():
    try:
        url = "https://www.koreabaseball.com/Player/Search.aspx"
        driver.get(url)
        
        all_pitchers = []
        target_positions = ['투수']

        team_select_element = wait.until(EC.presence_of_element_located((By.ID, "cphContents_cphContents_cphContents_ddlTeam")))
        team_select = Select(team_select_element)
        team_names = [opt.text for opt in team_select.options if opt.text != "팀 선택"]

        print(f"🚀 2026년 투수 명단 수집 시작...")

        for team in team_names:
            print(f"⚾ [{team}] 수집 시작...")
            
            Select(driver.find_element(By.ID, "cphContents_cphContents_cphContents_ddlTeam")).select_by_visible_text(team)
            time.sleep(2)

            for page in range(1, 6):
                try:
                    page_btn_id = f"cphContents_cphContents_cphContents_ucPager_btnNo{page}"
                    page_btn = driver.find_element(By.ID, page_btn_id)
                    page_btn.click()
                    time.sleep(2)
                    
                    rows = driver.find_elements(By.CSS_SELECTOR, ".tEx tbody tr")
                    for row in rows:
                        cols = row.find_elements(By.TAG_NAME, "td")
                        if len(cols) > 3:
                            name = cols[1].text
                            pos = cols[3].text
                            
                            if pos in target_positions:
                                all_pitchers.append({
                                    '팀': team,
                                    '선수': name,
                                    '포지션': pos
                                })
                except:
                    break
        
        df = pd.DataFrame(all_pitchers)
        df = df.drop_duplicates(subset=['선수'], keep='first')
        
        output_path = "/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_entry_pitchers_2026.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"✨ 수집 완료! 총 {len(df)}명의 투수 명단 확보")
        print(f"📁 저장: {output_path}")

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
    finally:
        print("💡 창은 수동으로 닫아주세요.")

if __name__ == "__main__":
    scrape_kbo_active_pitchers_2026()