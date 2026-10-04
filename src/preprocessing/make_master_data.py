import pandas as pd
import numpy as np

years = [2021, 2022, 2023, 2024, 2025]
all_years_data = []

print("🔥 5개년 투수 마스터 데이터 병합 시작...\n")

def convert_ip(ip_str):
    """'12 1/3' → 12.33, '19' → 19.0 변환"""
    try:
        ip_str = str(ip_str).strip()
        if ' ' in ip_str:
            parts = ip_str.split(' ')
            whole = int(parts[0])
            frac = parts[1]
            num, den = frac.split('/')
            return whole + int(num) / int(den)
        return float(ip_str)
    except:
        return 0.0

for year in years:
    try:
        kbo_df = pd.read_csv(f'/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/kbo_pitchers_{year}_all.csv')
        statiz_df = pd.read_csv(f'/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/statiz_pitcher_stats_{year}.csv')

        # 1. 컬럼명 통일 먼저
        if '선수명' in kbo_df.columns:
            kbo_df.rename(columns={'선수명': '이름'}, inplace=True)
        if 'Name' in statiz_df.columns:
            statiz_df.rename(columns={'Name': '이름'}, inplace=True)
        if '팀명_스탯티즈' in statiz_df.columns:
            statiz_df.rename(columns={'팀명_스탯티즈': '팀명'}, inplace=True)

        # 2. 개명 선수 처리
        name_mapping = {
            '최채흥': '최지명',
            '조영우': '조이현',
            '김이환': '김도현',
            '김정빈': '김사윤',
            '김윤수': '김무신'
        }
        statiz_df['이름'] = statiz_df['이름'].replace(name_mapping)

        # 3. IP 분수 변환
        kbo_df['IP'] = kbo_df['IP'].apply(convert_ip)

        # 4. 병합
        merged_df = pd.merge(kbo_df, statiz_df, on=['이름', '팀명'], how='inner', suffixes=('_KBO', '_STATIZ'))
        merged_df['시즌'] = year

        all_years_data.append(merged_df)
        print(f"✅ {year}년 병합 완료: 총 {len(merged_df)}명 매칭됨")

    except Exception as e:
        print(f"🚨 {year}년 에러: {e}")

final_df = pd.concat(all_years_data, ignore_index=True)

# IP 기준 필터링 (25이닝 이상)
initial_count = len(final_df)
final_df = final_df[final_df['IP_KBO'] >= 25].copy()
print(f"\n✂️ 25이닝 미만 필터링: {initial_count}명 -> {len(final_df)}명")

# 외국인 투수 제거
foreign_pitchers = [
    '가라비토', '가빌리오', '다카하시', '데스파이네', '데이비슨', '라미레즈', '라이블리',
    '로젠버그', '루친스키', '메르세데스', '모리만도', '몽고메리', '발라조빅', '벨라스케즈',
    '수아레즈', '스트레일리', '스파크맨', '시라카와', '알드레드', '알칸타라', '에르난데스',
    '엘리아스', '와이드너', '치리노스', '카스타노', '쿠에바스', '톨허스트', '헤이수스',
    '감보아', '더모디', '라우어', '라일리', '레예스', '맥카티', '맥키니', '메디나',
    '미란다', '바리아', '벤자민', '뷰캐넌', '브랜든', '브룩스', '브리검', '산체스',
    '애플러', '앤더슨', '와이스', '요키시', '윌커슨', '잭로그', '카펜터', '콜어빈',
    '크로우', '파노니', '파슨스', '패트릭', '프랑코', '플럿코', '화이트', '후라도'
]

before = len(final_df)
final_df = final_df[~final_df['이름'].isin(foreign_pitchers)]
print(f"\n🧹 외국인 투수 제거: {before}명 -> {len(final_df)}명")

# 파생 변수
print("\n⚙️ 투수 지표 계산 중...")

final_df['K/9'] = np.where(final_df['IP_KBO'] > 0,
                           final_df['SO_KBO'] / final_df['IP_KBO'] * 9, 0).round(3)
final_df['BB/9'] = np.where(final_df['IP_KBO'] > 0,
                            final_df['BB_KBO'] / final_df['IP_KBO'] * 9, 0).round(3)
final_df['K/BB'] = np.where(final_df['BB_KBO'] > 0,
                            final_df['SO_KBO'] / final_df['BB_KBO'], 0).round(3)

# Z-Score 표준화
print("\n📊 Z-Score 표준화 중...")

z_score_cols = ['ERA', 'WHIP', 'BB_KBO', 'SO_KBO', 'K/9', 'BB/9', 'K/BB']

for col in z_score_cols:
    if col in final_df.columns:
        yearly_mean = final_df.groupby('시즌')[col].transform('mean')
        yearly_std = final_df.groupby('시즌')[col].transform('std')
        final_df[f'{col}_z'] = ((final_df[col] - yearly_mean) / yearly_std).round(3)

print("✅ Z-Score 표준화 완료")

save_path = '/Users/soyeon/Documents/GitHub/KBO-Pitcher-Prediction/data/KBO_Pitcher_Master_2021_2025.csv'
final_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(f"\n🎉 5개년 투수 마스터 데이터 완성!")
print(f"📁 저장 위치: {save_path}")