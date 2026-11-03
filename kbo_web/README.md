# KBO 선수 성적 예측 웹앱

## 실행 순서

### 1. 준비
```bash
cd kbo_web
pip install -r requirements.txt
```

### 2. data 폴더에 CSV 파일 복사
```
kbo_web/data/
├── kbo_pitcher_master_final_2021_2025.csv   ← 투수 마스터 데이터
├── KBO_Master_Data_2021_2025.csv            ← 타자 마스터 데이터
└── kbo_injury_classified_final.csv          ← 부상 분류 데이터
```

### 3. 모델 학습 (최초 1회)
```bash
python model_train.py
```
→ `models/pitcher_ridge.pkl`, `models/batter_ridge.pkl` 생성됨

### 4. 서버 실행
```bash
python app.py
```
→ http://localhost:5000 에서 확인

---

## Render.com 배포

1. GitHub에 이 폴더 push (data 폴더 제외 — .gitignore에 추가)
2. Render → New Web Service → GitHub 연결
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app`
5. 데이터는 별도로 업로드하거나 GitHub LFS 사용

## 폴더 구조
```
kbo_web/
├── app.py              ← Flask 백엔드
├── model_train.py      ← 모델 학습 스크립트
├── requirements.txt
├── Procfile            ← Render/Heroku 배포용
├── data/               ← CSV 파일 (직접 복사 필요)
├── models/             ← 학습된 모델 저장 위치
└── templates/
    └── index.html      ← 프론트엔드
```
