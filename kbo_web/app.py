import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")

PITCHER_CSV        = os.path.join(DATA_DIR, "KBO_Pitcher_Master_2021_2025_agefix.csv")
BATTER_CSV         = os.path.join(DATA_DIR, "KBO_Master_Data_2021_2025_final.csv")
INJURY_CSV         = os.path.join(DATA_DIR, "kbo_injury_master_final_all.csv")
PITCHER_ACTUAL_CSV = os.path.join(DATA_DIR, "statiz_pitcher_stats_2026.csv")
BATTER_ACTUAL_CSV  = os.path.join(DATA_DIR, "statiz_batter_stats_2026.csv")

PITCHER_MODEL = os.path.join(BASE_DIR, "models", "pitcher_model.pkl")
BATTER_MODEL  = os.path.join(BASE_DIR, "models", "batter_model.pkl")

# ── 지표 분류 정의 ─────────────────────────────────────────────────────────────
PITCHER_STATS = {
    "기본": ["이름", "팀명", "연도", "나이"],
    "성과": ["평균자책점", "ERA", "WHIP", "이닝", "IP", "삼진", "K", "볼넷", "BB",
             "피안타", "H", "피홈런", "HR"],
    "비율": ["K/9", "BB/9", "K/BB", "HR/9", "피안타율", "BABIP"],
    "세이버": ["FIP", "xFIP", "WAR", "ERA+"],
    "부상": ["부상분류", "심각도점수", "실제부상여부"],
    "기타": ["Park_Factor", "선발등판", "구원등판", "세이브", "홀드"],
}

BATTER_ATTACK_STATS = [
    "타율", "AVG", "출루율", "OBP", "장타율", "SLG", "OPS",
    "안타", "H", "홈런", "HR", "타점", "RBI", "득점", "R",
    "2루타", "2B", "3루타", "3B", "볼넷", "BB", "삼진", "K",
    "도루", "SB", "wOBA", "wRC+", "BB/K", "ISO",
]

BATTER_DEFENSE_STATS = [
    "수비율", "FLD%", "실책", "E", "보살", "PO", "보조아웃", "A",
    "UZR", "DRS", "RF/9", "수비WAR", "포지션",
]

# ── 보완점 제안 로직 ───────────────────────────────────────────────────────────
def suggest_pitcher(row: dict) -> list:
    tips = []
    era  = row.get("ERA") or row.get("평균자책점")
    whip = row.get("WHIP")
    kbb  = row.get("K/BB")
    bb9  = row.get("BB/9")
    k9   = row.get("K/9")
    fip  = row.get("FIP")
    war  = row.get("WAR")

    if era and float(era) > 4.5:
        tips.append({"icon": "⚠️", "title": "평균자책점(ERA) 개선 필요",
                     "body": f"ERA {era:.2f}는 리그 평균 이상입니다. 피안타율 및 피홈런 억제에 집중하세요."})
    if whip and float(whip) > 1.4:
        tips.append({"icon": "🎯", "title": "WHIP 개선 필요",
                     "body": f"WHIP {whip:.2f}는 이닝당 주자를 많이 허용하는 수준입니다. 제구력 향상이 핵심입니다."})
    if bb9 and float(bb9) > 3.5:
        tips.append({"icon": "🔴", "title": "볼넷(BB/9) 과다",
                     "body": f"9이닝당 볼넷 {bb9:.1f}개 — 제구 불안이 ERA를 끌어올리는 주요 원인입니다."})
    if k9 and float(k9) < 6.0:
        tips.append({"icon": "💪", "title": "탈삼진 능력 향상 권장",
                     "body": f"K/9 {k9:.1f}은 리그 하위권입니다. 구위/구종 다양화를 고려하세요."})
    if kbb and float(kbb) < 1.5:
        tips.append({"icon": "⚡", "title": "K/BB 비율 낮음",
                     "body": f"K/BB {kbb:.2f} — 삼진보다 볼넷이 많습니다. 스트라이크 비율을 높이세요."})
    if fip and era and float(fip) < float(era) - 0.5:
        tips.append({"icon": "📈", "title": "운이 작용한 ERA — 반등 가능성",
                     "body": f"FIP({fip:.2f}) < ERA({era:.2f}): 수비·운 영향이 컸습니다. 실제 실력은 더 좋습니다."})
    if war and float(war) < 0.5:
        tips.append({"icon": "🏋️", "title": "전체 기여도(WAR) 향상 필요",
                     "body": f"WAR {war:.1f}은 대체 선수 수준입니다. 이닝 소화 능력과 안정성이 필요합니다."})
    if not tips:
        tips.append({"icon": "✅", "title": "전반적으로 양호",
                     "body": "주요 지표들이 리그 평균 이상입니다. 현재 수준을 유지하세요."})
    return tips


def suggest_batter(row: dict) -> list:
    tips = []
    avg  = row.get("타율") or row.get("AVG")
    obp  = row.get("출루율") or row.get("OBP")
    slg  = row.get("장타율") or row.get("SLG")
    ops  = row.get("OPS")
    hr   = row.get("홈런") or row.get("HR")
    bbk  = row.get("BB/K")
    woba = row.get("wOBA")

    if avg and float(avg) < 0.250:
        tips.append({"icon": "⚠️", "title": "타율(AVG) 개선 필요",
                     "body": f"타율 {float(avg):.3f}은 리그 평균 이하입니다. 컨택 능력 향상이 필요합니다."})
    if obp and float(obp) < 0.310:
        tips.append({"icon": "🎯", "title": "출루율(OBP) 개선 필요",
                     "body": f"출루율 {float(obp):.3f} — 선구안을 향상시켜 볼넷 비율을 높이세요."})
    if slg and float(slg) < 0.350:
        tips.append({"icon": "💪", "title": "장타력 강화 권장",
                     "body": f"장타율 {float(slg):.3f}은 낮은 편입니다. 파워 훈련과 타격 기술 개선이 필요합니다."})
    if ops and float(ops) < 0.680:
        tips.append({"icon": "🔴", "title": "종합 공격력(OPS) 향상 필요",
                     "body": f"OPS {float(ops):.3f}은 주전 타자로서 부족한 수준입니다."})
    if bbk and float(bbk) < 0.3:
        tips.append({"icon": "⚡", "title": "선구안 향상 필요",
                     "body": f"BB/K {float(bbk):.2f} — 삼진이 너무 많습니다. 선구안 훈련을 강화하세요."})
    if not tips:
        tips.append({"icon": "✅", "title": "전반적으로 양호",
                     "body": "주요 공격 지표들이 리그 평균 이상입니다. 현재 폼을 유지하세요."})
    return tips


# ── 데이터 로드 ────────────────────────────────────────────────────────────────
def load_data():
    global df_pitcher, df_batter, df_injury
    global df_pitcher_actual, df_batter_actual
    global pitcher_model, batter_model

    df_pitcher = pd.DataFrame()
    df_batter  = pd.DataFrame()
    df_injury  = pd.DataFrame()
    df_pitcher_actual = pd.DataFrame()
    df_batter_actual  = pd.DataFrame()

    if os.path.exists(PITCHER_CSV):
        df_pitcher = pd.read_csv(PITCHER_CSV)
        print(f"✅ 투수 데이터 로드: {len(df_pitcher)}행, {len(df_pitcher.columns)}컬럼")
    else:
        print(f"⚠️  투수 CSV 없음: {PITCHER_CSV}")

    if os.path.exists(BATTER_CSV):
        df_batter = pd.read_csv(BATTER_CSV)
        print(f"✅ 타자 데이터 로드: {len(df_batter)}행, {len(df_batter.columns)}컬럼")
    else:
        print(f"⚠️  타자 CSV 없음: {BATTER_CSV}")

    if os.path.exists(INJURY_CSV):
        df_injury = pd.read_csv(INJURY_CSV)
        print(f"✅ 부상 데이터 로드: {len(df_injury)}행")

    if os.path.exists(PITCHER_ACTUAL_CSV):
        df_pitcher_actual = pd.read_csv(PITCHER_ACTUAL_CSV)
        print(f"✅ 투수 2026 실제 성적 로드: {len(df_pitcher_actual)}행")
    if os.path.exists(BATTER_ACTUAL_CSV):
        df_batter_actual = pd.read_csv(BATTER_ACTUAL_CSV)
        print(f"✅ 타자 2026 실제 성적 로드: {len(df_batter_actual)}행")

    # 모델 로드
    pitcher_model = None
    batter_model  = None
    if os.path.exists(PITCHER_MODEL):
        with open(PITCHER_MODEL, "rb") as f:
            pitcher_model = pickle.load(f)
        print("✅ 투수 Ridge 모델 로드")
    if os.path.exists(BATTER_MODEL):
        with open(BATTER_MODEL, "rb") as f:
            batter_model = pickle.load(f)
        print("✅ 타자 Ridge 모델 로드")


def get_name_col(df):
    for c in ["이름", "선수", "선수명", "name"]:
        if c in df.columns:
            return c
    return df.columns[0]


def get_year_col(df):
    for c in ["연도", "시즌", "year", "Year"]:
        if c in df.columns:
            return c
    return None


def df_to_records(df):
    """NaN을 None으로 변환하여 JSON 직렬화 가능하게"""
    return json.loads(df.where(pd.notnull(df), None).to_json(orient="records", force_ascii=False))


# ── API 엔드포인트 ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/players")
def all_players():
    """전체 선수 목록 (검색 자동완성용)"""
    players = []
    if not df_pitcher.empty:
        nc = get_name_col(df_pitcher)
        for name in df_pitcher[nc].dropna().unique():
            players.append({"name": str(name), "type": "pitcher"})
    if not df_batter.empty:
        nc = get_name_col(df_batter)
        for name in df_batter[nc].dropna().unique():
            # 투수에 없는 경우만 추가
            pitcher_names = {p["name"] for p in players}
            if str(name) not in pitcher_names:
                players.append({"name": str(name), "type": "batter"})
    players.sort(key=lambda x: x["name"])
    return jsonify(players)


@app.route("/api/player/<name>")
def player_data(name):
    """선수 전체 데이터 반환"""
    result = {"name": name, "type": None, "history": [], "injury": [],
              "attack": [], "defense": [], "suggestions": []}

    # 투수 먼저
    if not df_pitcher.empty:
        nc = get_name_col(df_pitcher)
        rows = df_pitcher[df_pitcher[nc] == name]
        if not rows.empty:
            result["type"] = "pitcher"
            yc = get_year_col(rows)
            if yc:
                rows = rows.sort_values(yc)
            result["history"] = df_to_records(rows)
            # 최근 연도 기준 보완점 제안
            latest = rows.iloc[-1].to_dict()
            result["suggestions"] = suggest_pitcher(latest)

    # 타자
    if result["type"] is None and not df_batter.empty:
        nc = get_name_col(df_batter)
        rows = df_batter[df_batter[nc] == name]
        if not rows.empty:
            result["type"] = "batter"
            yc = get_year_col(rows)
            if yc:
                rows = rows.sort_values(yc)
            all_records = df_to_records(rows)
            result["history"] = all_records

            # 공격/수비 컬럼 분리
            attack_cols  = [c for c in rows.columns if c in BATTER_ATTACK_STATS]
            defense_cols = [c for c in rows.columns if c in BATTER_DEFENSE_STATS]
            id_cols = [get_name_col(rows), get_year_col(rows), "팀명"] if get_year_col(rows) else [get_name_col(rows), "팀명"]
            id_cols = [c for c in id_cols if c and c in rows.columns]

            result["attack"]  = df_to_records(rows[id_cols + attack_cols])
            result["defense"] = df_to_records(rows[id_cols + defense_cols]) if defense_cols else []

            latest = rows.iloc[-1].to_dict()
            result["suggestions"] = suggest_batter(latest)

    # 부상 이력
    if not df_injury.empty:
        inj_nc = get_name_col(df_injury)
        inj_rows = df_injury[df_injury[inj_nc].astype(str).str.split("(").str[0].str.strip() == name]
        if not inj_rows.empty:
            result["injury"] = df_to_records(inj_rows[["선수", "부상시작일", "부상분류", "심각도점수", "신뢰도"]
                                                        if all(c in inj_rows.columns for c in ["선수","부상시작일","부상분류","심각도점수","신뢰도"])
                                                        else inj_rows.columns[:6]])

    return jsonify(result)


@app.route("/api/predict/<name>")
def predict(name):
    """Ridge 모델로 다음 시즌 성적 예측"""
    # 투수 여부 확인
    is_pitcher = False
    latest_row = None

    if not df_pitcher.empty:
        nc = get_name_col(df_pitcher)
        rows = df_pitcher[df_pitcher[nc] == name]
        if not rows.empty:
            is_pitcher = True
            yc = get_year_col(rows)
            latest_row = rows.sort_values(yc).iloc[-1] if yc else rows.iloc[-1]

    if not is_pitcher and not df_batter.empty:
        nc = get_name_col(df_batter)
        rows = df_batter[df_batter[nc] == name]
        if not rows.empty:
            yc = get_year_col(rows)
            latest_row = rows.sort_values(yc).iloc[-1] if yc else rows.iloc[-1]

    if latest_row is None:
        return jsonify({"error": "선수를 찾을 수 없습니다."}), 404

    model = pitcher_model if is_pitcher else batter_model
    if model is None:
        return jsonify({"error": "모델 파일이 없습니다. model_train.py를 먼저 실행하세요.", "prediction": None})

    try:
        feature_cols = model["features"]
        le = model.get("le")

        # 마스터 CSV에 부상 컬럼이 없는 경우 injury CSV에서 보충
        INJURY_COLS = ['부상발생여부', '부상횟수', '최대결장일수', '최대심각도', '최고신뢰도점수', '대표부상분류']
        if not df_injury.empty and any(c not in latest_row.index for c in INJURY_COLS):
            inj_nc = get_name_col(df_injury)
            player_name = latest_row.get(get_name_col(df_pitcher if is_pitcher else df_batter), name)
            latest_year = latest_row.get(get_year_col(df_pitcher if is_pitcher else df_batter), None)
            inj_player = df_injury[df_injury[inj_nc].astype(str).str.split("(").str[0].str.strip() == str(player_name)]
            # 해당 연도 부상 데이터
            if latest_year is not None and '연도' in inj_player.columns:
                inj_year = inj_player[inj_player['연도'] == latest_year]
                if inj_year.empty:
                    inj_year = inj_player
            else:
                inj_year = inj_player

            if not inj_year.empty:
                injury_supplement = {
                    '부상발생여부': 1,
                    '부상횟수': len(inj_year),
                    '최대결장일수': inj_year['결장일수'].max() if '결장일수' in inj_year.columns else 0,
                    '최대심각도': inj_year['심각도점수'].max() if '심각도점수' in inj_year.columns else 0,
                    '최고신뢰도점수': inj_year['신뢰도'].max() if '신뢰도' in inj_year.columns else 0,
                    '대표부상분류': inj_year.loc[inj_year['결장일수'].idxmax(), '부상분류'] if '결장일수' in inj_year.columns and '부상분류' in inj_year.columns else '없음',
                }
            else:
                injury_supplement = {c: 0 for c in INJURY_COLS}
                injury_supplement['대표부상분류'] = '없음'

            latest_row = latest_row.copy()
            for k, v in injury_supplement.items():
                if k not in latest_row.index:
                    latest_row[k] = v

        # feature_cols 중 없는 컬럼은 0으로 채움
        row_data = latest_row.reindex(feature_cols, fill_value=0)

        # 대표부상분류 인코딩
        if le and '대표부상분류' in row_data.index:
            known = set(le.classes_)
            val = str(row_data['대표부상분류'])
            row_data['대표부상분류'] = le.transform([val if val in known else le.classes_[0]])[0]

        X = pd.to_numeric(row_data, errors='coerce').fillna(0).values.reshape(1, -1)
        pred = float(model["model"].predict(X)[0])
        target = model["target"]

        # 2026 실제 성적 조회
        actual = None
        actual_label = None
        if is_pitcher and not df_pitcher_actual.empty:
            row_a = df_pitcher_actual[df_pitcher_actual["Name"] == name]
            if not row_a.empty:
                actual = round(float(row_a.iloc[0]["ERA"]), 2)
                actual_label = "2026 실제 ERA"
        elif not is_pitcher and not df_batter_actual.empty:
            row_a = df_batter_actual[df_batter_actual["Name"] == name]
            if not row_a.empty:
                actual = round(float(row_a.iloc[0]["AVG"]), 3)
                actual_label = "2026 실제 타율"

        return jsonify({
            "prediction": round(pred, 4),
            "target": target,
            "type": "pitcher" if is_pitcher else "batter",
            "actual": actual,
            "actual_label": actual_label,
        })
    except Exception as e:
        return jsonify({"error": str(e), "prediction": None})


@app.route("/api/columns/<player_type>")
def get_columns(player_type):
    """투수/타자 컬럼 목록 반환"""
    if player_type == "pitcher" and not df_pitcher.empty:
        return jsonify(list(df_pitcher.columns))
    elif player_type == "batter" and not df_batter.empty:
        return jsonify(list(df_batter.columns))
    return jsonify([])


# ── 실행 ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_data()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
else:
    # Gunicorn 등 프로덕션 서버용
    load_data()
