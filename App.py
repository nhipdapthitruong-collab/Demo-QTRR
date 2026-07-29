import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Hệ thống QTRR Margin", page_icon="📈", layout="wide")
st.title("🛡️ Hệ thống Quản trị Rủi ro & Cấp Hạn mức Margin")
st.caption("Mô hình Quản trị Rủi ro Danh mục GDKQ")

# Khởi tạo tham số mặc định
if "params" not in st.session_state:
    st.session_state.params = {
        "TK1_MIN": 70e9, "TK2_MIN": 30e9, "TK3_MIN": 10e9, "TRADE_RATIO_MIN": 0.90,
        "BD1_MAX": 0.30, "BD2_MAX": 0.45, "BD3_MAX": 0.65, "MDD_FLAG": 0.35,
        "MARKET_BUFFER": 0.10, "VCSH": 3300e9, "CAP_VCSH_PCT": 0.10, "CAP_KLNY_PCT": 0.05,
        "PARTICIPATION_RATE": 0.10, "EXIT_DAYS": 5, "SAFETY_FACTOR": 1.0, "LOT_SIZE": 100,
        "STRESS_SHOCK": 0.25, "STRESS_HAIRCUT": 0.10,
    }

R_MATRIX = {
    "TK1": {"BD1": "R1", "BD2": "R1", "BD3": "R2", "BD4": "R3"},
    "TK2": {"BD1": "R1", "BD2": "R2", "BD3": "R3", "BD4": "R3"},
    "TK3": {"BD1": "R2", "BD2": "R3", "BD3": "R3", "BD4": "R4"},
    "TK4": {"BD1": "R3", "BD2": "R3", "BD3": "R4", "BD4": "R5"},
}
RM_MAP = {"R1": 0.50, "R2": 0.45, "R3": 0.40, "R4": 0.30, "R5": 0.20}

def process_qtrr(df_raw, df_phantich, params):
    results = []
    tickers = df_raw['Ticker'].unique()
    for ticker in tickers:
        df_t = df_raw[df_raw['Ticker'] == ticker].sort_values('Ngày')
        if len(df_t) < 60: continue
        pt_info = df_phantich[df_phantich['Ticker'] == ticker]
        kq_pt = pt_info['KQ Phân tích'].values[0] if len(pt_info) > 0 else "Không đạt"
        
        df_60, df_20 = df_t.tail(60), df_t.tail(20)
        mv60, av20 = df_60['GTGD'].median(), df_20['GTGD'].mean()
        base_liq = min(mv60, av20)
        trade_ratio = (df_60['Volume'] > 0).sum() / len(df_60)
        
        if base_liq >= params["TK1_MIN"]: tk_grade = "TK1"
        elif base_liq >= params["TK2_MIN"]: tk_grade = "TK2"
        elif base_liq >= params["TK3_MIN"]: tk_grade = "TK3"
        else: tk_grade = "TK4"
            
        prices = df_60['Adjusted Close'].values
        returns = np.log(prices[1:] / prices[:-1])
        vol60 = np.std(returns, ddof=1) * np.sqrt(252) if len(returns) > 0 else 0
        peaks = np.maximum.accumulate(prices)
        drawdowns = (prices - peaks) / peaks
        mdd60 = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        if vol60 < params["BD1_MAX"]: bd_grade = "BD1"
        elif vol60 < params["BD2_MAX"]: bd_grade = "BD2"
        elif vol60 < params["BD3_MAX"]: bd_grade = "BD3"
        else: bd_grade = "BD4"
            
        r_market = R_MATRIX[tk_grade][bd_grade]
        rm_rate = RM_MAP[r_market]
        
        is_blocked = (kq_pt == "Không đạt") or (trade_ratio < params["TRADE_RATIO_MIN"])
        rm_final = 0.0 if is_blocked else rm_rate
        r_final = "BLOCKED" if is_blocked else r_market
            
        median20 = df_20['Adjusted Close'].median()
        top5 = df_60.nlargest(5, 'Adjusted Close')
        vwap_top5 = (top5['Adjusted Close'] * top5['Volume']).sum() / top5['Volume'].sum() if top5['Volume'].sum() > 0 else median20
        max_price_chuan = max(median20 * (1 + params["MARKET_BUFFER"]), vwap_top5)
        p_ref = df_t['Adjusted Close'].iloc[-1]
        p_margin = min(p_ref, max_price_chuan)
        
        klny = df_t['KL niêm yết'].iloc[-1] if 'KL niêm yết' in df_t.columns else 10e6
        room_qty_vcsh = (params["VCSH"] * params["CAP_VCSH_PCT"]) / p_margin if p_margin > 0 else 0
        room_qty_klny = klny * params["CAP_KLNY_PCT"]
        room_val_liq = base_liq * params["PARTICIPATION_RATE"] * params["EXIT_DAYS"] * params["SAFETY_FACTOR"]
        room_qty_liq = room_val_liq / p_margin if p_margin > 0 else 0
        
        room_qty_raw = min(room_qty_vcsh, room_qty_klny, room_qty_liq) if not is_blocked else 0
        room_qty_final = np.floor(room_qty_raw / params["LOT_SIZE"]) * params["LOT_SIZE"]
        room_debt_final = room_qty_final * p_margin * rm_final
        
        results.append({
            "Ticker": ticker, "KQ Phân tích": kq_pt, "P_ref": p_ref,
            "BASE_LIQ (Tỷ)": round(base_liq / 1e9, 2), "TK_GRADE": tk_grade,
            "VOL60 (%)": round(vol60 * 100, 2), "BD_GRADE": bd_grade,
            "MDD60 (%)": round(mdd60 * 100, 2),
            "Cờ MDD": "🚩 Cảnh báo" if mdd60 >= params["MDD_FLAG"] else "OK",
            "R_MARKET": r_final, "Rm (%)": round(rm_final * 100, 1),
            "MaxPrice": round(max_price_chuan, 0), "P_margin": round(p_margin, 0),
            "Room Qty": int(room_qty_final), "Room Debt (Tỷ)": round(room_debt_final / 1e9, 2)
        })
    return pd.DataFrame(results)

@st.cache_data
def load_demo():
    np.random.seed(42)
    dates = pd.date_range(end='2026-04-15', periods=120)
    tickers = ["BSR", "PLX", "PVB", "PVC", "PVD", "SSI", "VND", "FPT", "MWG", "HPG"]
    raw = []
    for t in tickers:
        p_list = 30000 + np.cumsum(np.random.randn(120) * 500)
        v_list = np.random.randint(500000, 5000000, 120)
        for d, p, v in zip(dates, p_list, v_list):
            raw.append({"Ticker": t, "Ngày": d, "Adjusted Close": max(p, 1000), "Volume": v, "GTGD": p * v, "KL niêm yết": 500e6})
    return pd.DataFrame(raw), pd.DataFrame({"Ticker": tickers, "KQ Phân tích": ["Đạt", "Theo dõi", "Đạt", "Theo dõi", "Đạt", "Đạt", "Đạt", "Đạt", "Theo dõi", "Không đạt"]})

if "df_raw" not in st.session_state:
    st.session_state.df_raw, st.session_state.df_pt = load_demo()

df_results = process_qtrr(st.session_state.df_raw, st.session_state.df_pt, st.session_state.params)

# Giao diện
menu = st.sidebar.radio("Menu Chức năng:", ["📊 Dashboard", "🔍 Tra cứu Mã", "⚙️ Tham số QTRR", "💥 Stress Test"])

if menu == "📊 Dashboard":
    st.subheader("📊 Dashboard Tổng quan Danh mục Margin")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng số mã", len(df_results))
    c2.metric("Mã được cấp Margin", len(df_results[df_results['Rm (%)'] > 0]))
    c3.metric("Mã có cờ MDD", len(df_results[df_results['Cờ MDD'] == "🚩 Cảnh báo"]))
    c4.metric("Tổng Room Dư nợ", f"{df_results['Room Debt (Tỷ)'].sum():,.1f} Tỷ")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(df_results, names="R_MARKET", title="Phân bổ theo Nhóm R"), use_container_width=True)
    with col2:
        st.plotly_chart(px.scatter(df_results, x="BASE_LIQ (Tỷ)", y="Rm (%)", size="Room Debt (Tỷ)", color="R_MARKET", hover_name="Ticker", title="Tương quan Thanh khoản vs Rm"), use_container_width=True)
    st.dataframe(df_results, use_container_width=True)

elif menu == "🔍 Tra cứu Mã":
    ticker = st.selectbox("Chọn mã chứng khoán:", df_results['Ticker'].unique())
    row = df_results[df_results['Ticker'] == ticker].iloc[0]
    st.markdown(f"### Kết quả: **{ticker}** - Rm: **{row['Rm (%)']}%** - MaxPrice: **{row['MaxPrice']:,.0f}** - Room Debt: **{row['Room Debt (Tỷ)']} Tỷ**")
    df_t = st.session_state.df_raw[st.session_state.df_raw['Ticker'] == ticker]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_t['Ngày'], y=df_t['Adjusted Close'], name='Giá thị trường'))
    fig.add_trace(go.Scatter(x=df_t['Ngày'], y=[row['MaxPrice']]*len(df_t), name='MaxPrice', line=dict(dash='dash')))
    st.plotly_chart(fig, use_container_width=True)

elif menu == "⚙️ Tham số QTRR":
    st.subheader("⚙️ Điều chỉnh Tham số Vận hành")
    with st.form("p_form"):
        buf = st.slider("Market Risk Buffer (%):", 0, 30, int(st.session_state.params["MARKET_BUFFER"]*100)) / 100
        days = st.number_input("Số ngày thoát vị thế (ExitDays):", value=st.session_state.params["EXIT_DAYS"])
        if st.form_submit_button("Lưu & Tính Lại"):
            st.session_state.params["MARKET_BUFFER"] = buf
            st.session_state.params["EXIT_DAYS"] = days
            st.rerun()

elif menu == "💥 Stress Test":
    st.subheader("💥 Stress Test Kịch bản Thị trường Sụt giảm")
    shock = st.slider("Mức giảm giá thị trường (%):", 5, 50, 25) / 100
    df_st = df_results[df_results['Rm (%)'] > 0].copy()
    df_st['Shortfall (Tỷ)'] = np.maximum(0, df_st['Room Debt (Tỷ)'] - (df_st['Room Qty'] * df_st['P_margin'] * (1-shock) * 0.9 / 1e9))
    st.metric("Tổng thâm hụt dư nợ toàn danh mục", f"{df_st['Shortfall (Tỷ)'].sum():,.2f} Tỷ")
    st.dataframe(df_st[['Ticker', 'Rm (%)', 'Room Debt (Tỷ)', 'Shortfall (Tỷ)']], use_container_width=True)
