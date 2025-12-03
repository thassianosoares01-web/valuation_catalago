import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import math
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import hmac
from datetime import datetime
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import yfinance as yf

# ==========================================
# 0. CONFIGURAÇÃO E ESTILO
# ==========================================
st.set_page_config(page_title="Asset Manager Pro", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .card { background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
    div[data-testid="stMetricValue"] { font-size: 22px; }
    .ticker-header { color: #2c3e50; font-size: 24px; font-weight: bold; }
    .footer-link { color: #0077b5 !important; text-decoration: none; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- LOGIN ---
def check_password():
    if "password" not in st.secrets: return True
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else: st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False): return True
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("🔒 Acesso Restrito")
        st.text_input("Senha", type="password", on_change=password_entered, key="password")
        if "password_correct" in st.session_state: st.error("Senha incorreta.")
    return False

if not check_password(): st.stop()

# ==========================================
# 1. FUNÇÕES DE APOIO (DB E CALCULOS)
# ==========================================

# --- GOOGLE SHEETS ---
def conectar_gsheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open("DB_Valuation").sheet1 
    return sheet

def safe_float(valor):
    if isinstance(valor, (int, float)): return float(valor)
    try: return float(str(valor).replace("R$", "").replace(" ", "").replace(".", "").replace(",", "."))
    except:
        try: return float(str(valor).replace("R$", "").replace(" ", "").replace(",", "."))
        except: return 0.0

@st.cache_data(ttl=10) 
def carregar_dados_db():
    try:
        sheet = conectar_gsheets()
        return sheet.get_all_records()
    except: return []

def salvar_no_db(novo_dict):
    try:
        sheet = conectar_gsheets()
        linha = [
            novo_dict['Data'], novo_dict['Ticker'],
            str(novo_dict['Preço Justo']).replace(".", ","),
            str(novo_dict['Cotação Ref']).replace(".", ","),
            novo_dict['Método'], novo_dict['Tese'],
            json.dumps(novo_dict['Premissas'])
        ]
        sheet.append_row(linha)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Erro ao salvar: {e}"); return False

def deletar_do_db(indice_reverso):
    try:
        sheet = conectar_gsheets()
        total_rows = len(sheet.get_all_values())
        row_to_delete = total_rows - indice_reverso
        sheet.delete_rows(row_to_delete)
        st.cache_data.clear()
        return True
    except: return False

# --- YAHOO FINANCE ---
@st.cache_data(ttl=300)
def obter_cotacao_atual(ticker):
    try:
        t = ticker.strip().upper()
        if not t.endswith(".SA") and len(t) <= 6: t = f"{t}.SA"
        hist = yf.Ticker(t).history(period="1d")
        if not hist.empty: return hist['Close'].iloc[-1]
        return None
    except: return None

# --- VALUATION ---
def buscar_dividendos_ultimos_5_anos(ticker):
    url = f"https://playinvest.com.br/dividendos/{ticker.lower()}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200: return None
    except: return None
    soup = BeautifulSoup(r.text, 'html.parser')
    c = soup.find("div", class_="card featured-card per-year-chart")
    if not c: return None
    rows = c.find("table").find("tbody").find_all("tr")
    vals = []
    for r in rows:
        cols = r.find_all("td")
        if len(cols) >= 2:
            try: vals.append(float(cols[1].text.strip().replace("R$", "").replace(",", ".")))
            except: continue
    if not vals: return None
    vals.sort(reverse=True); u5 = vals[:5]
    return {"media": sum(u5)/len(u5), "historico": u5}

def extrair_dados_valuation(ticker, tb, tg, tc):
    url = f"https://investidor10.com.br/acoes/{ticker.lower()}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code!=200: return None
    except: return None
    soup = BeautifulSoup(r.text, 'html.parser')
    def g_tit(t):
        e = soup.find("span", title=t)
        return e.find_parent("div").find_next("div", class_="_card-body").span.text.strip() if e else "0"
    def g_val(l):
        e = soup.find(string=re.compile(fr"(?i){l}"))
        return e.find_parent().find_next("div", class_="value").span.text.strip() if e else "0"
    try:
        pl = float(g_tit("P/L").replace(',','.').replace('%',''))
        vpa = float(g_val("VPA").replace(',','.').replace('%',''))
        p = float(soup.find("div", class_="_card cotacao").find("div", class_="_card-body").span.text.strip().replace("R$", "").replace(",", "."))
        d_data = buscar_dividendos_ultimos_5_anos(ticker)
        dpa = d_data["media"] if d_data else (float(g_tit("DY").replace(',','.').replace('%',''))/100)*p
        g = round(math.sqrt(22.5* (p/pl) * vpa), 2) if pl>0 and vpa>0 else 0
        b = round(dpa/tb, 2)
        go = round(dpa/(tg-tc), 2)
        def cm(teto): return round(((teto - p) / p), 4) if teto > 0 else 0
        return {"Ticker": ticker.upper(), "Preço Atual": p, "DPA Est.": dpa, "Graham": g, "Margem Graham": cm(g), "Bazin": b, "Margem Bazin": cm(b), "Gordon": go, "Margem Gordon": cm(go), "Historico_Raw": []}
    except: return None

# --- MARKOWITZ (Lógica V28 Original) ---
def calcular_cagr(serie, fator_anual):
    # Lógica original da V28 (Retorno Simples Anualizado)
    # Para bater com o modelo V28, usamos a média simples * fator
    if len(serie) < 1: return 0.0
    return serie.mean() * fator_anual

def gerar_tabela_performance(df_retornos, fator_anual):
    stats = []
    for ativo in df_retornos.columns:
        serie = df_retornos[ativo]
        # Retorno Total (Simples acumulado para display)
        ret_total = (1 + serie).prod() - 1
        # Média Anualizada (Para Otimização - Lógica V28)
        media_anual = calcular_cagr(serie, fator_anual)
        
        stats.append({
            "Ativo": ativo,
            "Média Anualizada (Input Modelo)": media_anual * 100, 
            "Retorno Total do Período": ret_total * 100
        })
    return pd.DataFrame(stats)

def calc_portfolio(w, r, cov, rf):
    rp = np.sum(w * r)
    vp = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    sp = (rp - rf) / vp if vp > 0 else 0
    return rp, vp, sp

def min_sp(w, r, c, rf): return -calc_portfolio(w, r, c, rf)[2]
def min_vol(w, r, c, rf): return calc_portfolio(w, r, c, rf)[1]

def monte_carlo(mu, vol, ini, aporte, anos, inf, n=500):
    if np.isnan(mu) or np.isnan(vol): return [0]*12, [0]*12, [0]*12, 12, [0]*12
    dt = 1/12; steps = int(anos * 12)
    cam = np.zeros((n, steps+1)); cam[:,0] = ini
    teorico = np.zeros(steps+1); teorico[0] = ini
    taxa_m = (1+mu)**(1/12)-1; aporte_atual = aporte
    for t in range(1, steps+1):
        if t>1 and (t-1)%12==0: aporte_atual *= (1+inf)
        z = np.random.normal(0, 1, n)
        cam[:,t] = cam[:,t-1] * np.exp((mu-0.5*vol**2)*dt + vol*np.sqrt(dt)*z) + aporte_atual
        teorico[t] = teorico[t-1]*(1+taxa_m) + aporte_atual
    return np.percentile(cam, 95, axis=0), np.percentile(cam, 50, axis=0), np.percentile(cam, 5, axis=0), steps, teorico

def gerar_hover_text(nome, ret, vol, sharpe, pesos, ativos):
    t = f"<b>{nome}</b><br>Ret: {ret:.1%}<br>Vol: {vol:.1%}<br>Sharpe: {sharpe:.2f}<br>Alocação:<br>"
    for i, a in enumerate(ativos): 
        if pesos[i]>0.01: t+=f"{a}: {pesos[i]:.1%}<br>"
    return t

# ==========================================
# 3. INTERFACE E NAVEGAÇÃO
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2910/2910312.png", width=80)
st.sidebar.title("Asset Manager")

# ADMIN LOGIN
if "admin_logged" not in st.session_state: st.session_state.admin_logged = False
is_admin = st.sidebar.checkbox("🔓 Acesso Admin", value=st.session_state.admin_logged)
if is_admin and not st.session_state.admin_logged:
    senha = st.sidebar.text_input("Senha Admin:", type="password")
    if senha:
        if "password" in st.secrets and hmac.compare_digest(senha, st.secrets["password"]):
            st.session_state.admin_logged = True
            st.rerun()
        else: st.sidebar.error("Incorreto")
elif not is_admin: st.session_state.admin_logged = False

st.sidebar.markdown("---")
opcao = st.sidebar.radio("Navegação:", ["🏠 Início", "📊 Valuation (Ações)", "📉 Otimização (Markowitz)", "📚 Catálogo (Estudos)"])
st.sidebar.markdown("---")
st.sidebar.markdown('Dev: <a href="https://www.linkedin.com/in/thassianosoares/" target="_blank" class="footer-link">Thassiano Soares</a>', unsafe_allow_html=True)

if opcao == "🏠 Início":
    st.title("Asset Manager Pro")
    st.markdown("#### 🚀 Plataforma de Inteligência e Gestão de Ativos")
    st.markdown("""
        <a href="https://www.linkedin.com/in/thassianosoares/" target="_blank" style="text-decoration: none;">
            <div style="display: inline-flex; align-items: center; background-color: #0077b5; color: white; padding: 8px 16px; border-radius: 4px; font-family: sans-serif; font-weight: 600;">
                <span>Conectar no LinkedIn</span>
            </div>
        </a>
    """, unsafe_allow_html=True)
    st.divider()
    st.subheader("📺 Como usar a plataforma")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ") 
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **Valuation:** Graham, Bazin e Gordon.")
    with col2:
        st.info("📉 **Otimização:** Markowitz e Monte Carlo.")
    with col3:
        st.info("📚 **Catálogo:** Banco de teses (Google Sheets).")

elif opcao == "📊 Valuation (Ações)":
    st.title("📊 Valuation Fundamentalista")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        tb = c1.number_input("Taxa Bazin (Dec)", 0.01, 0.50, 0.08, format="%.2f", help="TMA")
        tg = c2.number_input("Taxa Gordon", 0.01, 0.50, 0.12, format="%.2f", help="Custo Capital")
        tc = c3.number_input("Cresc. g", 0.00, 0.10, 0.02, format="%.2f", help="Crescimento perpétuo")
        tickers = st.text_area("Tickers", "BBAS3, ITSA4, WEG3")
    if st.button("🔍 Calcular", type="primary"):
        lista = [t.strip() for t in tickers.split(',') if t.strip()]
        res = []; bar = st.progress(0)
        for i, t in enumerate(lista):
            d = extrair_dados_valuation(t, tb, tg, tc)
            if d: res.append(d)
            bar.progress((i+1)/len(lista))
        if res:
            df = pd.DataFrame(res)
            st.markdown("### Resultados")
            fig = go.Figure()
            l = df['Ticker'].tolist()
            fig.add_trace(go.Bar(x=l, y=df['Preço Atual'], name='Atual', marker_color='#95a5a6', text=df['Preço Atual'], texttemplate='R$ %{y:.2f}'))
            fig.add_trace(go.Bar(x=l, y=df['Graham'], name='Graham', marker_color='#27ae60', text=df['Graham'], texttemplate='R$ %{y:.2f}'))
            fig.add_trace(go.Bar(x=l, y=df['Bazin'], name='Bazin', marker_color='#2980b9', text=df['Bazin'], texttemplate='R$ %{y:.2f}'))
            fig.add_trace(go.Bar(x=l, y=df['Gordon'], name='Gordon', marker_color='#9b59b6', text=df['Gordon'], texttemplate='R$ %{y:.2f}'))
            fig.update_layout(barmode='group', template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, column_config={"Preço Atual": st.column_config.NumberColumn(format="R$ %.2f"), "DPA Est.": st.column_config.NumberColumn(format="R$ %.4f"), "Graham": st.column_config.NumberColumn(format="R$ %.2f"), "Bazin": st.column_config.NumberColumn(format="R$ %.2f"), "Gordon": st.column_config.NumberColumn(format="R$ %.2f"), "Margem Graham": st.column_config.NumberColumn(format="%.2f%%"), "Margem Bazin": st.column_config.NumberColumn(format="%.2f%%"), "Margem Gordon": st.column_config.NumberColumn(format="%.2f%%")}, use_container_width=True, hide_index=True)
        else: st.warning("Sem dados.")

# --- MARKOWITZ (V28 RESTAURADA) ---
elif opcao == "📉 Otimização (Markowitz)":
    st.title("📉 Otimizador de Carteira")
    
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        arquivo = c1.file_uploader("Upload Excel", type=['xlsx'])
        with c2:
            # Definição de variáveis ANTES do uso para evitar NameError
            tipo_dados = st.radio("Conteúdo:", ["Preços Históricos (R$)", "Retornos (%)"])
            freq_input = st.selectbox("Freq:", ["Diário (252)", "Mensal (12)"])
            # Lógica de Fator Anual da V28
            fator_anual = 252 if freq_input.startswith("Diário") else 12

    if 'otimizacao_feita' not in st.session_state: st.session_state.otimizacao_feita = False
    
    if arquivo:
        try:
            df_raw = pd.read_excel(arquivo)
            # Lógica de Datas V28
            first_col = df_raw.iloc[:, 0]
            if not np.issubdtype(first_col.dtype, np.number):
                df_raw = df_raw.set_index(df_raw.columns[0])
                try: df_raw.index = pd.to_datetime(df_raw.index, dayfirst=True)
                except: df_raw.index = pd.to_datetime(df_raw.index, dayfirst=True, errors='coerce')
            
            df_raw.sort_index(ascending=True, inplace=True)
            
            col_num = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            sel = st.multiselect("Ativos:", options=df_raw.columns, default=col_num)
            
            if len(sel)<2: st.error("Selecione 2+ ativos."); st.stop()
            
            df_ativos = df_raw[sel].dropna()
            if tipo_dados.startswith("Preços"): 
                retornos = df_ativos.pct_change().dropna()
            else: 
                retornos = df_ativos
            
            # Tabela de Performance
            df_perf = gerar_tabela_performance(retornos, fator_anual)
            st.markdown("---")
            st.info("Confira os retornos calculados abaixo:")
            st.dataframe(df_perf.set_index("Ativo").style.format("{:.2f}%", na_rep="-"), use_container_width=True)
            
            # Inputs do Modelo
            cov_matrix = retornos.cov() * fator_anual
            media_historica = df_perf["Média Anualizada (Input Modelo)"].values / 100 # Divide por 100 pq na tabela ta em %
            
        except Exception as e: 
            st.error(f"Erro no arquivo: {e}")
            st.stop()
        
        with st.container(border=True):
            df_c = pd.DataFrame({
                "Ativo": sel, 
                "Visão (%)": [round(m*100, 2) for m in media_historica], # Volta pra % para exibir
                "Min (%)": [0.0]*len(sel), 
                "Max (%)": [100.0]*len(sel)
            })
            cfg = st.data_editor(df_c, num_rows="fixed", hide_index=True, use_container_width=True)
            rf = st.number_input("Risk Free (%)", 10.0)/100
        
        if st.button("🚀 Otimizar", type="primary"):
            visoes = cfg["Visão (%)"].values/100
            pesos_user = np.ones(len(sel))/len(sel)
            b = [(r["Min (%)"]/100, r["Max (%)"]/100) for _, r in cfg.iterrows()]
            n = len(sel); w0 = np.ones(n)/n
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
            try:
                res = minimize(min_sp, w0, args=(visoes, cov_matrix, rf), method='SLSQP', bounds=b, constraints=cons)
                w = res.x; r_opt, v_opt, s_opt = calc_portfolio(w, visoes, cov_matrix, rf)
                r_u, v_u, _ = calc_portfolio(pesos_user, visoes, cov_matrix, rf)
                st.session_state.otimizacao_feita = True
                st.session_state.res = {'sel': sel, 'r_opt': r_opt, 'v_opt': v_opt, 's_opt': s_opt, 'w': w, 'v': visoes, 'cov': cov_matrix, 'r_u': r_u, 'v_u': v_u}
            except: st.error("Erro matemático.")

        if st.session_state.otimizacao_feita:
            r = st.session_state.res
            st.markdown("---"); st.markdown("### 🏆 Resultado")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sharpe", f"{r['s_opt']:.2f}"); col2.metric("Retorno Esp.", f"{r['r_opt']:.1%}"); col3.metric("Risco", f"{r['v_opt']:.1%}")
            c_chart1, c_chart2 = st.columns([2, 1])
            with c_chart1:
                max_ret = max(r['v']); 
                if max_ret < r['r_opt']: max_ret = r['r_opt']*1.1
                if max_ret > 2.0: max_ret = 2.0
                tgs = np.linspace(0, max_ret, 40)
                vx, vy, txt = [], [], []
                for t in tgs:
                    res = minimize(min_vol, np.ones(len(r['sel']))/len(r['sel']), args=(r['v'], r['cov'], rf), method='SLSQP', bounds=b, constraints=({'type':'eq','fun':lambda x:np.sum(x)-1}, {'type':'eq','fun':lambda x:calc_portfolio(x,r['v'],r['cov'],rf)[0]-t}))
                    if res.success:
                        ret, vol, _ = calc_portfolio(res.x, r['v'], r['cov'], rf)
                        vx.append(vol); vy.append(ret); txt.append(gerar_hover_text("Curva", ret, vol, _, res.x, r['sel']))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vx, y=vy, mode='lines', name='Fronteira', line=dict(color='#3498db', width=3), hoverinfo='text', text=txt))
                fig.add_trace(go.Scatter(x=[r['v_opt']], y=[r['r_opt']], mode='markers', marker=dict(size=15, color='#f1c40f', line=dict(width=2, color='black')), name='Ideal', hoverinfo='text', text=gerar_hover_text("Ideal", r['r_opt'], r['v_opt'], r['s_opt'], r['w'], r['sel'])))
                fig.add_trace(go.Scatter(x=[r['v_u']], y=[r['r_u']], mode='markers', marker=dict(size=12, color='black', symbol='x'), name='Atual', hoverinfo='text', text=gerar_hover_text("Atual", r['r_u'], r['v_u'], _, np.ones(len(r['sel']))/len(r['sel']), r['sel'])))
                fig.update_layout(title="Risco vs. Retorno", xaxis_title="Risco", yaxis_title="Retorno", template="plotly_white", xaxis=dict(tickformat=".1%"), yaxis=dict(tickformat=".1%"), height=400)
                st.plotly_chart(fig, use_container_width=True)
            with c_chart2:
                fig_p = go.Figure(data=[go.Pie(labels=r['sel'], values=r['w'], hole=.4)])
                fig_p.update_layout(title="Alocação Ideal", height=400, showlegend=False)
                fig_p.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_p, use_container_width=True)
            st.markdown("### 🔮 Projeção Monte Carlo")
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                inv_ini = c1.number_input("Inicial (R$)", 10000.0)
                aporte = c2.number_input("Mensal (R$)", 1000.0)
                anos = c3.number_input("Anos", 10)
                inflacao = c4.number_input("Inflação (%)", 5.0, format="%.2f") / 100
            if st.button("🎲 Simular", type="primary"):
                if np.isnan(res['r_opt']): st.error("Erro.")
                else:
                    opt_top, opt_mid, opt_low, steps, linha_teorica = monte_carlo(r['r_opt'], r['v_opt'], inv_ini, aporte, int(anos), inflacao)
                    x = np.linspace(0, int(anos), steps + 1)
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Scatter(x=x, y=linha_teorica, mode='lines', name='Teórico (Juros Compostos)', line=dict(color='#f1c40f', width=2, dash='dot')))
                    fig_sim.add_trace(go.Scatter(x=x, y=opt_mid, mode='lines', name='Ideal (Esperado)', line=dict(color='#27ae60', width=3)))
                    fig_sim.add_trace(go.Scatter(x=x, y=opt_low, mode='lines', name='Ideal (Pessimista)', line=dict(color='#abebc6', width=0), fill='tonexty'))
                    fig_sim.update_layout(title="Crescimento Patrimonial", xaxis_title="Anos", yaxis_title="Patrimônio", template="plotly_white", hovermode="x unified", separators=",.", yaxis=dict(tickprefix="R$ ", tickformat=",.0f"))
                    st.plotly_chart(fig_sim, use_container_width=True)
                    final_val = opt_mid[-1]
                    st.success(f"💰 **Patrimônio Estimado (Cenário Ideal):** R$ {final_val:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

elif opcao == "📚 Catálogo (Estudos)":
    st.title("📚 Diário de Valuation")
    
    if st.session_state.admin_logged:
        if 'temp_p' not in st.session_state: st.session_state.temp_p = {}
        with st.expander("📝 **[ADMIN] Novo Estudo**", expanded=True):
            c1, c2 = st.columns(2)
            tik = c1.text_input("Ticker").upper()
            met = c2.selectbox("Método", ["Graham", "Gordon", "DCF", "Bazin"])
            c3, c4 = st.columns(2)
            cot = c3.text_input("Ref (R$)", "0,00")
            jus = c4.text_input("Justo (R$)", "0,00")
            tese = st.text_area("Tese")
            c5, c6, c7 = st.columns([2,2,1])
            k = c5.text_input("Premissa"); v = c6.text_input("Valor")
            if c7.button("➕"): st.session_state.temp_p[k] = v
            if st.session_state.temp_p: st.write(st.session_state.temp_p)
            if st.button("💾 Salvar"):
                val_j = safe_float(jus); val_c = safe_float(cot)
                if tik:
                    salvar_no_db({"Data": datetime.now().strftime("%d/%m/%Y"), "Ticker": tik, "Preço Justo": val_j, "Cotação Ref": val_c, "Método": met, "Tese": tese, "Premissas": st.session_state.temp_p.copy()})
                    st.session_state.temp_p = {}; st.rerun()
    else:
        st.info("Modo Leitura (Público).")

    ldb = carregar_dados_db()
    if ldb:
        for i, item in enumerate(ldb[::-1]):
            try: p = json.loads(item['Premissas_JSON']) 
            except: p = {}
            pj = safe_float(item.get('Preco_Justo', 0)); pr = safe_float(item.get('Cotacao_Ref', 0))
            live = obter_cotacao_atual(item.get('Ticker')); cur = live if live else pr
            up = ((pj-cur)/cur)*100 if cur>0 else 0
            with st.container(border=True):
                ch1, ch2 = st.columns([5, 1])
                ch1.subheader(f"📊 {item.get('Ticker')} | {item.get('Metodo')}")
                if st.session_state.admin_logged:
                    if ch2.button("🗑️", key=f"del_{i}"): deletar_do_db(i); st.rerun()
                else: ch2.caption(item.get('Data'))
                st.divider()
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Ref", f"R$ {pr:.2f}"); k2.metric("Live", f"R$ {cur:.2f}"); k3.metric("Justo", f"R$ {pj:.2f}"); k4.metric("Upside", f"{up:.1f}%")
                with st.expander("Ver Tese"): 
                    st.info(item.get('Tese'))
                    if p: st.table(pd.DataFrame(list(p.items()), columns=['Item', 'Valor']))
                    fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=cur, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Margem"}, delta={'reference': pj}, gauge={'axis': {'range': [None, pj*1.5]}, 'bar': {'color': "gray"}, 'steps': [{'range': [0, pj], 'color': "#d4edda"}], 'threshold': {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': pj}}))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20)); st.plotly_chart(fig, use_container_width=True)
