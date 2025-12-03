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
import yfinance as yf

# Tenta importar bibliotecas do Google (Modo Online)
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

# ==============================================================================
# 0. CONFIGURAÇÃO GERAL E ESTILO (GLOBAL)
# ==============================================================================
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

# --- SISTEMA DE LOGIN ---
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

# ==============================================================================
# 1. FUNÇÕES COMPARTILHADAS (DB, TEXTO, YAHOO)
# ==============================================================================
def conectar_gsheets():
    if not HAS_GOOGLE: return None
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        if "gcp_service_account" not in st.secrets: return None
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("DB_Valuation").sheet1 
        return sheet
    except Exception as e: return None

def safe_float(valor):
    if isinstance(valor, (int, float)): return float(valor)
    try: return float(str(valor).replace("R$", "").replace(" ", "").replace(".", "").replace(",", "."))
    except:
        try: return float(str(valor).replace("R$", "").replace(" ", "").replace(",", "."))
        except: return 0.0

@st.cache_data(ttl=10) 
def carregar_dados_db():
    sheet = conectar_gsheets()
    if sheet:
        try: return sheet.get_all_records()
        except: return []
    return []

def salvar_no_db(novo_dict):
    sheet = conectar_gsheets()
    if sheet:
        try:
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
        except Exception as e: st.error(f"Erro ao salvar: {e}"); return False
    return False

def deletar_do_db(indice_reverso):
    try:
        sheet = conectar_gsheets()
        if sheet:
            total_rows = len(sheet.get_all_values())
            row_to_delete = total_rows - indice_reverso
            sheet.delete_rows(row_to_delete)
            st.cache_data.clear()
            return True
    except: return False
    return False

@st.cache_data(ttl=300)
def obter_cotacao_atual(ticker):
    try:
        t = ticker.strip().upper()
        if not t.endswith(".SA") and len(t) <= 6: t = f"{t}.SA"
        hist = yf.Ticker(t).history(period="1d")
        if not hist.empty: return hist['Close'].iloc[-1]
        return None
    except: return None


################################################################################
# MÓDULO: VALUATION (Início)
################################################################################
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
            try: 
                ano = int(cols[0].text.strip())
                val = float(cols[1].text.strip().replace("R$", "").replace(",", "."))
                vals.append((ano, val)) 
            except: continue
    if not vals: return None
    vals.sort(key=lambda x: x[0], reverse=True)
    u5 = vals[:5]
    media = sum([v[1] for v in u5]) / len(u5)
    return {"media": media, "historico": u5}

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
        
        if d_data:
            dpa = d_data["media"]
            historico_raw = d_data["historico"]
        else:
            dpa = (float(g_tit("DY").replace(',','.').replace('%',''))/100)*p
            historico_raw = []

        g = round(math.sqrt(22.5* (p/pl) * vpa), 2) if pl>0 and vpa>0 else 0
        b = round(dpa/tb, 2)
        go = round(dpa/(tg-tc), 2)
        def cm(teto): return round(((teto - p) / p), 4) if teto > 0 else 0
        
        return {
            "Ticker": ticker.upper(), "Preço Atual": p, "DPA Est.": dpa, 
            "Graham": g, "Margem Graham": cm(g), 
            "Bazin": b, "Margem Bazin": cm(b), 
            "Gordon": go, "Margem Gordon": cm(go), 
            "Historico_Raw": historico_raw
        }
    except: return None
################################################################################
# MÓDULO: VALUATION (Fim)
################################################################################


################################################################################
# MÓDULO: MARKOWITZ (LÓGICA E VISUAL ATUALIZADOS) (Início)
################################################################################
def gerar_tabela_performance_v28(df_retornos, fator_anual):
    stats = []
    for ativo in df_retornos.columns:
        serie = df_retornos[ativo]
        ret_total_arquivo = (1 + serie).prod() - 1
        media_hist = serie.mean() * fator_anual
        p_12m = 12 if fator_anual == 12 else 252
        p_24m = 24 if fator_anual == 12 else 504
        ret_12m = (1 + serie.tail(p_12m)).prod() - 1 if len(serie) >= p_12m else np.nan
        ret_24m = (1 + serie.tail(p_24m)).prod() - 1 if len(serie) >= p_24m else np.nan

        stats.append({
            "Ativo": ativo,
            "Retorno Total do Arquivo": ret_total_arquivo * 100,
            "Média Histórica (Total)": media_hist * 100,
            "Últimos 12 Meses": ret_12m * 100 if not np.isnan(ret_12m) else None,
            "Últimos 24 Meses": ret_24m * 100 if not np.isnan(ret_24m) else None
        })
    return pd.DataFrame(stats)

def calc_portfolio(w, r, cov, rf):
    rp = np.sum(w * r)
    vp = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    sp = (rp - rf) / vp if vp > 0 else 0
    return rp, vp, sp

# Funções objetivo para otimização
def min_sp_obj(w, r, c, rf): return -calc_portfolio(w, r, c, rf)[2] # Maximizar Sharpe (Minimizar negativo)
def min_vol_obj(w, r, c, rf): return calc_portfolio(w, r, c, rf)[1] # Minimizar Volatilidade

def monte_carlo(mu_anual, vol_anual, valor_ini, aporte_mensal_ini, anos, inflacao_anual, n_sim=500):
    dt = 1/12
    steps = int(anos * 12)
    caminhos = np.zeros((n_sim, steps + 1))
    caminhos[:, 0] = valor_ini
    aporte_atual = aporte_mensal_ini
    for t in range(1, steps + 1):
        if t > 1 and (t-1) % 12 == 0:
            aporte_atual *= (1 + inflacao_anual)
        z = np.random.normal(0, 1, n_sim)
        drift = (mu_anual - 0.5 * vol_anual**2) * dt
        diffusion = vol_anual * np.sqrt(dt) * z
        caminhos[:, t] = caminhos[:, t-1] * np.exp(drift + diffusion) + aporte_atual
    return np.percentile(caminhos, 95, axis=0), np.percentile(caminhos, 50, axis=0), np.percentile(caminhos, 5, axis=0), steps

# --- NOVA FUNÇÃO DE TOOLTIP (HOVER) ---
def gerar_tooltip_detalhado(nome_ponto, ret, vol, shp, pesos, ativos):
    """Gera o tooltip HTML no formato solicitado."""
    tooltip = f"<b>{nome_ponto}</b><br>"
    tooltip += f"Ret: {ret:.1%}<br>"
    tooltip += f"Vol: {vol:.1%}<br>"
    tooltip += f"Shp: {shp:.2f}<br><br>"
    tooltip += "<b>Alocação:</b><br>"
    
    # Cria lista de (peso, ativo) e ordena decrescente pelo peso
    alocacao = sorted(zip(pesos, ativos), key=lambda x: x[0], reverse=True)
    
    for peso, ativo in alocacao:
        if peso >= 0.001: # Mostra apenas alocações relevantes (>0.1%)
            tooltip += f"{ativo}: {peso:.1%}<br>"
    return tooltip
################################################################################
# MÓDULO: MARKOWITZ (Fim)
################################################################################


# ==============================================================================
# INTERFACE E NAVEGAÇÃO
# ==============================================================================
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

# --- TELA INICIAL ---
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


################################################################################
# INTERFACE: VALUATION
################################################################################
elif opcao == "📊 Valuation (Ações)":
    st.title("📊 Valuation Fundamentalista")
    with st.container(border=True):
        st.subheader("1. Parâmetros de Entrada")
        c1, c2, c3 = st.columns(3)
        tb = c1.number_input("Taxa Bazin (Dec)", 0.01, 0.50, 0.08, format="%.2f", help="TMA")
        tg = c2.number_input("Taxa Gordon", 0.01, 0.50, 0.12, format="%.2f", help="Custo Capital")
        tc = c3.number_input("Cresc. g", 0.00, 0.10, 0.02, format="%.2f", help="Crescimento perpétuo")
        tickers = st.text_area("Tickers", "BBAS3, ITSA4, WEG3")
    
    if st.button("🔍 Calcular", type="primary"):
        lista = [t.strip() for t in tickers.split(',') if t.strip()]
        res_valuation = []
        res_historicos = {}
        
        bar = st.progress(0)
        for i, t in enumerate(lista):
            d = extrair_dados_valuation(t, tb, tg, tc)
            if d:
                hist = d.pop("Historico_Raw") 
                res_valuation.append(d)
                res_historicos[d["Ticker"]] = hist
            bar.progress((i+1)/len(lista))
            
        if res_valuation:
            df = pd.DataFrame(res_valuation)
            st.markdown("### 🎯 Resultados")
            fig = go.Figure()
            l = df['Ticker'].tolist()
            fig.add_trace(go.Bar(x=l, y=df['Preço Atual'], name='Atual', marker_color='#95a5a6', text=df['Preço Atual'], texttemplate='R$ %{y:.2f}'))
            fig.add_trace(go.Bar(x=l, y=df['Graham'], name='Graham', marker_color='#27ae60', text=df['Graham'], texttemplate='R$ %{y:.2f}'))
            fig.add_trace(go.Bar(x=l, y=df['Bazin'], name='Bazin', marker_color='#2980b9', text=df['Bazin'], texttemplate='R$ %{y:.2f}'))
            fig.add_trace(go.Bar(x=l, y=df['Gordon'], name='Gordon', marker_color='#9b59b6', text=df['Gordon'], texttemplate='R$ %{y:.2f}'))
            fig.update_layout(barmode='group', template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df, column_config={"Preço Atual": st.column_config.NumberColumn(format="R$ %.2f"), "DPA Est.": st.column_config.NumberColumn(format="R$ %.4f"), "Graham": st.column_config.NumberColumn(format="R$ %.2f"), "Bazin": st.column_config.NumberColumn(format="R$ %.2f"), "Gordon": st.column_config.NumberColumn(format="R$ %.2f"), "Margem Graham": st.column_config.NumberColumn(format="%.2f%%"), "Margem Bazin": st.column_config.NumberColumn(format="%.2f%%"), "Margem Gordon": st.column_config.NumberColumn(format="%.2f%%")}, use_container_width=True, hide_index=True)
            
            with st.expander("📂 Histórico de Dividendos (Ano a Ano)"):
                if res_historicos:
                    rows = []
                    all_years = set()
                    for ticker, hist_list in res_historicos.items():
                        if hist_list:
                            row = {"Ticker": ticker}
                            vals = [v for k,v in hist_list]
                            row["Média Usada"] = sum(vals)/len(vals) if vals else 0
                            for ano, valor in hist_list:
                                row[str(ano)] = valor
                                all_years.add(str(ano))
                            rows.append(row)
                    if rows:
                        cols_order = ["Ticker", "Média Usada"] + sorted(list(all_years), reverse=True)
                        df_hist_final = pd.DataFrame(rows).reindex(columns=cols_order)
                        st.dataframe(df_hist_final.style.format(precision=4, na_rep="-"), use_container_width=True)
        else: st.warning("Sem dados.")


################################################################################
# INTERFACE: MARKOWITZ (Visual Profissional Ajustado)
################################################################################
# ==============================================================================
# MÓDULO: MARKOWITZ (V41 - Visual Pizza Ajustado)
# ==============================================================================
elif opcao == "📉 Otimização (Markowitz)":
    st.title("📉 Otimizador de Carteira")
    
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        arquivo = c1.file_uploader("Upload Excel", type=['xlsx'])
        with c2:
            st.markdown("**Calibragem**")
            tipo_dados = st.radio("Conteúdo:", ["Preços Históricos (R$)", "Retornos Já Calculados (%)"])
            freq_option = st.selectbox("Freq:", ["Diário (252)", "Mensal (12)"])
            fator_anual = 252 if freq_option.startswith("Diário") else 12
    
    if 'otimizacao_feita' not in st.session_state: st.session_state.otimizacao_feita = False
    
    if arquivo:
        try:
            df = pd.read_excel(arquivo)
            # Lógica V28: Tratamento de Data e Ordenação
            first_col = df.iloc[:, 0]
            if not np.issubdtype(first_col.dtype, np.number):
                df = df.set_index(df.columns[0])
                try: df.index = pd.to_datetime(df.index, dayfirst=True)
                except: df.index = pd.to_datetime(df.index, dayfirst=True, errors='coerce')
            
            df.sort_index(ascending=True, inplace=True)
            
            col_num = df.select_dtypes(include=[np.number]).columns.tolist()
            sel = st.multiselect("Ativos:", options=df.columns, default=col_num)
            
            if len(sel)<2: st.error("Selecione 2+ ativos."); st.stop()
            
            df_ativos = df[sel].dropna()
            if tipo_dados.startswith("Preços"): 
                retornos = df_ativos.pct_change().dropna()
            else: 
                retornos = df_ativos
            
            # TABELA 1: Performance
            df_perf = gerar_tabela_performance(retornos, fator_anual)
            st.markdown("---")
            st.info("Confira os retornos calculados abaixo:")
            st.dataframe(df_perf.set_index("Ativo").style.format("{:.2f}%", na_rep="-"), use_container_width=True)
            
            cov_matrix = retornos.cov() * fator_anual
            media_historica = df_perf["Média Anualizada (Input Modelo)"].values / 100 
            
        except Exception as e: 
            st.error(f"Erro no arquivo: {e}")
            st.stop()
        
        with st.container(border=True):
            # TABELA 2: Configuração
            df_c = pd.DataFrame({
                "Ativo": sel, 
                "Peso Atual (%)": [round(100/len(sel), 2)] * len(sel),
                "Visão (%)": [round(m*100, 2) for m in media_historica], 
                "Min (%)": [0.0]*len(sel), 
                "Max (%)": [100.0]*len(sel)
            })
            cfg = st.data_editor(df_c, num_rows="fixed", hide_index=True, use_container_width=True)
            rf = st.number_input("Risk Free (%)", 10.0)/100
        
        if st.button("🚀 Otimizar", type="primary"):
            visoes = cfg["Visão (%)"].values/100
            pesos_user = cfg["Peso Atual (%)"].values/100
            
            if abs(sum(pesos_user) - 1.0) > 0.01: 
                 pesos_user = pesos_user / sum(pesos_user)

            b = [(r["Min (%)"]/100, r["Max (%)"]/100) for _, r in cfg.iterrows()]
            n = len(sel); w0 = np.ones(n)/n
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
            
            try:
                res = minimize(min_sp, w0, args=(visoes, cov_matrix, rf), method='SLSQP', bounds=b, constraints=cons)
                w = res.x; r_opt, v_opt, s_opt = calc_portfolio(w, visoes, cov_matrix, rf)
                r_u, v_u, _ = calc_portfolio(pesos_user, visoes, cov_matrix, rf)
                st.session_state.otimizacao_feita = True
                st.session_state.res = {
                    'sel': sel, 
                    'r_opt': r_opt, 'v_opt': v_opt, 's_opt': s_opt, 'w': w, 
                    'v': visoes, 'cov': cov_matrix, 'rf': rf, 
                    'r_u': r_u, 'v_u': v_u, 'pesos_user': pesos_user,
                    'bounds': b 
                }
            except: st.error("Erro matemático.")

        if st.session_state.otimizacao_feita:
            r = st.session_state.res
            st.markdown("---"); st.markdown("### 🏆 Resultado")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sharpe", f"{r['s_opt']:.2f}"); col2.metric("Retorno Esp.", f"{r['r_opt']:.1%}"); col3.metric("Risco", f"{r['v_opt']:.1%}")
            
            c_chart1, c_chart2 = st.columns([2,1])
            with c_chart1:
                # Lógica do Gráfico de Fronteira (V28 Visual Style)
                max_ret = max(r['v']); 
                if max_ret < r['r_opt']: max_ret = r['r_opt']*1.1
                if max_ret > 2.0: max_ret = 2.0
                tgs = np.linspace(0, max_ret, 40)
                vx, vy, txt = [], [], []
                for t in tgs:
                    res = minimize(min_vol, np.ones(len(r['sel']))/len(r['sel']), args=(r['v'], r['cov'], r['rf']), method='SLSQP', bounds=r['bounds'], constraints=({'type':'eq','fun':lambda x:np.sum(x)-1}, {'type':'eq','fun':lambda x:calc_portfolio(x,r['v'],r['cov'],r['rf'])[0]-t}))
                    if res.success:
                        ret, vol, _ = calc_portfolio(res.x, r['v'], r['cov'], r['rf'])
                        vx.append(vol); vy.append(ret); txt.append(gerar_hover_text("Curva", ret, vol, _, res.x, r['sel']))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vx, y=vy, mode='lines', name='Fronteira', line=dict(color='#3498db', width=3), hoverinfo='text', text=txt))
                fig.add_trace(go.Scatter(x=[r['v_opt']], y=[r['r_opt']], mode='markers', marker=dict(size=15, color='#f1c40f', line=dict(width=2, color='black')), name='Ideal', hoverinfo='text', text=gerar_hover_text("Ideal", r['r_opt'], r['v_opt'], r['s_opt'], r['w'], r['sel'])))
                fig.add_trace(go.Scatter(x=[r['v_u']], y=[r['r_u']], mode='markers', marker=dict(size=12, color='black', symbol='x'), name='Atual', hoverinfo='text', text=gerar_hover_text("Atual", r['r_u'], r['v_u'], _, r['pesos_user'], r['sel'])))
                
                fig.update_layout(title="Risco vs. Retorno", xaxis_title="Risco", yaxis_title="Retorno", template="plotly_white", xaxis=dict(tickformat=".1%"), yaxis=dict(tickformat=".1%"), height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with c_chart2:
                # --- GRÁFICO DE PIZZA ATUALIZADO (IGUAL SUA IMAGEM) ---
                fig_p = go.Figure(data=[go.Pie(
                    labels=r['sel'], 
                    values=r['w'], 
                    hole=.45, # Buraco do Donut
                    textinfo='percent', # Mostra % dentro da fatia
                    textposition='inside',
                    marker=dict(colors=['#0077b5', '#27ae60', '#c0392b', '#f1c40f', '#8e44ad']) # Cores sólidas
                )])
                
                fig_p.update_layout(
                    title="Alocação Ideal", 
                    height=400, 
                    showlegend=True, # Legenda ativada
                    legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5) # Legenda embaixo
                )
                st.plotly_chart(fig_p, use_container_width=True)
            
            st.markdown("### 🔮 Monte Carlo")
            c1, c2, c3 = st.columns(3)
            ini = c1.number_input("Inicial", 10000.0); aport = c2.number_input("Mensal", 1000.0); ano = c3.number_input("Anos", 10)
            
            if st.button("Simular"):
                o, m, p, s, t = monte_carlo(r['r_opt'], r['v_opt'], ini, aport, int(ano), 0.05)
                f = go.Figure(); x = np.linspace(0, int(ano), s+1)
                f.add_trace(go.Scatter(x=x, y=t, name='Teórico', line=dict(color='orange', dash='dot')))
                f.add_trace(go.Scatter(x=x, y=m, name='Esperado', line=dict(color='green')))
                f.add_trace(go.Scatter(x=x, y=p, name='Pessimista', line=dict(color='#abebc6', width=0), fill='tonexty'))
                f.add_trace(go.Scatter(x=x, y=usr_mid, mode='lines', name='Atual (Esperado)', line=dict(color='black', dash='dash'))) # Linha atual faltava aqui em cima
                
                f.update_layout(title="Crescimento Patrimonial", xaxis_title="Anos", yaxis_title="Patrimônio", template="plotly_white", yaxis=dict(tickprefix="R$ ", tickformat=",.0f"))
                st.plotly_chart(f, use_container_width=True)
                st.success(f"💰 **Final Estimado:** R$ {m[-1]:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))


################################################################################
# INTERFACE: CATÁLOGO (Estudos)
################################################################################
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

