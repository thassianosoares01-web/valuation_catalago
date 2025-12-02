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
    """Converte R$ 1.000,00 ou 1000.00 para float puro."""
    if isinstance(valor, (int, float)): return float(valor)
    try:
        return float(str(valor).replace("R$", "").replace(" ", "").replace(".", "").replace(",", "."))
    except:
        try: return float(str(valor).replace("R$", "").replace(" ", "").replace(",", "."))
        except: return 0.0

@st.cache_data(ttl=10) 
def carregar_dados_db():
    try:
        sheet = conectar_gsheets()
        dados = sheet.get_all_records()
        return dados
    except Exception as e: return []

def salvar_no_db(novo_dict):
    try:
        sheet = conectar_gsheets()
        linha = [
            novo_dict['Data'],
            novo_dict['Ticker'],
            str(novo_dict['Preço Justo']).replace(".", ","),
            str(novo_dict['Cotação Ref']).replace(".", ","),
            novo_dict['Método'],
            novo_dict['Tese'],
            json.dumps(novo_dict['Premissas'])
        ]
        sheet.append_row(linha)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Erro ao salvar: {e}")
        return False

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
        resposta = requests.get(url, headers=headers, timeout=5)
        if resposta.status_code != 200: return None
    except: return None
    soup = BeautifulSoup(resposta.text, 'html.parser')
    container = soup.find("div", class_="card featured-card per-year-chart")
    if not container: return None
    tabela = container.find("table")
    if not tabela: return None
    linhas = tabela.find("tbody").find_all("tr")
    dados = []
    for linha in linhas:
        colunas = linha.find_all("td")
        if len(colunas) >= 2:
            try:
                ano = int(colunas[0].text.strip())
                valor = float(colunas[1].text.strip().replace("R$", "").replace(",", "."))
                dados.append((ano, valor))
            except: continue
    if not dados: return None
    dados.sort(key=lambda x: x[0], reverse=True)
    ultimos_5 = dados[:5]
    if not ultimos_5: return None
    media = sum([v for _, v in ultimos_5]) / len(ultimos_5)
    return {"media": round(media, 4), "historico": ultimos_5}

def extrair_dados_valuation(ticker, taxa_bazin, taxa_gordon, taxa_crescimento):
    url = f"https://investidor10.com.br/acoes/{ticker.lower()}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resposta = requests.get(url, headers=headers, timeout=5)
        if resposta.status_code != 200: return None
    except: return None
    soup = BeautifulSoup(resposta.text, 'html.parser')
    def get_text(soup, title):
        el = soup.find("span", title=title)
        if el:
            body = el.find_parent("div").find_next("div", class_="_card-body")
            val = body.find("span") if body else None
            return val.text.strip().replace('%', '').replace(',', '.') if val else "0"
        return "0"
    def get_val_by_label(soup, label):
        el = soup.find(string=re.compile(fr"(?i){label}"))
        if el:
            val = el.find_parent().find_next("div", class_="value")
            return val.span.text.strip().replace('%', '').replace(',', '.') if val else "0"
        return "0"
    try:
        pl = float(get_text(soup, "P/L"))
        dy = float(get_text(soup, "DY"))
        vpa = float(get_val_by_label(soup, "VPA"))
        cotacao = soup.find("div", class_="_card cotacao")
        preco = float(cotacao.find("div", class_="_card-body").span.text.strip().replace("R$", "").replace(",", ".")) if cotacao else 0.0
        dados_divs = buscar_dividendos_ultimos_5_anos(ticker)
        historico_raw = []
        if dados_divs:
            dpa = dados_divs["media"]
            historico_raw = dados_divs["historico"]
        else:
            dpa = (dy / 100) * preco 
            historico_raw = []
        preco_bazin = round(dpa / taxa_bazin, 2) if dpa > 0 else 0
        lpa = round(preco / pl, 2) if pl > 0 else 0
        preco_graham = round(math.sqrt(22.5 * lpa * vpa), 2) if lpa > 0 and vpa > 0 else 0
        taxa_liq = taxa_gordon - taxa_crescimento
        preco_gordon = round(dpa / taxa_liq, 2) if dpa > 0 and taxa_liq > 0 else 0
        def calc_margem(teto): return round(((teto - preco) / preco), 4) if teto > 0 else 0
        return {
            "Ticker": ticker.upper(), "Preço Atual": preco, "DPA Est.": dpa,
            "Graham": preco_graham, "Margem Graham": calc_margem(preco_graham),
            "Bazin": preco_bazin, "Margem Bazin": calc_margem(preco_bazin),
            "Gordon": preco_gordon, "Margem Gordon": calc_margem(preco_gordon),
            "Historico_Raw": historico_raw
        }
    except: return None

# --- MARKOWITZ ---
def calcular_cagr(serie, fator_anual):
    if len(serie) < 1: return 0.0
    retorno_total = (1 + serie).prod()
    n = len(serie)
    if fator_anual == 1: return retorno_total - 1
    expoente = fator_anual / n
    try: return (retorno_total ** expoente) - 1
    except: return 0.0

def gerar_tabela_performance(df_retornos, fator_anual):
    stats = []
    for ativo in df_retornos.columns:
        serie = df_retornos[ativo]
        ret_total = calcular_cagr(serie, fator_anual)
        p_12m = 12 if fator_anual == 12 else 252
        p_24m = 24 if fator_anual == 12 else 504
        ret_12m = calcular_cagr(serie.tail(p_12m), fator_anual) if len(serie) >= p_12m else np.nan
        ret_24m = calcular_cagr(serie.tail(p_24m), fator_anual) if len(serie) >= p_24m else np.nan
        ret_abs = (1 + serie).prod() - 1
        stats.append({
            "Ativo": ativo, "Retorno Total (Arquivo)": ret_abs * 100,
            "Média Histórica (Total)": ret_total * 100,
            "Últimos 12 Meses": ret_12m * 100 if not np.isnan(ret_12m) else None,
            "Últimos 24 Meses": ret_24m * 100 if not np.isnan(ret_24m) else None
        })
    return pd.DataFrame(stats)

def calc_portfolio(w, r, cov, rf):
    rp = np.sum(w * r)
    vp = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    sp = (rp - rf) / vp if vp > 0 else 0
    return rp, vp, sp

def min_sp(w, r, c, rf): return -calc_portfolio(w, r, c, rf)[2]
def min_vol(w, r, c, rf): return calc_portfolio(w, r, c, rf)[1]

def monte_carlo(mu_anual, vol_anual, valor_ini, aporte_mensal_ini, anos, inflacao_anual, n_sim=500):
    if np.isnan(mu_anual) or np.isnan(vol_anual) or vol_anual == 0:
        return np.zeros(12), np.zeros(12), np.zeros(12), 12, np.zeros(12)
    dt = 1/12; steps = int(anos * 12)
    caminhos = np.zeros((n_sim, steps + 1)); caminhos[:, 0] = valor_ini
    aporte_atual = aporte_mensal_ini
    linha_teorica = np.zeros(steps + 1); linha_teorica[0] = valor_ini
    taxa_mensal_equiv = (1 + mu_anual)**(1/12) - 1
    aporte_teorico = aporte_mensal_ini
    for t in range(1, steps + 1):
        if t > 1 and (t-1) % 12 == 0: 
            aporte_atual *= (1 + inflacao_anual)
            aporte_teorico *= (1 + inflacao_anual)
        z = np.random.normal(0, 1, n_sim)
        drift = (mu_anual - 0.5 * vol_anual**2) * dt
        diffusion = vol_anual * np.sqrt(dt) * z
        caminhos[:, t] = caminhos[:, t-1] * np.exp(drift + diffusion) + aporte_atual
        linha_teorica[t] = linha_teorica[t-1] * (1 + taxa_mensal_equiv) + aporte_teorico
    return np.percentile(caminhos, 95, axis=0), np.percentile(caminhos, 50, axis=0), np.percentile(caminhos, 5, axis=0), steps, linha_teorica

def gerar_hover_text(nome, ret, vol, sharpe, pesos, ativos):
    texto = f"<b>{nome}</b><br>Retorno: {ret:.1%}<br>Risco: {vol:.1%}<br>Sharpe: {sharpe:.2f}<br><br><b>Alocação:</b><br>"
    for i, ativo in enumerate(ativos):
        if pesos[i] > 0.01: texto += f"{ativo}: {pesos[i]:.1%}<br>"
    return texto

# ==========================================
# 3. INTERFACE E NAVEGAÇÃO
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2910/2910312.png", width=80)
st.sidebar.title("Asset Manager Pro")
st.sidebar.markdown("---")
opcao = st.sidebar.radio("Navegação:", ["🏠 Início", "📊 Valuation (Ações)", "📉 Otimização (Markowitz)", "📚 Catálogo (Estudos)"])

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido por:")
st.sidebar.markdown('<a href="https://www.linkedin.com/in/thassianosoares/" target="_blank" class="footer-link">Thassiano Soares</a>', unsafe_allow_html=True)

if opcao == "🏠 Início":
    st.title("Asset Manager Pro")
    st.markdown("Bem-vindo ao seu painel de controle financeiro.")
    st.markdown("""<br><a href="https://www.linkedin.com/in/thassianosoares/" target="_blank" style="text-decoration: none;"><div style="display: inline-flex; align-items: center; background-color: #0077b5; color: white; padding: 8px 16px; border-radius: 4px; font-family: sans-serif; font-weight: 600; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="white" style="margin-right: 8px;"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg><span>Conectar no LinkedIn</span></div></a><br><br>""", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("📊 Valuation Fundamentalista")
            st.markdown("Descubra o preço justo de ações utilizando métodos clássicos.")
    with col2:
        with st.container(border=True):
            st.subheader("📉 Otimização de Portfólio")
            st.markdown("Utilize a Teoria Moderna de Portfólio (Markowitz) e Simulação de Monte Carlo.")
    with st.container(border=True):
        st.subheader("📚 Catálogo de Estudos")
        st.markdown("Organize suas teses de investimento e valuations passados (Salvo na Nuvem).")

elif opcao == "📊 Valuation (Ações)":
    st.title("📊 Valuation Fundamentalista")
    with st.container(border=True):
        st.subheader("1. Parâmetros de Entrada")
        c1, c2, c3 = st.columns(3)
        tb = c1.number_input("Taxa Bazin (Dec)", 0.01, 0.50, 0.08, step=0.01, format="%.2f", help="Taxa Mínima de Atratividade (TMA).")
        tg = c2.number_input("Taxa Desconto - Gordon", 0.01, 0.50, 0.12, step=0.01, format="%.2f", help="Custo de Capital.")
        tc = c3.number_input("Taxa Crescimento - Gordon", 0.00, 0.10, 0.02, step=0.01, format="%.2f", help="Crescimento perpétuo (g).")
        tickers = st.text_area("Tickers", "BBAS3, ITSA4, WEG3")
    if st.button("🔍 Calcular", type="primary"):
        lista = [t.strip() for t in tickers.split(',') if t.strip()]
        res_valuation = []
        res_dividendos = [] 
        bar = st.progress(0)
        for i, tick in enumerate(lista):
            dados = extrair_dados_valuation(tick, tb, tg, tc)
            if dados:
                hist = dados.pop("Historico_Raw") 
                res_valuation.append(dados)
                linha_div = {"Ticker": dados["Ticker"], "Média Usada": dados["DPA Est."]}
                for ano, valor in hist: linha_div[str(ano)] = valor
                res_dividendos.append(linha_div)
            bar.progress((i+1)/len(lista))
        if res_valuation:
            df = pd.DataFrame(res_valuation)
            st.markdown("### 🎯 Resultados")
            tickers_list = df['Ticker'].tolist()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=tickers_list, y=df['Preço Atual'], name='Preço Atual', marker_color='#95a5a6', text=df['Preço Atual'], texttemplate='R$ %{y:.2f}'))
            fig.add_trace(go.Bar(x=tickers_list, y=df['Graham'], name='Graham', marker_color='#27ae60', text=df['Graham'], texttemplate='R$ %{y:.2f}'))
            fig.add_trace(go.Bar(x=tickers_list, y=df['Bazin'], name='Bazin', marker_color='#2980b9', text=df['Bazin'], texttemplate='R$ %{y:.2f}'))
            fig.add_trace(go.Bar(x=tickers_list, y=df['Gordon'], name='Gordon', marker_color='#9b59b6', text=df['Gordon'], texttemplate='R$ %{y:.2f}'))
            fig.update_layout(barmode='group', yaxis_tickprefix="R$ ", template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                df,
                column_config={
                    "Preço Atual": st.column_config.NumberColumn(format="R$ %.2f"),
                    "DPA Est.": st.column_config.NumberColumn(format="R$ %.4f"),
                    "Graham": st.column_config.NumberColumn(format="R$ %.2f"),
                    "Bazin": st.column_config.NumberColumn(format="R$ %.2f"),
                    "Gordon": st.column_config.NumberColumn(format="R$ %.2f"),
                    "Margem Graham": st.column_config.NumberColumn(format="%.2f%%"),
                    "Margem Bazin": st.column_config.NumberColumn(format="%.2f%%"),
                    "Margem Gordon": st.column_config.NumberColumn(format="%.2f%%"),
                },
                use_container_width=True,
                hide_index=True
            )
            with st.expander("📂 Histórico de Dividendos"):
                if res_dividendos:
                    df_divs = pd.DataFrame(res_dividendos).set_index("Ticker")
                    st.dataframe(df_divs.style.format("R$ {:.4f}", na_rep="-"), use_container_width=True)
        else: st.warning("Nenhum dado encontrado.")

elif opcao == "📉 Otimização (Markowitz)":
    st.title("📉 Otimizador de Carteira")
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        arquivo = c1.file_uploader("Upload Excel", type=['xlsx'])
        with c2:
            st.markdown("**Calibragem**")
            tipo_dados = st.radio("Conteúdo do Excel:", ["Preços Históricos (R$)", "Retornos Já Calculados (%)"], horizontal=True)
            freq_option = st.selectbox("Periodicidade:", ["Diário (252)", "Mensal (12)", "Sem Anualização"])
            if freq_option.startswith("Diário"): fator_anual = 252
            elif freq_option.startswith("Mensal"): fator_anual = 12
            else: fator_anual = 1

    if 'otimizacao_feita' not in st.session_state: st.session_state.otimizacao_feita = False
    
    if arquivo:
        try:
            df_raw = pd.read_excel(arquivo)
            first_col = df_raw.iloc[:, 0]
            if not np.issubdtype(first_col.dtype, np.number):
                df_raw = df_raw.set_index(df_raw.columns[0])
                try: df_raw.index = pd.to_datetime(df_raw.index, dayfirst=True)
                except: df_raw.index = pd.to_datetime(df_raw.index, dayfirst=True, errors='coerce')
            df_raw.sort_index(ascending=True, inplace=True)
            cols_numericas = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            cols_selecionadas = st.multiselect("Selecione os ATIVOS:", options=df_raw.columns, default=cols_numericas)
            if len(cols_selecionadas) < 2: st.error("Selecione 2+ ativos."); st.stop()
            df_ativos = df_raw[cols_selecionadas].dropna()
            if tipo_dados.startswith("Preços"): retornos = df_ativos.pct_change().dropna()
            else: retornos = df_ativos
            df_perf = gerar_tabela_performance(retornos, fator_anual)
            st.markdown("---")
            st.warning("⚠️ **Raio-X do Arquivo:** Confira se os números abaixo fazem sentido.")
            st.dataframe(df_perf.set_index("Ativo").style.format("{:.2f}%", na_rep="-"), use_container_width=True)
            cov_matrix = retornos.cov() * fator_anual
            media_historica = df_perf["Média Histórica (Total)"].values
        except Exception as e: st.error(f"Erro: {e}"); st.stop()
        
        with st.container(border=True):
            df_config = pd.DataFrame({
                "Ativo": cols_selecionadas,
                "Peso Atual (%)": [round(100/len(cols_selecionadas), 2)] * len(cols_selecionadas),
                "Visão Retorno (%)": [round(m, 2) for m in media_historica], 
                "Min (%)": [0.0] * len(cols_selecionadas),
                "Max (%)": [100.0] * len(cols_selecionadas)
            })
            config_editada = st.data_editor(df_config, num_rows="fixed", hide_index=True, use_container_width=True)
            rf_input = st.number_input("Taxa Livre de Risco (%)", 0.0, 50.0, 10.0, format="%.2f") / 100
        
        if st.button("🚀 Otimizar", type="primary"):
            visoes = config_editada["Visão Retorno (%)"].values / 100
            pesos_user = config_editada["Peso Atual (%)"].values / 100
            bounds = [(r["Min (%)"]/100, r["Max (%)"]/100) for _, r in config_editada.iterrows()]
            if abs(sum(pesos_user) - 100) > 1: pesos_user = pesos_user / sum(pesos_user)
            elif sum(pesos_user) > 10: pesos_user = pesos_user / 100
            n = len(cols_selecionadas); w0 = np.ones(n) / n
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            try:
                res_opt = minimize(min_sp, w0, args=(visoes, cov_matrix, rf_input), method='SLSQP', bounds=bounds, constraints=constraints)
                w_opt = res_opt.x
                r_opt, v_opt, s_opt = calc_portfolio(w_opt, visoes, cov_matrix, rf_input)
                r_user, v_user, s_user = calc_portfolio(pesos_user, visoes, cov_matrix, rf_input)
                res_min = minimize(min_vol, w0, args=(visoes, cov_matrix, rf_input), method='SLSQP', bounds=bounds, constraints=constraints)
                r_min, v_min, s_min = calc_portfolio(res_min.x, visoes, cov_matrix, rf_input)
                if np.isnan(r_opt): st.error("Erro matemático.")
                else:
                    st.session_state.otimizacao_feita = True
                    st.session_state.resultados = {
                        'ativos_lista': cols_selecionadas, 'r_opt': r_opt, 'v_opt': v_opt, 's_opt': s_opt, 'w_opt': w_opt,
                        'r_user': r_user, 'v_user': v_user, 's_user': s_user, 'r_min': r_min, 'v_min': v_min,
                        'visoes': visoes, 'bounds': bounds, 'cov': cov_matrix, 'pesos_user': pesos_user
                    }
            except Exception as e: st.error(f"Solver Error: {e}")

        if st.session_state.otimizacao_feita:
            res = st.session_state.resultados
            st.markdown("---"); st.markdown("### 🏆 Resultado")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sharpe", f"{res['s_opt']:.2f}")
            col2.metric("Retorno Esp.", f"{res['r_opt']:.1%}")
            col3.metric("Risco", f"{res['v_opt']:.1%}")
            c_chart1, c_chart2 = st.columns([2, 1])
            with c_chart1:
                max_ret = max(res['visoes']); 
                if max_ret > 2.0: max_ret = 2.0
                if max_ret < res['r_opt']: max_ret = res['r_opt'] * 1.05
                rets_target = np.linspace(res['r_min'], max_ret, 40)
                vol_curve, ret_curve, hover_texts = [], [], []
                n_salvo = len(res['ativos_lista']); w0_grafico = np.ones(n_salvo) / n_salvo
                for r_target in rets_target:
                    cons_curve = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'eq', 'fun': lambda x: calc_portfolio(x, res['visoes'], res['cov'], rf_input)[0] - r_target})
                    result = minimize(min_vol, w0_grafico, args=(res['visoes'], res['cov'], rf_input), method='SLSQP', bounds=res['bounds'], constraints=cons_curve)
                    if result.success:
                        r_c, v_c, s_c = calc_portfolio(result.x, res['visoes'], res['cov'], rf_input)
                        ret_curve.append(r_c); vol_curve.append(v_c)
                        hover_texts.append(gerar_hover_text("Curva", r_c, v_c, s_c, result.x, res['ativos_lista']))
                fig = go.Figure()
                if len(vol_curve) > 0: fig.add_trace(go.Scatter(x=vol_curve, y=ret_curve, mode='lines', name='Fronteira', line=dict(color='#3498db', width=3), hoverinfo='text', text=hover_texts))
                fig.add_trace(go.Scatter(x=[res['v_opt']], y=[res['r_opt']], mode='markers', marker=dict(size=15, color='#f1c40f', line=dict(width=2, color='black')), name='Ideal', hoverinfo='text', text=gerar_hover_text("Ideal", res['r_opt'], res['v_opt'], res['s_opt'], res['w_opt'], res['ativos_lista'])))
                fig.add_trace(go.Scatter(x=[res['v_user']], y=[res['r_user']], mode='markers', marker=dict(size=12, color='black', symbol='x'), name='Atual', hoverinfo='text', text=gerar_hover_text("Atual", res['r_user'], res['v_user'], res['s_user'], res['pesos_user'], res['ativos_lista'])))
                fig.update_layout(title="Risco vs. Retorno", xaxis_title="Risco", yaxis_title="Retorno", template="plotly_white", xaxis=dict(tickformat=".1%"), yaxis=dict(tickformat=".1%"), height=400)
                st.plotly_chart(fig, use_container_width=True)
            with c_chart2:
                fig_pie = go.Figure(data=[go.Pie(labels=res['ativos_lista'], values=res['w_opt'], hole=.4)])
                fig_pie.update_layout(title="Alocação Ideal", height=400, showlegend=False)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("---"); st.markdown("### 🔮 Projeção Monte Carlo")
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                inv_ini = c1.number_input("Inicial (R$)", 10000.0)
                aporte = c2.number_input("Mensal (R$)", 1000.0)
                anos = c3.number_input("Anos", 10)
                inflacao = c4.number_input("Inflação (%)", 5.0, format="%.2f") / 100
            if st.button("🎲 Simular", type="primary"):
                if np.isnan(res['r_opt']): st.error("Erro.")
                else:
                    opt_top, opt_mid, opt_low, steps, linha_teorica = monte_carlo(res['r_opt'], res['v_opt'], inv_ini, aporte, int(anos), inflacao)
                    usr_top, usr_mid, usr_low, _, _ = monte_carlo(res['r_user'], res['v_user'], inv_ini, aporte, int(anos), inflacao)
                    x = np.linspace(0, int(anos), steps + 1)
                    fig_sim = go.Figure()
                    fig_sim.add_trace(go.Scatter(x=x, y=linha_teorica, mode='lines', name='Teórico (Juros Compostos)', line=dict(color='#f1c40f', width=2, dash='dot')))
                    fig_sim.add_trace(go.Scatter(x=x, y=opt_mid, mode='lines', name='Ideal (Esperado)', line=dict(color='#27ae60', width=3)))
                    fig_sim.add_trace(go.Scatter(x=x, y=opt_low, mode='lines', name='Ideal (Pessimista)', line=dict(color='#abebc6', width=0), fill='tonexty'))
                    fig_sim.add_trace(go.Scatter(x=x, y=usr_mid, mode='lines', name='Atual (Esperado)', line=dict(color='black', dash='dash')))
                    fig_sim.update_layout(title="Crescimento Patrimonial", xaxis_title="Anos", yaxis_title="Patrimônio", template="plotly_white", hovermode="x unified", separators=",.", yaxis=dict(tickprefix="R$ ", tickformat=",.0f"))
                    st.plotly_chart(fig_sim, use_container_width=True)
                    final_val = opt_mid[-1]
                    st.success(f"💰 **Patrimônio Estimado (Cenário Ideal):** R$ {final_val:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

elif opcao == "📚 Catálogo (Estudos)":
    st.title("📚 Diário de Valuation")
    st.info("Todos os estudos são salvos automaticamente na nuvem (Google Sheets).")

    if 'temp_premissas' not in st.session_state: st.session_state.temp_premissas = {}

    with st.container(border=True):
        st.subheader("Novo Registro")
        c1, c2 = st.columns(2)
        f_ticker = c1.text_input("Ticker (Ex: VALE3)").upper().strip()
        f_metodo = c2.selectbox("Método", ["Graham", "Bazin", "Gordon", "DCF", "Múltiplos"])
        c3, c4 = st.columns(2)
        # --- BLINDAGEM NOS CAMPOS DE ENTRADA DE PREÇO ---
        f_cotacao = c3.text_input("Cotação Ref. (R$)", "0,00")
        f_justo = c4.text_input("Preço Justo (R$)", "0,00")
        # ---------------------------------------
        f_tese = st.text_area("Racional / Tese", height=100)
        
        st.markdown("---")
        st.write("**Adicionar Premissas (Ex: WACC, Beta):**")
        cp1, cp2, cp3 = st.columns([2, 2, 1])
        pk = cp1.text_input("Nome")
        pv = cp2.text_input("Valor")
        if cp3.button("➕ Add"):
            if pk and pv: st.session_state.temp_premissas[pk] = pv
        
        if st.session_state.temp_premissas:
            st.write(st.session_state.temp_premissas)
            if st.button("Limpar Premissas"): st.session_state.temp_premissas = {}

        if st.button("💾 SALVAR ESTUDO", type="primary"):
            if f_ticker:
                # --- CONVERSÃO SEGURA DE TEXTO PARA FLOAT ---
                try:
                    val_justo_final = safe_float(f_justo)
                    val_cotacao_final = safe_float(f_cotacao)
                except:
                    st.error("Erro no valor do preço.")
                    st.stop()
                
                if val_justo_final == 0:
                    st.error("Preço Justo não pode ser zero.")
                    st.stop()

                novo = {
                    "Data": datetime.now().strftime("%d/%m/%Y"),
                    "Ticker": f_ticker,
                    "Preço Justo": val_justo_final,
                    "Cotação Ref": val_cotacao_final,
                    "Método": f_metodo,
                    "Tese": f_tese,
                    "Premissas": st.session_state.temp_premissas.copy()
                }
                with st.spinner("Salvando..."):
                    if salvar_no_db(novo):
                        st.session_state.temp_premissas = {}
                        st.success("Salvo com sucesso!")
                        st.rerun()
            else: st.error("Preencha Ticker.")

    st.markdown("---")
    lista_db = carregar_dados_db()
    
    if lista_db:
        for item in lista_db[::-1]:
            try:
                if isinstance(item.get('Premissas_JSON'), str):
                    premissas = json.loads(item['Premissas_JSON'])
                else:
                    premissas = item['Premissas_JSON']
            except: premissas = {}

            # Tenta ler usando as chaves sem acento (do seu DB)
            p_ref = safe_float(item.get('Cotacao_Ref', 0))
            p_justo = safe_float(item.get('Preco_Justo', 0))
            ticker = item.get('Ticker', '')
            metodo = item.get('Metodo', '')
            data_estudo = item.get('Data', '')
            tese = item.get('Tese', '')

            live = obter_cotacao_atual(ticker)
            atual = live if live and live > 0 else p_ref
            lbl = "Ao Vivo" if live and live > 0 else "Ref. Offline"
            
            upside = ((p_justo - atual) / atual) * 100 if atual > 0 else 0
            
            with st.container(border=True):
                col_head1, col_head2 = st.columns([4, 1])
                col_head1.subheader(f"📊 {ticker} | {metodo}")
                col_head2.caption(data_estudo)
                st.divider()
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Ref. Inicial", f"R$ {p_ref:.2f}")
                k2.metric(lbl, f"R$ {atual:.2f}")
                k3.metric("Preço Justo", f"R$ {p_justo:.2f}")
                k4.metric("Upside", f"{upside:+.1f}%", delta="Margem", delta_color="normal")
                
                with st.expander("📖 Ver Tese"):
                    st.info(tese)
                    if premissas:
                        st.table(pd.DataFrame(list(premissas.items()), columns=['Item', 'Valor']))
    else:
        st.info("Nenhum estudo encontrado.")
