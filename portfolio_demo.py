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
# 1. CONEXÃO E HELPERS
# ==========================================
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

@st.cache_data(ttl=300)
def obter_cotacao_atual(ticker):
    try:
        t = ticker.strip().upper()
        if not t.endswith(".SA") and len(t) <= 6: t = f"{t}.SA"
        hist = yf.Ticker(t).history(period="1d")
        if not hist.empty: return hist['Close'].iloc[-1]
        return None
    except: return None

# Funções de Cálculo (Mantidas)
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

def calcular_cagr(s, f):
    if len(s)<1: return 0
    return ((1+s).prod())**(f/len(s))-1 if f!=1 else (1+s).prod()-1

def gerar_tabela_performance(r, f):
    s = []
    for a in r.columns:
        ser = r[a]; rt = calcular_cagr(ser, f)
        p12 = 12 if f==12 else 252
        r12 = calcular_cagr(ser.tail(p12), f) if len(ser)>=p12 else np.nan
        s.append({"Ativo": a, "Média Histórica (Total)": rt*100, "Últimos 12 Meses": r12*100 if not np.isnan(r12) else None})
    return pd.DataFrame(s)

def calc_portfolio(w, r, cov, rf):
    rp = np.sum(w * r); vp = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    return rp, vp, (rp - rf) / vp if vp > 0 else 0

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
    
    # Botão LinkedIn
    st.markdown("""
        <a href="https://www.linkedin.com/in/thassianosoares/" target="_blank" style="text-decoration: none;">
            <div style="display: inline-flex; align-items: center; background-color: #0077b5; color: white; padding: 8px 16px; border-radius: 4px; font-family: sans-serif; font-weight: 600;">
                <span>Conectar no LinkedIn</span>
            </div>
        </a>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # --- VÍDEO TUTORIAL (MENOR E CENTRALIZADO) ---
    st.subheader("📺 Como usar a plataforma")
    c_vid1, c_vid2, c_vid3 = st.columns([1, 2, 1])
    with c_vid2:
        # TROQUE PELO SEU LINK DO YOUTUBE AQUI
        st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ") 
    
    st.divider()
    
    # --- CARDS DOS MÓDULOS ---
    st.subheader("🛠️ Ferramentas Disponíveis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown("### 📊 Valuation")
            st.markdown("Cálculo automático de preço justo.")
            st.info("Acesse no Menu Lateral")
            
    with col2:
        with st.container(border=True):
            st.markdown("### 📉 Otimização")
            st.markdown("Fronteira Eficiente e Monte Carlo.")
            st.info("Acesse no Menu Lateral")
            
    with col3:
        with st.container(border=True):
            st.markdown("### 📚 Catálogo")
            st.markdown("Banco de dados de teses de investimento.")
            st.info("Acesse no Menu Lateral")

elif opcao == "📊 Valuation (Ações)":
    st.title("📊 Valuation Fundamentalista")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        # Inputs com Tooltips (RESTAURADO)
        tb = c1.number_input("Taxa Bazin (Dec)", 0.01, 0.50, 0.08, format="%.2f", help="Taxa Mínima de Atratividade (TMA). Comum: 0.06 a 0.10.")
        tg = c2.number_input("Taxa Gordon", 0.01, 0.50, 0.12, format="%.2f", help="Custo de Capital (Retorno Exigido).")
        tc = c3.number_input("Cresc. g", 0.00, 0.10, 0.02, format="%.2f", help="Crescimento perpétuo (g). Deve ser < PIB.")
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
            # Gráfico Restaurado com 4 Barras
            fig.add_trace(go.Bar(x=l, y=df['Preço Atual'], name='Atual', marker_color='#95a5a6', text=df['Preço Atual'], texttemplate='R$ %{y:.2f}'))
            fig.add_trace(go.Bar(x=l, y=df['Graham'], name='Graham', marker_color='#27ae60', text=df['Graham'], texttemplate='R$ %{y:.2f}'))
            fig.add_trace(go.Bar(x=l, y=df['Bazin'], name='Bazin', marker_color='#2980b9', text=df['Bazin'], texttemplate='R$ %{y:.2f}'))
            fig.add_trace(go.Bar(x=l, y=df['Gordon'], name='Gordon', marker_color='#9b59b6', text=df['Gordon'], texttemplate='R$ %{y:.2f}'))
            fig.update_layout(barmode='group', template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela Formatada (Restaurada)
            st.dataframe(df, column_config={"Preço Atual": st.column_config.NumberColumn(format="R$ %.2f"), "DPA Est.": st.column_config.NumberColumn(format="R$ %.4f"), "Graham": st.column_config.NumberColumn(format="R$ %.2f"), "Bazin": st.column_config.NumberColumn(format="R$ %.2f"), "Gordon": st.column_config.NumberColumn(format="R$ %.2f"), "Margem Graham": st.column_config.NumberColumn(format="%.2f%%"), "Margem Bazin": st.column_config.NumberColumn(format="%.2f%%"), "Margem Gordon": st.column_config.NumberColumn(format="%.2f%%")}, use_container_width=True, hide_index=True)
        else: st.warning("Sem dados.")

elif opcao == "📉 Otimização (Markowitz)":
    st.title("📉 Otimizador de Carteira")
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        arquivo = c1.file_uploader("Upload Excel", type=['xlsx'])
        with c2:
            tipo_dados = st.radio("Conteúdo:", ["Preços (R$)", "Retornos (%)"])
            freq_option = st.selectbox("Freq:", ["Diário (252)", "Mensal (12)"])
            fator = 252 if freq_option.startswith("Diário") else 12
    if 'otimizacao_feita' not in st.session_state: st.session_state.otimizacao_feita = False
    if arquivo:
        try:
            df = pd.read_excel(arquivo)
            first_col = df.iloc[:, 0]
            if not np.issubdtype(first_col.dtype, np.number):
                df = df.set_index(df.columns[0])
                try: df.index = pd.to_datetime(df.index, dayfirst=True)
                except: df.index = pd.to_datetime(df.index, dayfirst=True, errors='coerce')
            df.sort_index(ascending=True, inplace=True)
            col_num = df.select_dtypes(include=[np.number]).columns.tolist()
            sel = st.multiselect("Ativos:", options=df.columns, default=col_num)
            if len(sel)<2: st.error("Selecione 2+ ativos."); st.stop()
            df_sel = df[sel].dropna()
            ret = df_sel.pct_change().dropna() if tipo_dados.startswith("Preços") else df_sel
            st.dataframe(gerar_tabela_performance(ret, fator).set_index("Ativo").style.format("{:.2f}%", na_rep="-"), use_container_width=True)
            cov = ret.cov() * fator
            media = gerar_tabela_performance(ret, fator)["Média Histórica (Total)"].values
        except: st.error("Erro no arquivo."); st.stop()
        
        with st.container(border=True):
            df_c = pd.DataFrame({"Ativo": sel, "Visão (%)": [round(m, 2) for m in media], "Min (%)": [0.0]*len(sel), "Max (%)": [100.0]*len(sel)})
            cfg = st.data_editor(df_c, num_rows="fixed", hide_index=True, use_container_width=True)
            rf = st.number_input("Risk Free (%)", 10.0)/100
        
        if st.button("🚀 Otimizar", type="primary"):
            visoes = cfg["Visão (%)"].values/100
            pesos_user = np.ones(len(sel))/len(sel)
            b = [(r["Min (%)"]/100, r["Max (%)"]/100) for _, r in cfg.iterrows()]
            n = len(sel); w0 = np.ones(n)/n
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1})
            try:
                res = minimize(min_sp, w0, args=(visoes, cov, rf), method='SLSQP', bounds=b, constraints=cons)
                w = res.x; r_opt, v_opt, s_opt = calc_portfolio(w, visoes, cov, rf)
                r_u, v_u, _ = calc_portfolio(pesos_user, visoes, cov, rf)
                st.session_state.otimizacao_feita = True
                st.session_state.res = {'sel': sel, 'r_opt': r_opt, 'v_opt': v_opt, 's_opt': s_opt, 'w': w, 'v': visoes, 'cov': cov, 'r_u': r_u, 'v_u': v_u}
            except: st.error("Erro matemático.")

        if st.session_state.otimizacao_feita:
            r = st.session_state.res
            c1, c2, c3 = st.columns(3)
            c1.metric("Sharpe", f"{r['s_opt']:.2f}"); c2.metric("Retorno", f"{r['r_opt']:.1%}"); c3.metric("Volatilidade", f"{r['v_opt']:.1%}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[r['v_opt']], y=[r['r_opt']], mode='markers', marker=dict(size=15, color='#f1c40f'), name='Ideal'))
            fig.add_trace(go.Scatter(x=[r['v_u']], y=[r['r_u']], mode='markers', marker=dict(size=12, color='black', symbol='x'), name='Atual'))
            st.plotly_chart(fig, use_container_width=True)
            fig_p = go.Figure(data=[go.Pie(labels=r['sel'], values=r['w'], hole=.4)]); st.plotly_chart(fig_p, use_container_width=True)
            
            st.markdown("### 🔮 Monte Carlo")
            c1, c2, c3 = st.columns(3)
            ini = c1.number_input("Inicial", 10000.0); aport = c2.number_input("Mensal", 1000.0); ano = c3.number_input("Anos", 10)
            if st.button("Simular"):
                o, m, p, s, t = monte_carlo(r['r_opt'], r['v_opt'], ini, aport, int(ano), 0.05)
                f = go.Figure(); x = np.linspace(0, int(ano), s+1)
                f.add_trace(go.Scatter(x=x, y=t, name='Teórico', line=dict(color='orange', dash='dot'))); f.add_trace(go.Scatter(x=x, y=m, name='Esperado', line=dict(color='green')))
                st.plotly_chart(f, use_container_width=True)

# --- CATÁLOGO (ESTUDOS) ---
elif opcao == "📚 Catálogo (Estudos)":
    st.title("📚 Diário de Valuation")
    
    # SÓ MOSTRA O FORMULÁRIO SE FOR ADMIN
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

    # LISTAGEM PÚBLICA (COM VISUAL NOVO E YAHOO)
    lista_db = carregar_dados_db()
    if ldb := lista_db:
        for i, item in enumerate(ldb[::-1]):
            # 1. Recupera Premissas e Preços
            try: premissas = json.loads(item['Premissas_JSON']) if isinstance(item.get('Premissas_JSON'), str) else {}
            except: premissas = {}
            p_justo = safe_float(item.get('Preco_Justo', 0))
            p_ref = safe_float(item.get('Cotacao_Ref', 0))
            ticker = item.get('Ticker', '')

            # 2. Busca Live (Yahoo)
            live = obter_cotacao_atual(ticker)
            atual = live if live and live > 0 else p_ref
            lbl = "Ao Vivo" if live else "Ref. Offline"
            
            # 3. Calcula Upside
            upside = ((p_justo - atual) / atual) * 100 if atual > 0 else 0
            
            # 4. Card Visual Completo (Com Gauge)
            with st.container(border=True):
                ch1, ch2 = st.columns([5, 1])
                ch1.subheader(f"📊 {ticker} | {item.get('Metodo', '')}")
                
                if st.session_state.admin_logged:
                    if ch2.button("🗑️", key=f"del_{i}"): deletar_do_db(i); st.rerun()
                else: ch2.caption(item.get('Data', ''))
                
                st.divider()
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Ref. Inicial", f"R$ {p_ref:.2f}")
                k2.metric(lbl, f"R$ {atual:.2f}")
                k3.metric("Preço Justo", f"R$ {p_justo:.2f}")
                k4.metric("Upside", f"{upside:+.1f}%", delta="Margem", delta_color="normal")
                
                with st.expander("📖 Ver Tese e Gráfico"):
                    ct, cg = st.columns([1.5, 1])
                    with ct:
                        st.info(item.get('Tese', ''))
                        if premissas: st.table(pd.DataFrame(list(premissas.items()), columns=['Item', 'Valor']))
                    with cg:
                        # Gráfico Gauge Integrado
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta", value=atual,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Margem"},
                            delta={'reference': p_justo, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                            gauge={'axis': {'range': [None, p_justo*1.5]}, 'bar': {'color': "gray"}, 'steps': [{'range': [0, p_justo], 'color': "#d4edda"}], 'threshold': {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': p_justo}}
                        ))
                        fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig, use_container_width=True)
