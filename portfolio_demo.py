import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# Configuração da Página
st.set_page_config(page_title="Equity Research | Catálogo", layout="wide", page_icon="🏛️")

# CSS
st.markdown("""
<style>
    div[data-testid="stMetricValue"] { font-size: 20px; }
    .card {
        background-color: #ffffff; padding: 20px; border-radius: 12px;
        border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- CONEXÃO COM GOOGLE SHEETS ---
def conectar_gsheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    # Abre a planilha pelo NOME EXATO
    sheet = client.open("DB_Valuation").sheet1 
    return sheet

# --- LEITURA (CACHEADA) ---
@st.cache_data(ttl=60) 
def carregar_dados():
    try:
        sheet = conectar_gsheets()
        dados = sheet.get_all_records()
        return dados
    except Exception as e:
        # st.error(f"Erro no Google Sheets: {e}") # Descomente para debug
        return []

# --- ESCRITA ---
def salvar_novo_estudo(novo_dict):
    try:
        sheet = conectar_gsheets()
        linha = [
            novo_dict['Data'],
            novo_dict['Ticker'],
            novo_dict['Preço Justo'],
            novo_dict['Cotação Ref'],
            novo_dict['Método'],
            novo_dict['Tese'],
            json.dumps(novo_dict['Premissas']) # Salva como texto JSON
        ]
        sheet.append_row(linha)
        st.cache_data.clear() # Limpa cache para atualizar a tela
        return True
    except Exception as e:
        st.error(f"Erro ao salvar: {e}")
        return False

# --- YAHOO FINANCE ---
@st.cache_data(ttl=300)
def obter_cotacao_atual(ticker):
    try:
        t = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
        hist = yf.Ticker(t).history(period="1d")
        if not hist.empty: return hist['Close'].iloc[-1]
        return None
    except: return None

# --- STATE INTERNO PARA PREMISSAS ---
if 'temp_premissas' not in st.session_state:
    st.session_state.temp_premissas = {}

# ==========================================
# BARRA LATERAL (CADASTRO)
# ==========================================
with st.sidebar:
    st.title("👨‍💻 Admin Panel")
    
    st.subheader("1. Dados do Ativo")
    f_ticker = st.text_input("Ticker", placeholder="PETR4").upper().strip()
    
    c1, c2 = st.columns(2)
    f_cotacao = c1.number_input("Cotação Ref. (R$)", 0.0, value=0.00, step=0.01, format="%.2f")
    f_justo = c2.number_input("Preço Justo (R$)", 0.0, value=0.00, step=0.01, format="%.2f")
    
    f_metodo = st.selectbox("Método Principal", ["Graham", "Bazin", "Gordon", "DCF", "Múltiplos", "Híbrido"])
    
    st.subheader("2. Tese")
    f_tese = st.text_area("Racional", height=100)
    
    st.markdown("---")
    st.subheader("3. Montar Premissas")
    
    col_new1, col_new2 = st.columns(2)
    new_key = col_new1.text_input("Nome (Ex: WACC)")
    new_val = col_new2.text_input("Valor (Ex: 13%)")
    
    if st.button("➕ Adicionar Premissa"):
        if new_key and new_val:
            st.session_state.temp_premissas[new_key] = new_val
            st.success(f"{new_key} adicionado!")
            st.rerun()

    if st.session_state.temp_premissas:
        st.markdown("###### Premissas Atuais:")
        df_temp = pd.DataFrame(list(st.session_state.temp_premissas.items()), columns=['Item', 'Valor'])
        st.dataframe(df_temp, hide_index=True, use_container_width=True)
        if st.button("🗑️ Limpar Lista"):
            st.session_state.temp_premissas = {}
            st.rerun()
            
    st.markdown("---")
    
    # BOTÃO SALVAR NO GOOGLE SHEETS
    if st.button("💾 SALVAR NA NUVEM", type="primary"):
        if f_ticker and f_justo > 0:
            novo_estudo = {
                "Ticker": f_ticker,
                "Data": datetime.now().strftime("%d/%m/%Y"),
                "Preço Justo": f_justo,
                "Cotação Ref": f_cotacao,
                "Método": f_metodo,
                "Tese": f_tese,
                "Premissas": st.session_state.temp_premissas.copy()
            }
            
            with st.spinner("Salvando no Google Sheets..."):
                sucesso = salvar_novo_estudo(novo_estudo)
            
            if sucesso:
                st.session_state.temp_premissas = {}
                st.success(f"{f_ticker} salvo no banco de dados!")
                st.rerun()
        else:
            st.error("Preencha Ticker e Preço.")

# ==========================================
# ÁREA PRINCIPAL
# ==========================================
c_title, c_refresh = st.columns([4, 1])
c_title.title("🏛️ Catálogo de Estudos")
if c_refresh.button("🔄 Atualizar"):
    st.cache_data.clear()
    st.rerun()

st.markdown("---")

# CARREGA DO BANCO DE DADOS
lista_db = carregar_dados()

if not lista_db:
    st.info("Nenhum estudo encontrado no banco de dados (Google Sheets).")
else:
    # Loop Inverso (Mais recente primeiro)
    for item in lista_db[::-1]:
        
        # Recupera premissas do JSON
        try:
            premissas_dict = json.loads(item['Premissas_JSON'])
        except:
            premissas_dict = {}

        # Valores Seguros (Converter string da planilha em float)
        try: p_ref = float(str(item['Cotacao_Ref']).replace("R$", "").replace(",", "."))
        except: p_ref = 0.0
        try: p_justo = float(str(item['Preco_Justo']).replace("R$", "").replace(",", "."))
        except: p_justo = 0.0

        # Lógica de Preço
        live = obter_cotacao_atual(item['Ticker'])
        atual = live if live else p_ref
        lbl = "Ao Vivo" if live else "Ref. Offline"
        
        upside = ((p_justo - atual) / atual) * 100 if atual > 0 else 0
        
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            c1.subheader(f"📊 {item['Ticker']} | {item['Metodo']}")
            c2.caption(item['Data'])
            
            st.divider()
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Ref. Inicial", f"R$ {p_ref:.2f}")
            k2.metric(lbl, f"R$ {atual:.2f}")
            k3.metric("Preço Justo", f"R$ {p_justo:.2f}")
            k4.metric("Upside", f"{upside:+.1f}%", delta="Margem", delta_color="normal")
            
            with st.expander("📖 Ver Tese Detalhada"):
                col_txt, col_graph = st.columns([1.5, 1])
                with col_txt:
                    st.markdown("**Racional:**")
                    st.info(item['Tese'])
                    st.markdown("**Premissas:**")
                    if premissas_dict:
                        st.table(pd.DataFrame(list(premissas_dict.items()), columns=['Item', 'Valor']))
                
                with col_graph:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta", value=atual,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Margem de Segurança"},
                        delta={'reference': p_justo, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                        gauge={'axis': {'range': [None, p_justo*1.5]}, 'bar': {'color': "gray"}, 'steps': [{'range': [0, p_justo], 'color': "#d4edda"}], 'threshold': {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': p_justo}}
                    ))
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, use_container_width=True)

