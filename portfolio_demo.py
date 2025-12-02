import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import hmac

# ==========================================
# 0. CONFIGURAÇÃO
# ==========================================
st.set_page_config(page_title="Asset Manager Pro", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .card { background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
    div[data-testid="stMetricValue"] { font-size: 22px; }
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
    st.text_input("Senha", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state: st.error("Senha incorreta")
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

# --- FUNÇÃO DE LIMPEZA DE NÚMEROS (CRUCIAL) ---
def safe_float(valor):
    """Converte qualquer bagunça (R$ 30,00 / 30.00 / 30) para float puro."""
    if isinstance(valor, (int, float)):
        return float(valor)
    try:
        # Remove R$, troca vírgula por ponto e limpa espaços
        limpo = str(valor).replace("R$", "").replace(" ", "").replace(",", ".")
        return float(limpo)
    except:
        return 0.0

@st.cache_data(ttl=10) 
def carregar_dados_db():
    try:
        sheet = conectar_gsheets()
        # get_all_records usa a primeira linha como chave do dicionário
        dados = sheet.get_all_records()
        return dados
    except Exception as e:
        st.error(f"Erro ao ler planilha: {e}")
        return []

def salvar_no_db(novo_dict):
    try:
        sheet = conectar_gsheets()
        # Salva convertendo para string segura (com ponto) para o Google não confundir
        linha = [
            novo_dict['Data'],
            novo_dict['Ticker'],
            str(novo_dict['Preço Justo']).replace(".", ","), # Salva com virgula pro Excel brasileiro ficar feliz
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

# --- YAHOO FINANCE ROBUSTO ---
@st.cache_data(ttl=300)
def obter_cotacao_atual(ticker):
    try:
        ticker = ticker.strip().upper()
        # Garante .SA
        if not ticker.endswith(".SA") and len(ticker) <= 6:
            ticker_full = f"{ticker}.SA"
        else:
            ticker_full = ticker
            
        # Tenta baixar
        ativo = yf.Ticker(ticker_full)
        hist = ativo.history(period="1d")
        
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return None
    except:
        return None

# --- ESTADO ---
if 'temp_premissas' not in st.session_state:
    st.session_state.temp_premissas = {}

# ==========================================
# 2. PÁGINA CATÁLOGO (Focada no seu pedido)
# ==========================================
st.sidebar.title("Asset Manager Pro")
# (Aqui eu removi as outras abas do menu para simplificar o código que te mando, 
# mas você pode manter o resto do seu app.py e só substituir a parte do Catálogo)
# Focando apenas na lógica do Catálogo que estava dando erro:

st.title("📚 Diário de Valuation")
st.info("Dados sincronizados com Google Sheets (DB_Valuation).")

# --- SIDEBAR DE CADASTRO ---
with st.sidebar:
    st.markdown("---")
    st.subheader("Novo Estudo")
    
    f_ticker = st.text_input("Ticker (Ex: VALE3)", key="fticker").upper().strip()
    f_metodo = st.selectbox("Método", ["DCF (Fluxo de Caixa)", "Graham", "Bazin", "Gordon", "Múltiplos"])
    
    # Inputs de Texto para permitir digitação livre (ex: "37,50" ou "37.50")
    c3, c4 = st.columns(2)
    txt_cotacao = c3.text_input("Cotação Ref.", "0,00")
    txt_justo = c4.text_input("Preço Justo", "0,00")
    
    f_tese = st.text_area("Racional / Tese", height=100)
    
    st.markdown("---")
    st.write("**Premissas:**")
    cp1, cp2, cp3 = st.columns([2, 2, 1])
    pk = cp1.text_input("Nome (ex: WACC)")
    pv = cp2.text_input("Valor (ex: 12%)")
    if cp3.button("➕"):
        if pk and pv: st.session_state.temp_premissas[pk] = pv
    
    if st.session_state.temp_premissas:
        st.dataframe(pd.DataFrame(list(st.session_state.temp_premissas.items()), columns=['Item', 'Valor']), hide_index=True)
        if st.button("Limpar Premissas"): st.session_state.temp_premissas = {}

    if st.button("💾 SALVAR ESTUDO", type="primary"):
        if f_ticker:
            # Converte o texto digitado para número float usando a função segura
            val_justo = safe_float(txt_justo)
            val_cotacao = safe_float(txt_cotacao)
            
            if val_justo == 0:
                st.error("Erro: O Preço Justo não pode ser zero. Verifique se usou ponto ou vírgula corretamente.")
            else:
                novo = {
                    "Data": datetime.now().strftime("%d/%m/%Y"),
                    "Ticker": f_ticker,
                    "Preço Justo": val_justo,
                    "Cotação Ref": val_cotacao,
                    "Método": f_metodo,
                    "Tese": f_tese,
                    "Premissas": st.session_state.temp_premissas.copy()
                }
                with st.spinner("Salvando..."):
                    if salvar_no_db(novo):
                        st.session_state.temp_premissas = {}
                        st.success("Salvo!")
                        st.rerun()
        else:
            st.error("Preencha o Ticker.")

# --- EXIBIÇÃO DOS CARDS ---
st.markdown("---")
if st.button("🔄 Atualizar Lista"):
    st.cache_data.clear()
    st.rerun()

lista_db = carregar_dados_db()

if not lista_db:
    st.warning("Nenhum estudo encontrado. Verifique se a planilha 'DB_Valuation' tem os cabeçalhos: Data, Ticker, Preco_Justo, Cotacao_Ref, Metodo, Tese, Premissas_JSON")
else:
    for item in lista_db[::-1]:
        # 1. Recupera Premissas
        try:
            if isinstance(item.get('Premissas_JSON'), str):
                premissas = json.loads(item['Premissas_JSON'])
            else: premissas = {}
        except: premissas = {}

        # 2. Recupera Preços (Função safe_float resolve o problema do 0.00)
        # Nota: Usamos .get() para evitar erro se a coluna mudar de nome
        p_justo = safe_float(item.get('Preco_Justo', 0))
        p_ref = safe_float(item.get('Cotacao_Ref', 0))
        ticker = item.get('Ticker', 'N/A')
        
        # 3. Busca Live
        live = obter_cotacao_atual(ticker)
        
        # Lógica de exibição
        atual = live if live and live > 0 else p_ref
        lbl_atual = "Ao Vivo (Yahoo)" if live and live > 0 else "Ref. (Offline)"
        
        # Evita divisão por zero
        if atual > 0:
            upside = ((p_justo - atual) / atual) * 100
        else:
            upside = 0
        
        # Visual
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            c1.subheader(f"📊 {ticker} | {item.get('Metodo', '')}")
            c2.caption(item.get('Data', ''))
            
            st.divider()
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Ref. Inicial", f"R$ {p_ref:.2f}")
            
            # Mostra variação do live
            delta_live = f"{((atual - p_ref)/p_ref)*100:.1f}%" if live and p_ref > 0 else None
            k2.metric(lbl_atual, f"R$ {atual:.2f}", delta=delta_live)
            
            k3.metric("Preço Justo", f"R$ {p_justo:.2f}")
            
            # Cor do Upside
            k4.metric("Upside", f"{upside:+.1f}%", delta="Margem", delta_color="normal")
            
            with st.expander("📖 Ver Tese"):
                st.info(item.get('Tese', ''))
                if premissas:
                    st.table(pd.DataFrame(list(premissas.items()), columns=['Item', 'Valor']))
