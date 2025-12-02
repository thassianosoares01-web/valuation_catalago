import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

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

# --- FUNÇÃO YAHOO ---
@st.cache_data(ttl=300)
def obter_cotacao_atual(ticker):
    try:
        t = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
        hist = yf.Ticker(t).history(period="1d")
        if not hist.empty: return hist['Close'].iloc[-1]
        return None
    except: return None

# --- INICIALIZAÇÃO DO BANCO DE DADOS ---
if 'catalogo' not in st.session_state:
    st.session_state.catalogo = []

# --- INICIALIZAÇÃO DA LISTA TEMPORÁRIA DE PREMISSAS ---
if 'temp_premissas' not in st.session_state:
    st.session_state.temp_premissas = {}

# ==========================================
# BARRA LATERAL (CADASTRO DINÂMICO)
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
    f_tese = st.text_area("Racional", height=100, placeholder="Pontos chave...")
    
    st.markdown("---")
    st.subheader("3. Montar Premissas")
    st.caption("Adicione quantas premissas quiser (Ex: WACC, Beta, Ke).")
    
    # Campos para adicionar nova premissa
    col_new1, col_new2 = st.columns(2)
    new_key = col_new1.text_input("Nome (Ex: WACC)")
    new_val = col_new2.text_input("Valor (Ex: 13%)")
    
    if st.button("➕ Adicionar Premissa"):
        if new_key and new_val:
            st.session_state.temp_premissas[new_key] = new_val
            st.success(f"{new_key} adicionado!")
            st.rerun() # Atualiza a tela para mostrar na tabela abaixo
        else:
            st.warning("Preencha Nome e Valor.")

    # Mostra o que já foi adicionado
    if st.session_state.temp_premissas:
        st.markdown("###### Premissas Atuais:")
        # Mostra como tabela
        df_temp = pd.DataFrame(list(st.session_state.temp_premissas.items()), columns=['Item', 'Valor'])
        st.dataframe(df_temp, hide_index=True, use_container_width=True)
        
        if st.button("🗑️ Limpar Premissas"):
            st.session_state.temp_premissas = {}
            st.rerun()
            
    st.markdown("---")
    
    # BOTÃO FINAL DE SALVAR
    if st.button("💾 SALVAR ESTUDO COMPLETO", type="primary"):
        if f_ticker and f_justo > 0:
            # Salva no catálogo principal
            novo_estudo = {
                "Ticker": f_ticker,
                "Data": datetime.now().strftime("%d/%m/%Y"),
                "Preço Justo": f_justo,
                "Cotação Ref": f_cotacao,
                "Método": f_metodo,
                "Tese": f_tese,
                "Premissas": st.session_state.temp_premissas.copy() # Copia o que está na lista temporária
            }
            st.session_state.catalogo.insert(0, novo_estudo)
            
            # Limpa a lista temporária para o próximo
            st.session_state.temp_premissas = {}
            
            st.success(f"Estudo de {f_ticker} salvo com sucesso!")
            st.rerun()
        else:
            st.error("Preencha pelo menos Ticker e Preço Justo.")

# ==========================================
# ÁREA PRINCIPAL
# ==========================================
c_title, c_search = st.columns([2, 1])
c_title.title("🏛️ Catálogo de Estudos")
termo = c_search.text_input("🔍 Buscar", placeholder="Ticker...").upper()
st.markdown("---")

if not st.session_state.catalogo:
    st.info("Nenhum estudo cadastrado. Use a barra lateral para criar.")

# Filtro
lista_final = [x for x in st.session_state.catalogo if termo in x['Ticker']]

for i, item in enumerate(lista_final):
    # Lógica de Preço
    live = obter_cotacao_atual(item['Ticker'])
    atual = live if live else item['Cotação Ref']
    lbl = "Ao Vivo" if live else "Ref. Offline"
    
    upside = ((item['Preço Justo'] - atual) / atual) * 100 if atual > 0 else 0
    
    with st.container(border=True):
        # Header
        c1, c2 = st.columns([5, 1])
        c1.subheader(f"📊 {item['Ticker']} | {item['Método']}")
        c2.caption(item['Data'])
        
        # Botão Excluir
        if c2.button("🗑️", key=f"del_{i}"):
            st.session_state.catalogo.remove(item)
            st.rerun()
            
        st.divider()
        
        # Métricas
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ref. Inicial", f"R$ {item['Cotação Ref']:.2f}")
        k2.metric(lbl, f"R$ {atual:.2f}")
        k3.metric("Preço Justo", f"R$ {item['Preço Justo']:.2f}")
        k4.metric("Upside", f"{upside:+.1f}%", delta="Margem", delta_color="normal")
        
        # Detalhes
        with st.expander("📖 Ver Tese e Premissas Detalhadas"):
            col_txt, col_graph = st.columns([1.5, 1])
            
            with col_txt:
                st.markdown("**Racional:**")
                st.info(item['Tese'])
                
                st.markdown("**Premissas Utilizadas:**")
                if item['Premissas']:
                    df_p = pd.DataFrame(list(item['Premissas'].items()), columns=['Item', 'Valor'])
                    st.table(df_p)
                else:
                    st.caption("Nenhuma premissa registrada.")
            
            with col_graph:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=atual,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Margem de Segurança"},
                    delta={'reference': item['Preço Justo'], 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [None, item['Preço Justo']*1.5]},
                        'bar': {'color': "gray"},
                        'steps': [{'range': [0, item['Preço Justo']], 'color': "#d4edda"}],
                        'threshold': {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': item['Preço Justo']}
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)