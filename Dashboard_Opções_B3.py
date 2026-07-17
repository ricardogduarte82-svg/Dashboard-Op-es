import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si

st.set_page_config(page_title="Dashboard Opções B3", layout="wide")

# Estilização CSS para melhorar a interface
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
    }
    .greek-title {
        font-weight: bold;
        color: #1e3d59;
    }
</style>
""", unsafe_allow_html=True)

# --- FÓRMULAS BLACK-SCHOLES E GREGAS ---
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_price(S, K, T, r, sigma, option_type):
    if T <= 0:
        return max(0.0, S - K) if option_type == 'Call' else max(0.0, K - S)
    try:
        d_1 = d1(S, K, T, r, sigma)
        d_2 = d2(S, K, T, r, sigma)
        if option_type == 'Call':
            return (S * si.norm.cdf(d_1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d_2, 0.0, 1.0))
        else:
            return (K * np.exp(-r * T) * si.norm.cdf(-d_2, 0.0, 1.0) - S * si.norm.cdf(-d_1, 0.0, 1.0))
    except Exception:
        return 0.0

def bs_greeks(S, K, T, r, sigma, option_type):
    if T <= 0:
        return {'Delta': 0.0, 'Gamma': 0.0, 'Theta': 0.0, 'Vega': 0.0}
    try:
        d_1 = d1(S, K, T, r, sigma)
        d_2 = d2(S, K, T, r, sigma)
        
        gamma = si.norm.pdf(d_1, 0.0, 1.0) / (S * sigma * np.sqrt(T))
        vega = S * si.norm.pdf(d_1, 0.0, 1.0) * np.sqrt(T) / 100 
        
        if option_type == 'Call':
            delta = si.norm.cdf(d_1, 0.0, 1.0)
            theta = (- (S * sigma * si.norm.pdf(d_1, 0.0, 1.0)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d_2, 0.0, 1.0)) / 365
        else:
            delta = si.norm.cdf(d_1, 0.0, 1.0) - 1
            theta = (- (S * sigma * si.norm.pdf(d_1, 0.0, 1.0)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d_2, 0.0, 1.0)) / 365
        return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega}
    except Exception:
        return {'Delta': 0.0, 'Gamma': 0.0, 'Theta': 0.0, 'Vega': 0.0}

# --- ALGORITMO: RECONHECIMENTO AUTOMÁTICO DE ESTRATÉGIAS ---
def classificar_estrategia(legs):
    n = len(legs)
    if n == 0:
        return "Nenhuma perna adicionada"
        
    # Operações de 1 Perna (Direcionais ou Lançamento)
    if n == 1:
        leg = legs[0]
        if leg['Ação'] == 'Compra' and leg['Tipo'] == 'Call': return "Compra de Call (Call Seca)"
        if leg['Ação'] == 'Compra' and leg['Tipo'] == 'Put': return "Compra de Put (Put Seca)"
        if leg['Ação'] == 'Venda' and leg['Tipo'] == 'Call': return "Venda de Call (Lançamento a Descoberto)"
        if leg['Ação'] == 'Venda' and leg['Tipo'] == 'Put': return "Venda de Put (Lançamento a Descoberto)"
        
    # Operações de 2 Pernas (Travas, Straddles, Strangles)
    if n == 2:
        # Ordena as pernas pelo Strike para identificar travas corretamente
        l1, l2 = sorted(legs, key=lambda x: x['Strike'])
        
        # Travas com Calls
        if l1['Tipo'] == l2['Tipo'] == 'Call':
            if l1['Ação'] == 'Compra' and l2['Ação'] == 'Venda': return "Trava de Alta com Calls (Bull Call Spread)"
            if l1['Ação'] == 'Venda' and l2['Ação'] == 'Compra': return "Trava de Baixa com Calls (Bear Call Spread)"
        
        # Travas com Puts
        if l1['Tipo'] == l2['Tipo'] == 'Put':
            if l1['Ação'] == 'Compra' and l2['Ação'] == 'Venda': return "Trava de Alta com Puts (Bull Put Spread)"
            if l1['Ação'] == 'Venda' and l2['Ação'] == 'Compra': return "Trava de Baixa com Puts (Bear Put Spread)"
            
        # Operações Mistas (Call e Put)
        has_call = any(l['Tipo'] == 'Call' for l in legs)
        has_put = any(l['Tipo'] == 'Put' for l in legs)
        if has_call and has_put:
            call_leg = next(l for l in legs if l['Tipo'] == 'Call')
            put_leg = next(l for l in legs if l['Tipo'] == 'Put')
            if call_leg['Ação'] == put_leg['Ação'] == 'Compra':
                return "Straddle (Comprado)" if call_leg['Strike'] == put_leg['Strike'] else "Strangle (Comprado)"
            if call_leg['Ação'] == put_leg['Ação'] == 'Venda':
                return "Short Straddle (Vendido)" if call_leg['Strike'] == put_leg['Strike'] else "Short Strangle (Vendido)"

    # Operações mais complexas
    if n == 3:
        return "Operação de 3 pernas (Ex: Borboleta / Condor Incompleto)"
    if n >= 4:
        return f"Operação Estruturada Complexa ({n} pernas)"
        
    return "Estrutura Personalizada"

# --- ESTADO DA SESSÃO PARA AS PERNAS ---
if 'legs' not in st.session_state:
    st.session_state.legs = [
        {'Ticker': 'PETRK380', 'Tipo': 'Call', 'Ação': 'Compra', 'Strike': 38.00, 'Prêmio': 1.80, 'Qtd': 1000},
        {'Ticker': 'PETRK400', 'Tipo': 'Call', 'Ação': 'Venda', 'Strike': 40.00, 'Prêmio': 0.70, 'Qtd': 1000}
    ]

# --- CABEÇALHO SUPERIOR (DADOS DO ATIVO BASE) ---
st.title("📊 Dashboard Operações Opções - B3")
st.markdown("---")

col_ticker, col_spot, col_vol, col_selic, col_venc = st.columns(5)
with col_ticker:
    ticker_acao = st.text_input("Ticker da Ação", value="PETR4").upper()
with col_spot:
    spot_atual = st.number_input("Preço Spot Atual (R$)", value=38.50, step=0.10, format="%.2f")
with col_vol:
    vol_historica = st.number_input("Vol. Histórica (%)", value=32.0, step=1.0) / 100
with col_selic:
    selic = st.number_input("Taxa Selic (%)", value=10.5, step=0.1) / 100
with col_venc:
    dte_original = st.number_input("Dias Úteis até Vencimento", value=21, step=1)

st.markdown("---")

# --- DIVISÃO DA TELA: ESQUERDA (EXECUÇÃO) | DIREITA (SIMULAÇÃO) ---
col_esquerda, col_direita = st.columns([1, 1.2])

# ================= LADO ESQUERDO =================
with col_esquerda:
    st.header("🛒 Operação Executada")
    
    # Classificador Automático de Estratégia Substitui o Text Input
    nome_operacao = classificar_estrategia(st.session_state.legs)
    st.info(f"**Estratégia Identificada:** {nome_operacao}")
    
    with st.expander("➕ Adicionar Nova Opção (Perna)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            opt_ticker = st.text_input("Ticker da Opção", value="PETRK390").upper()
            opt_tipo = st.selectbox("Tipo de Opção", ["Call", "Put"])
        with c2:
            opt_strike = st.number_input("Strike (R$)", value=39.00, step=0.50, format="%.2f")
            opt_premio = st.number_input("Preço/Prêmio (R$)", value=1.10, step=0.05, format="%.2f")
        with c3:
            opt_acao = st.selectbox("Ação", ["Compra", "Venda"])
            opt_qtd = st.number_input("Quantidade", value=1000, step=100)
            
        if st.button("Inserir Perna na Estrutura"):
            st.session_state.legs.append({
                'Ticker': opt_ticker, 'Tipo': opt_tipo, 'Ação': opt_acao,
                'Strike': opt_strike, 'Prêmio': opt_premio, 'Qtd': opt_qtd
            })
            st.rerun()

    if len(st.session_state.legs) > 0:
        df_legs = pd.DataFrame(st.session_state.legs)
        st.subheader("Pernas Ativas")
        st.dataframe(df_legs, use_container_width=True)
        
        c_del, c_clear = st.columns(2)
        with c_del:
            idx_to_remove = st.number_input("Remover perna (Índice)", min_value=0, max_value=len(st.session_state.legs)-1, step=1, value=0)
            if st.button("🗑️ Remover Selecionada"):
                st.session_state.legs.pop(int(idx_to_remove))
                st.rerun()
        with c_clear:
            if st.button("❌ Limpar Toda a Operação"):
                st.session_state.legs = []
                st.rerun()
                
        custo_total = 0.0
        for leg in st.session_state.legs:
            direcao = -1.0 if leg['Ação'] == 'Compra' else 1.0
            custo_total += (leg['Prêmio'] * leg['Qtd']) * direcao
        
        tipo_custo = "Crédito Recebido" if custo_total > 0 else "Débito Pago (Custo)"
        st.metric(f"Resultado de Montagem ({tipo_custo})", f"R$ {abs(custo_total):,.2f}")
    else:
        st.info("Adicione opções para compor a sua operação estruturada.")

# ================= LADO DIREITO =================
with col_direita:
    st.header("🎛️ Simulador de Cenários e Payoff")
    
    if len(st.session_state.legs) > 0:
        st.subheader("Movimentar Variáveis de Mercado")
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            sim_spot = st.slider("Preço Spot Simulado", float(spot_atual * 0.7), float(spot_atual * 1.3), float(spot_atual), format="%.2f")
        with s_col2:
            sim_dte = st.slider("Dias para Vencimento Simulados", 0, int(dte_original), int(dte_original))
        with s_col3:
            sim_vol = st.slider("Vol. Implícita Simulada (%)", 10.0, 100.0, float(vol_historica * 100)) / 100

        T_sim = sim_dte / 252.0 
        
        total_delta = total_gamma = total_theta = total_vega = 0.0
        
        for leg in st.session_state.legs:
            sinal = 1.0 if leg['Ação'] == 'Compra' else -1.0
            gregas = bs_greeks(sim_spot, leg['Strike'], T_sim, selic, sim_vol, leg['Tipo'])
            
            total_delta += gregas['Delta'] * leg['Qtd'] * sinal
            total_gamma += gregas['Gamma'] * leg['Qtd'] * sinal
            total_theta += gregas['Theta'] * leg['Qtd'] * sinal
            total_vega += gregas['Vega'] * leg['Qtd'] * sinal

        st.subheader("🇬🇷 Gregas Consolidadas da Operação")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Delta do Portfólio", f"{total_delta:,.2f}")
        g2.metric("Gamma", f"{total_gamma:,.4f}")
        g3.metric("Theta (por dia)", f"R$ {total_theta:,.2f}")
        g4.metric("Vega (1% vol)", f"R$ {total_vega:,.2f}")

        precos_teste = np.linspace(spot_atual * 0.8, spot_atual * 1.2, 150)
        payoff_vencimento = np.zeros_like(precos_teste)
        payoff_hoje = np.zeros_like(precos_teste)
        
        for leg in st.session_state.legs:
            sinal = 1.0 if leg['Ação'] == 'Compra' else -1.0
            
            for idx, p in enumerate(precos_teste):
                if leg['Tipo'] == 'Call':
                    val_venc = max(0.0, p - leg['Strike'])
                else:
                    val_venc = max(0.0, leg['Strike'] - p)
                
                payoff_vencimento[idx] += ((val_venc - leg['Prêmio']) * sinal) * leg['Qtd']
                
                p_teorico = bs_price(p, leg['Strike'], T_sim, selic, sim_vol, leg['Tipo'])
                payoff_hoje[idx] += ((p_teorico - leg['Prêmio']) * sinal) * leg['Qtd']

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(precos_teste, payoff_vencimento, label="No Vencimento (DTE = 0)", color="#1e3d59", linewidth=2.5)
        ax.plot(precos_teste, payoff_hoje, label=f"Simulado (DTE = {sim_dte})", color="#ff6e40", linestyle="--", linewidth=2)
        
        ax.axhline(0, color="grey", linestyle="-", alpha=0.5)
        ax.axvline(spot_atual, color="green", linestyle=":", label=f"Spot Atual ({spot_atual})")
        ax.axvline(sim_spot, color="purple", linestyle=":", label=f"Spot Simulado ({sim_spot:.2f})")
        
        ax.fill_between(precos_teste, payoff_vencimento, 0, where=(payoff_vencimento >= 0), color="green", alpha=0.15)
        ax.fill_between(precos_teste, payoff_vencimento, 0, where=(payoff_vencimento < 0), color="red", alpha=0.15)
        
        ax.set_title(f"Gráfico de Payoff - {nome_operacao}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Preço do Ativo Objeto (R$)")
        ax.set_ylabel("Lucro / Prejuízo Estimado (R$)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        
    else:
        st.warning("Adicione pelo menos uma perna para habilitar o Simulador e o Payoff.")
