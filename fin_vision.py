# Force use of patched sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# finance3.py
import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama.llms import OllamaLLM
import chromadb
from groq import Client
import os
import tempfile
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from st_aggrid import AgGrid
import time
import requests

# ====== Configuration ======
st.set_page_config(
    page_title="FinVision Pro", 
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== Premium Style ======
st.markdown("""
<style>
    @keyframes gradient {
        0% {background-position: 0% 50%}
        50% {background-position: 100% 50%}
        100% {background-position: 0% 50%}
    }
    
    .main {
        background: linear-gradient(-45deg, #f8f9fa, #ffffff);
        animation: gradient 15s ease infinite;
    }
    
    .metric-card {
        padding: 1.5rem;
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .positive {
        color: #2ecc71;
        font-weight: 700;
    }
    
    .negative {
        color: #e74c3c;
        font-weight: 700;
    }
    
    .action-plan {
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ====== Stock Data ======
STOCK_LIST = [
    ("AAPL", "Apple Inc.", "🖥️"),
    ("MSFT", "Microsoft Corporation", "💻"),
    ("GOOGL", "Alphabet Inc.", "🔍"),
    ("AMZN", "Amazon.com Inc.", "📦"),
    ("TSLA", "Tesla Inc.", "⚡"),
    ("NVDA", "NVIDIA Corporation", "🎮"),
    ("META", "Meta Platforms Inc.", "👥"),
    ("JPM", "JPMorgan Chase", "🏦"),
    ("V", "Visa Inc.", "💳"),
    ("DIS", "Walt Disney Co.", "🎥")
]

# ====== Main App ======
def main():
    st.title("💹 FinVision Pro")
    st.caption("AI-Powered Financial Intelligence Platform")

    # ====== Session State ======
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    if 'symbol' not in st.session_state:
        st.session_state.symbol = "AAPL"

    # ====== Sidebar ======
    with st.sidebar:
        st.header("⭐ Favorites")
        new_fav = st.text_input("Add symbol:", key="new_fav")
        if new_fav:
            cleaned_fav = new_fav.upper().strip()
            if cleaned_fav not in [f[0] for f in st.session_state.favorites]:
                match = next((s for s in STOCK_LIST if s[0] == cleaned_fav), None)
                if match:
                    st.session_state.favorites.append(match)
        
        if st.session_state.favorites:
            st.write("Favorite Stocks:")
            for sym, name, icon in st.session_state.favorites:
                if st.button(f"{icon} {sym} - {name}", key=f"fav_{sym}"):
                    st.session_state.symbol = sym
                    st.rerun()

    # ====== Stock Selector ======
    with st.container():
        col1, col2 = st.columns([3,1])
        with col1:
            search_term = st.text_input("🔍 Search stocks:", 
                                      value=st.session_state.symbol,
                                      placeholder="Search by company or symbol...")
        with col2:
            if st.button("🎲 Random Pick", use_container_width=True):
                selected = STOCK_LIST[np.random.randint(0, len(STOCK_LIST))]
                st.session_state.symbol = selected[0]
                st.rerun()

        if search_term:
            matches = [s for s in STOCK_LIST if search_term.upper() in s[0] or search_term.upper() in s[1].upper()]
            for sym, name, icon in matches[:8]:
                if st.button(f"{icon} **{sym}** - {name}", use_container_width=True):
                    st.session_state.symbol = sym
                    st.rerun()

    # ====== Main Content ======
    symbol = st.session_state.symbol
    ticker = yf.Ticker(symbol)
    info = ticker.info
    
    try:
        current_price = ticker.history(period='1d')['Close'].iloc[-1]
        prev_close = info.get('previousClose', current_price)
        delta = current_price - prev_close
        pct_change = (delta / prev_close) * 100
    except Exception as e:
        st.error("Could not fetch market data. Please try another symbol.")
        st.stop()

    # ====== Key Metrics ======
    with st.container():
        cols = st.columns(4)
        metrics = [
            ("💵 Current Price", f"${current_price:.2f}", "#2ecc71" if delta >=0 else "#e74c3c"),
            ("📈 52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}", "#2ecc71"),
            ("📉 52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}", "#e74c3c"),
            ("📊 Volume", f"{info.get('volume', 0):,}", "#3498db")
        ]
        
        for col, (title, value, color) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: {color}; font-size: 1.2rem; margin-bottom: 0.5rem">{title}</div>
                    <div style="font-size: 1.5rem; font-weight: 700">{value}</div>
                </div>
                """, unsafe_allow_html=True)

    # ====== Interactive Chart ======
    with st.container():
        st.header("📈 Price Analysis", divider="blue")
        hist_data = ticker.history(period="1y")
        fig = px.area(hist_data, x=hist_data.index, y='Close',
                     title=f"{symbol} Price Trend",
                     color_discrete_sequence=["#3498db"])
        fig.update_layout(template="plotly_white", height=400,
                         hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ====== Enhanced AI Analysis ======
    def generate_action_plan():
        client = Client(api_key=st.secrets["GROQ_API_KEY"])
        
        prompt = f"""
        As a top financial analyst, provide {symbol} analysis with clear action steps:
        
        1. Technical Analysis:
        - Key support/resistance levels
        - Momentum indicators (RSI, MACD)
        - Volume analysis
        - Chart patterns
        
        2. Fundamental Analysis:
        - Valuation metrics (P/E, P/S)
        - Growth projections
        - Competitive landscape
        - Management quality
        
        3. Risk Assessment:
        - Market risks
        - Sector risks
        - Company-specific risks
        
        4. Action Plan:
        - Recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
        - Entry price range
        - Price targets (3 levels with timelines)
        - Stop-loss levels
        - Position sizing (% of portfolio)
        
        5. Scenarios:
        - Bull case (best scenario)
        - Base case (likely scenario)
        - Bear case (worst scenario)
        
        Format with clear headers (##), bullet points, and emojis. 
        Use simple but professional language.
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a hedge fund manager providing institutional-grade analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content

    with st.container():
        st.header("🤖 AI Portfolio Manager", divider="blue")
        if st.button("🚀 Generate Action Plan", type="primary"):
            with st.spinner("🔭 Building comprehensive strategy..."):
                try:
                    analysis = generate_action_plan()
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.1rem; line-height: 1.6">
                        {analysis}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.download_button(
                        label="📥 Download Full Report",
                        data=analysis,
                        file_name=f"{symbol}_action_plan.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error("Strategic analysis unavailable. Please try again later.")

                    
        # ====== Document Analysis ======
    def black_scholes(S, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) if option_type == 'call' else K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        return {'price': float(price)}

    # ====== Options Analysis ======
    def get_options_chain(symbol):
        try:
            ticker = yf.Ticker(symbol)
            exp_dates = ticker.options
            if not exp_dates:
                raise ValueError(f"No options data for {symbol}")
            return exp_dates, ticker.option_chain(exp_dates[0])
        except Exception as e:
            st.error(f"Options error: {str(e)}")
            return None, None

    with st.container():
        st.header("📊 Options Lab", divider="blue")
        exp_dates, chain = get_options_chain(symbol)
        if chain:
            with st.expander("🧮 Options Calculator", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    option_type = st.selectbox("Contract Type", ["call", "put"])
                    risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.5) / 100
                with col2:
                    expiry = st.selectbox("Expiration Date", exp_dates)
                    days_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days
                with col3:
                    min_strike = float(chain.calls['strike'].min())
                    max_strike = float(chain.calls['strike'].max())
                    strike = st.slider("Strike Price", min_strike, max_strike, (min_strike + max_strike)/2)
                
                if st.button("💵 Calculate Premium", type="primary"):
                    volatility = chain.calls[chain.calls['strike'] == strike]['impliedVolatility'].iloc[0]
                    result = black_scholes(
                        current_price,
                        strike,
                        days_to_expiry/365,
                        risk_free,
                        volatility,
                        option_type
                    )
                    
                    cols = st.columns(2)
                    cols[0].metric("Theoretical Price", f"${result['price']:.2f}")
                    cols[1].metric("Implied Volatility", f"{volatility*100:.1f}%")

    with st.container():
        st.header("📄 Research Center", divider="blue")
        uploaded_files = st.file_uploader("Upload Financial Documents", type="pdf", accept_multiple_files=True)
        doc_query = st.text_input("Ask about documents:", placeholder="Search for risk factors or strategies...")
        
        if uploaded_files and doc_query:
            with st.status("🔍 Analyzing Documents...", expanded=True):
                try:
                    docs = []
                    for file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False) as tmp:
                            tmp.write(file.read())
                            loader = UnstructuredPDFLoader(tmp.name)
                            docs.extend(loader.load())
                        os.unlink(tmp.name)
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
                    chunks = text_splitter.split_documents(docs)
                    llm = OllamaLLM(model="mistral")
                    
                    cols = st.columns(2)
                    for i, chunk in enumerate(chunks[:4]):
                        with cols[i%2]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div style="color: #3498db; margin-bottom: 0.5rem">📌 Insight {i+1}</div>
                                {chunk.page_content[:200]}...
                            </div>
                            """, unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"Document processing error: {str(e)}")

if __name__ == "__main__":
    main()
