import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math
from math import log, sqrt
from py_vollib.black_scholes.greeks.analytical import delta as bs_delta
from py_vollib.black_scholes.greeks.analytical import gamma as bs_gamma
from py_vollib.black_scholes.greeks.analytical import vega as bs_vega
import re
import time  # <-- new import

st.set_page_config(layout="wide")

# -------------------------------
# Helper Funments
# -------------------------------
def extract_expiry_from_contract(contract_symbol):
    """
    Extracts the expiration date from an option contract symbol.
    Handles both 6-digit (YYMMDD) and 8-digit (YYYYMMDD) date formats.
    """
    pattern = r'[A-Z]+W?(?P<date>\d{6}|\d{8})[CP]\d+'
    match = re.search(pattern, contract_symbol)
    if match:
        date_str = match.group("date")
        try:
            if len(date_str) == 6:
                # Parse as YYMMDD
                expiry_date = datetime.strptime(date_str, "%y%m%d").date()
            else:
                # Parse as YYYYMMDD
                expiry_date = datetime.strptime(date_str, "%Y%m%d").date()
            return expiry_date
        except ValueError:
            return None
    return None

def add_current_price_line(fig, current_price):
    """
    Adds a vertical dashed white line at the current price to a Plotly figure.
    """
    fig.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7,
        annotation_text=f"{current_price}",
        annotation_position="top",
    )
    return fig

# -------------------------------
# New function: Fetch all options and add extracted expiry column
# -------------------------------
def fetch_all_options(ticker):
    """
    Fetches option chains for all available expirations for the given ticker.
    Iterates over each expiry available in ticker.options.
    If ticker.options is empty (as for SPX), a fallback expiry is used.
    Returns two DataFrames: one for calls and one for puts, with an added column 'extracted_expiry'.
    """
    stock = yf.Ticker(ticker)
    all_calls = []
    all_puts = []
    
    if stock.options:
        for exp in stock.options:
            try:
                chain = stock.option_chain(exp)
                calls = chain.calls
                puts = chain.puts
                if not calls.empty:
                    calls = calls.copy()
                    calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
                    all_calls.append(calls)
                if not puts.empty:
                    puts = puts.copy()
                    puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
                    all_puts.append(puts)
            except Exception as e:
                st.error(f"Error fetching chain for expiry {exp}: {e}")
                continue
    else:
        # Fallback for tickers like SPX which return an empty options list.
        current_date = datetime.now().date()
        default_expiry = current_date
        try:
            chain = stock.option_chain(default_expiry)
            calls = chain.calls
            puts = chain.puts
            if not calls.empty:
                calls = calls.copy()
                calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
                all_calls.append(calls)
            if not puts.empty:
                puts = puts.copy()
                puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
                all_puts.append(puts)
        except Exception as e:
            st.error(f"Error fetching fallback options data for expiry {default_expiry}: {e}")
    
    if all_calls:
        combined_calls = pd.concat(all_calls, ignore_index=True)
    else:
        combined_calls = pd.DataFrame()
    if all_puts:
        combined_puts = pd.concat(all_puts, ignore_index=True)
    else:
        combined_puts = pd.DataFrame()
    
    return combined_calls, combined_puts

# =========================================
# 2) Existing Visualization Functions
# =========================================
def create_oi_volume_charts(calls, puts):
     # Get underlying price
    stock = yf.Ticker(ticker)
    S = stock.info.get("regularMarketPrice")
    if S is None:
        S = stock.fast_info.get("lastPrice")
    if S is None:
        st.error("Could not fetch underlying price.")
        return

    calls_df = calls[['strike', 'openInterest', 'volume']].copy()
    calls_df['OptionType'] = 'Call'
    
    puts_df = puts[['strike', 'openInterest', 'volume']].copy()
    puts_df['OptionType'] = 'Put'
    
    combined = pd.concat([calls_df, puts_df], ignore_index=True)
    combined.sort_values(by='strike', inplace=True)
    
    fig_oi = px.bar(
        combined,
        x='strike',
        y='openInterest',
        color='OptionType',
        title='Open Interest by Strike',
        barmode='group',
    )
    fig_oi.update_layout(
        xaxis_title='Strike Price',
        yaxis_title='Open Interest',
        hovermode='x unified'
    )
    fig_oi.update_xaxes(rangeslider=dict(visible=True))
    
    fig_volume = px.bar(
        combined,
        x='strike',
        y='volume',
        color='OptionType',
        title='Volume by Strike',
        barmode='group',
    )
    fig_volume.update_layout(
        xaxis_title='Strike Price',
        yaxis_title='Volume',
        hovermode='x unified'
    )
    fig_volume.update_xaxes(rangeslider=dict(visible=True))
    
    # Add current price line
    S = round(S, 2)
    fig_oi = add_current_price_line(fig_oi, S)
    fig_volume = add_current_price_line(fig_volume, S)
    
    
    return fig_oi, fig_volume

def create_heatmap(calls, puts, value='volume'):
    calls_df = calls[['strike', 'openInterest', 'volume']].copy()
    calls_df['OptionType'] = 'Call'
    
    puts_df = puts[['strike', 'openInterest', 'volume']].copy()
    puts_df['OptionType'] = 'Put'
    
    combined = pd.concat([calls_df, puts_df], ignore_index=True)
    pivot_df = combined.pivot(index='strike', columns='OptionType', values=value).fillna(0)
    pivot_df.sort_index(inplace=True)
    
    fig = px.imshow(
        pivot_df,
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale='Blues',
        aspect='auto',
        labels=dict(
            x="Option Type",
            y="Strike",
            color=value.title()
        ),
        title=f"{value.title()} Heatmap (Calls vs Puts)"
    )
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(hovermode='closest')
    
    return fig

def create_donut_chart(call_volume, put_volume):
    labels = ['Calls', 'Puts']
    values = [call_volume, put_volume]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(
        title_text='Call vs Put Volume Ratio',
        showlegend=True
    )
    fig.update_traces(hoverinfo='label+percent+value')
    return fig

def create_gex_bubble_chart(calls, puts):
    calls_gex = calls[['strike', 'gamma', 'openInterest']].copy()
    calls_gex['GEX'] = calls_gex['gamma'] * calls_gex['openInterest'] * 100
    calls_gex['Type'] = 'Call'
    
    puts_gex = puts[['strike', 'gamma', 'openInterest']].copy()
    puts_gex['GEX'] = puts_gex['gamma'] * puts_gex['openInterest'] * 100
    puts_gex['Type'] = 'Put'
    
    gex_df = pd.concat([calls_gex, puts_gex], ignore_index=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gex_df.loc[gex_df['Type'] == 'Call', 'strike'],
        y=gex_df.loc[gex_df['Type'] == 'Call', 'GEX'],
        mode='markers',
        name='Calls',
        marker=dict(
            size=gex_df.loc[gex_df['Type'] == 'Call', 'GEX'].abs() / 1000,
            color='green',
            opacity=0.6,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        hovertemplate='Strike: %{x}<br>Gamma Exp: %{y}'
    ))
    fig.add_trace(go.Scatter(
        x=gex_df.loc[gex_df['Type'] == 'Put', 'strike'],
        y=gex_df.loc[gex_df['Type'] == 'Put', 'GEX'],
        mode='markers',
        name='Puts',
        marker=dict(
            size=gex_df.loc[gex_df['Type'] == 'Put', 'GEX'].abs() / 1000,
            color='red',
            opacity=0.6,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        hovertemplate='Strike: %{x}<br>Gamma Exp: %{y}'
    ))
    fig.update_layout(
        title='Gamma Exposure (GEX) Bubble Chart',
        xaxis_title='Strike Price',
        yaxis_title='Gamma Exposure',
        hovermode='closest',
        showlegend=True
    )
    return fig

# =========================================
# 3) Greek Calculation Helper Function
# =========================================
def calculate_greeks(flag, S, K, t, sigma):
    """
    Calculate delta, gamma and vanna for an option.
    t: time to expiration in years.
    flag: 'c' for call, 'p' for put.
    """
    try:
        d1 = (log(S / K) + (0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        delta_val = bs_delta(flag, S, K, t, 0, sigma)  # Risk-free rate set to 0
        gamma_val = bs_gamma(flag, S, K, t, 0, sigma)  # Risk-free rate set to 0
        vega_val = bs_vega(flag, S, K, t, 0, sigma)  # Risk-free rate set to 0
        vanna_val = -vega_val * d2 / (S * sigma * sqrt(t))
        return delta_val, gamma_val, vanna_val
    except Exception as e:
        st.error(f"Error calculating greeks: {e}")
        return None, None, None

# =========================================
# 4) Streamlit App Navigation
# =========================================
st.title("Real-Time Stock Options Data")

# Helper function to format tickers for indices
def format_ticker(ticker):
    ticker = ticker.upper()
    if ticker == "SPX":
        return "%5ESPX"
    elif ticker == "NDX":
        return "%5ENDX"
    elif ticker == "VIX":
        return "^VIX"
    return ticker

# ------------------------------------------------------------------
# A) OPTIONS DATA PAGE
# ------------------------------------------------------------------
if page == "Options Data":
    st.write("**Select filters below to see updated data, charts, and tables.**")
    user_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, SPX, NDX):", "SPY")
    ticker = format_ticker(user_ticker)
    if ticker:
        calls, puts = fetch_all_options(ticker)
        if calls.empty and puts.empty:
            st.warning("No options data available for this ticker.")
        else:
            combined = pd.concat([calls, puts])
            combined = combined.dropna(subset=['extracted_expiry'])
            unique_exps = sorted({d for d in combined['extracted_expiry'].unique() if d is not None})
            if not unique_exps:
                st.error("No expiration dates could be extracted from contract symbols.")
            else:
                unique_exps_str = [d.strftime("%Y-%m-%d") for d in unique_exps]
                expiry_date_str = st.selectbox("Select an Expiry Date (extracted from contract symbols):", options=unique_exps_str)
                selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
                calls = calls[calls['extracted_expiry'] == selected_expiry]
                puts = puts[puts['extracted_expiry'] == selected_expiry]
                if calls.empty and puts.empty:
                    st.warning("No options data found for the selected expiry.")
                else:
                    min_strike = float(min(calls['strike'].min(), puts['strike'].min()))
                    max_strike = float(max(calls['strike'].max(), puts['strike'].max()))
                    strike_range = st.slider(
                        "Select Strike Range:",
                        min_value=min_strike,
                        max_value=max_strike,
                        value=(min_strike, max_strike),
                        step=1.0
                    )
                    volume_over_oi = st.checkbox("Show only rows where Volume > Open Interest")
                    min_selected, max_selected = strike_range
                    calls_filtered = calls[(calls['strike'] >= min_selected) & (calls['strike'] <= max_selected)].copy()
                    puts_filtered = puts[(puts['strike'] >= min_selected) & (puts['strike'] <= max_selected)].copy()
                    if volume_over_oi:
                        calls_filtered = calls_filtered[calls_filtered['volume'] > calls_filtered['openInterest']]
                        puts_filtered = puts_filtered[puts_filtered['volume'] > puts_filtered['openInterest']]
                    if calls_filtered.empty and puts_filtered.empty:
                        st.warning("No data left after applying filters.")
                    else:
                        charts_container = st.container()
                        heatmaps_container = st.container()
                        tables_container = st.container()
                        with charts_container:
                            st.subheader(f"Options Data for {ticker} (Expiry: {expiry_date_str})")
                            if not calls_filtered.empty and not puts_filtered.empty:
                                fig_oi, fig_volume = create_oi_volume_charts(calls_filtered, puts_filtered)
                                st.plotly_chart(fig_oi, use_container_width=True, key="oi_chart")
                                st.plotly_chart(fig_volume, use_container_width=True, key="vol_chart")
                            else:
                                st.warning("No data to chart for the chosen filters.")
                        with heatmaps_container:
                            if not calls_filtered.empty and not puts_filtered.empty:
                                st.write("### Heatmaps")
                                volume_heatmap = create_heatmap(calls_filtered, puts_filtered, value='volume')
                                oi_heatmap = create_heatmap(calls_filtered, puts_filtered, value='openInterest')
                                st.plotly_chart(volume_heatmap, use_container_width=True, key="vol_heatmap")
                                st.plotly_chart(oi_heatmap, use_container_width=True, key="oi_heatmap")
                        with tables_container:
                            st.write("### Filtered Data Tables")
                            if not calls_filtered.empty:
                                st.write("**Calls Table**")
                                st.dataframe(calls_filtered)
                            else:
                                st.write("No calls match filters.")
                            if not puts_filtered.empty:
                                st.write("**Puts Table**")
                                st.dataframe(puts_filtered)
                            else:
                                st.write("No puts match filters.")


elif page == "Volume Ratio":
    user_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL", key="greek_ticker")
    ticker = format_ticker(user_ticker)
    if ticker:
        calls, puts = fetch_all_options(ticker)
        if calls.empty and puts.empty:
            st.warning("No options data available for this ticker.")
        else:
            combined = pd.concat([calls, puts])
            combined = combined.dropna(subset=['extracted_expiry'])
            unique_exps = sorted({d for d in combined['extracted_expiry'].unique() if d is not None})
            if not unique_exps:
                st.error("No expiration dates could be extracted from contract symbols.")
            else:
                unique_exps_str = [d.strftime("%Y-%m-%d") for d in unique_exps]
                expiry_date_str = st.selectbox("Select an Expiry Date (extracted from contract symbols):", options=unique_exps_str)
                selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
                calls = calls[calls['extracted_expiry'] == selected_expiry]
                puts = puts[puts['extracted_expiry'] == selected_expiry]
                if calls.empty and puts.empty:
                    st.warning("No options data found for the selected expiry.")
                else:
                    call_volume = calls['volume'].sum()
                    put_volume = puts['volume'].sum()
                    fig = create_donut_chart(call_volume, put_volume)
                    st.plotly_chart(fig, use_container_width=True, key="donut_chart")
                    st.markdown(f"**Total Call Volume:** {call_volume}")
                    st.markdown(f"**Total Put Volume:** {put_volume}")


elif page == "Gamma Exposure":
    user_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, SPX):", "AAPL", key="gamma_ticker")
    ticker = format_ticker(user_ticker)  # Apply formatting here
    
    if ticker:
        calls, puts = fetch_all_options(ticker)
        if calls.empty and puts.empty:
            st.warning("No options data available for this ticker.")
        else:
            combined = pd.concat([calls, puts])
            combined = combined.dropna(subset=['extracted_expiry'])
            unique_exps = sorted({d for d in combined['extracted_expiry'].unique() if d is not None})
            if not unique_exps:
                st.error("No expiration dates could be extracted from contract symbols.")
            else:
                unique_exps_str = [d.strftime("%Y-%m-%d") for d in unique_exps]
                expiry_date_str = st.selectbox("Select an Expiry Date (extracted from contract symbols):", options=unique_exps_str, key="gamma_expiry")
                selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
                calls = calls[calls['extracted_expiry'] == selected_expiry]
                puts = puts[puts['extracted_expiry'] == selected_expiry]
                
                # Get underlying price
                stock = yf.Ticker(ticker)
                S = stock.info.get("regularMarketPrice")
                if S is None:
                    S = stock.fast_info.get("lastPrice")
                if S is None:
                    st.error("Could not fetch underlying price.")
                else:
                    S = round(S, 2)
                    st.markdown(f"**Underlying Price (S):** {S}")
                    today = datetime.today().date()
                    t_days = (selected_expiry - today).days
                    if t_days <= 0:
                        st.error("The selected expiration date is in the past!")
                    else:
                        t = t_days / 365.0
                        st.markdown(f"**Time to Expiration (t in years):** {t:.4f}")
                        
                        def compute_gamma(row, flag):
                            sigma = row.get("impliedVolatility", None)
                            if sigma is None or sigma <= 0:
                                return None
                            try:
                                _, gamma_val, _ = calculate_greeks(flag, S, row["strike"], t, sigma)
                                return gamma_val
                            except Exception:
                                return None
                        
                        calls = calls.copy()
                        puts = puts.copy()
                        calls["calc_gamma"] = calls.apply(lambda row: compute_gamma(row, "c"), axis=1)
                        puts["calc_gamma"] = puts.apply(lambda row: compute_gamma(row, "p"), axis=1)
                        
                        calls = calls.dropna(subset=["calc_gamma"])
                        puts = puts.dropna(subset=["calc_gamma"])
                        
                        calls["GEX"] = calls["calc_gamma"] * calls["openInterest"] * 100
                        puts["GEX"] = puts["calc_gamma"] * puts["openInterest"] * 100

                        
                        # Grouped Bar Chart for Gamma Exposure by Strike
                        def create_gex_bar_chart(calls, puts):
                            calls_df = calls[['strike', 'GEX']].copy()
                            calls_df['OptionType'] = 'Call'
                            puts_df = puts[['strike', 'GEX']].copy()
                            puts_df['OptionType'] = 'Put'
                            combined_chart = pd.concat([calls_df, puts_df], ignore_index=True)
                            combined_chart.sort_values(by='strike', inplace=True)
                            fig = px.bar(
                                combined_chart,
                                x='strike',
                                y='GEX',
                                color='OptionType',
                                title='Gamma Exposure by Strike',
                                barmode='group'
                            )
                            fig.update_layout(
                                xaxis_title='Strike Price',
                                yaxis_title='Gamma Exposure',
                                hovermode='x unified'
                            )
                            fig.update_xaxes(rangeslider=dict(visible=True))


                            fig = add_current_price_line(fig, S)  # Add price line

                            return fig
                        
                        fig_bar = create_gex_bar_chart(calls, puts)
                        st.plotly_chart(fig_bar, use_container_width=True)



elif page == "Delta Exposure":
    user_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, SPX):", "AAPL", key="delta_ticker")
    ticker = format_ticker(user_ticker)  # Apply formatting here
    
    if ticker:
        calls, puts = fetch_all_options(ticker)
        if calls.empty and puts.empty:
            st.warning("No options data available for this ticker.")
        else:
            combined = pd.concat([calls, puts])
            combined = combined.dropna(subset=['extracted_expiry'])
            unique_exps = sorted({d for d in combined['extracted_expiry'].unique() if d is not None})
            if not unique_exps:
                st.error("No expiration dates could be extracted from contract symbols.")
            else:
                unique_exps_str = [d.strftime("%Y-%m-%d") for d in unique_exps]
                expiry_date_str = st.selectbox("Select an Expiry Date (extracted from contract symbols):", options=unique_exps_str, key="delta_expiry")
                selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
                calls = calls[calls['extracted_expiry'] == selected_expiry]
                puts = puts[puts['extracted_expiry'] == selected_expiry]
                
                # Get underlying price
                stock = yf.Ticker(ticker)
                S = stock.info.get("regularMarketPrice")
                if S is None:
                    S = stock.fast_info.get("lastPrice")
                if S is None:
                    st.error("Could not fetch underlying price.")
                else:
                    S = round(S, 2)
                    st.markdown(f"**Underlying Price (S):** {S}")
                    today = datetime.today().date()
                    t_days = (selected_expiry - today).days
                    if t_days <= 0:
                        st.error("The selected expiration date is in the past!")
                    else:
                        t = t_days / 365.0
                        st.markdown(f"**Time to Expiration (t in years):** {t:.4f}")
                        
                        def compute_delta(row, flag):
                            sigma = row.get("impliedVolatility", None)
                            if sigma is None or sigma <= 0:
                                return None
                            try:
                                delta_val, _, _ = calculate_greeks(flag, S, row["strike"], t, sigma)
                                return delta_val
                            except Exception:
                                return None
                        
                        calls = calls.copy()
                        puts = puts.copy()
                        calls["calc_delta"] = calls.apply(lambda row: compute_delta(row, "c"), axis=1)
                        puts["calc_delta"] = puts.apply(lambda row: compute_delta(row, "p"), axis=1)
                        
                        calls = calls.dropna(subset=["calc_delta"])
                        puts = puts.dropna(subset=["calc_delta"])
                        
                        calls["DEX"] = calls["calc_delta"] * calls["openInterest"] * 100
                        puts["DEX"] = puts["calc_delta"] * puts["openInterest"] * 100

                        
                        # Grouped Bar Chart for Delta Exposure by Strike
                        def create_dex_bar_chart(calls, puts):
                            calls_df = calls[['strike', 'DEX']].copy()
                            calls_df['OptionType'] = 'Call'
                            puts_df = puts[['strike', 'DEX']].copy()
                            puts_df['OptionType'] = 'Put'
                            combined_chart = pd.concat([calls_df, puts_df], ignore_index=True)
                            combined_chart.sort_values(by='strike', inplace=True)
                            fig = px.bar(
                                combined_chart,
                                x='strike',
                                y='DEX',
                                color='OptionType',
                                title='Delta Exposure by Strike',
                                barmode='group'
                            )
                            fig.update_layout(
                                xaxis_title='Strike Price',
                                yaxis_title='Delta Exposure',
                                hovermode='x unified'
                            )
                            fig.update_xaxes(rangeslider=dict(visible=True))

                            fig = add_current_price_line(fig, S)  # Add price line
                            return fig
                        
                        fig_bar = create_dex_bar_chart(calls, puts)
                        st.plotly_chart(fig_bar, use_container_width=True)


elif page == "Vanna Exposure":
    user_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, SPX):", "AAPL", key="vanna_ticker")
    ticker = format_ticker(user_ticker)  # Apply formatting here
    
    if ticker:
        calls, puts = fetch_all_options(ticker)
        if calls.empty and puts.empty:
            st.warning("No options data available for this ticker.")
        else:
            combined = pd.concat([calls, puts])
            combined = combined.dropna(subset=['extracted_expiry'])
            unique_exps = sorted({d for d in combined['extracted_expiry'].unique() if d is not None})
            if not unique_exps:
                st.error("No expiration dates could be extracted from contract symbols.")
            else:
                unique_exps_str = [d.strftime("%Y-%m-%d") for d in unique_exps]
                expiry_date_str = st.selectbox("Select an Expiry Date (extracted from contract symbols):", options=unique_exps_str, key="vanna_expiry")
                selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
                calls = calls[calls['extracted_expiry'] == selected_expiry]
                puts = puts[puts['extracted_expiry'] == selected_expiry]
                
                # Get underlying price
                stock = yf.Ticker(ticker)
                S = stock.info.get("regularMarketPrice")
                if S is None:
                    S = stock.fast_info.get("lastPrice")
                if S is None:
                    st.error("Could not fetch underlying price.")
                else:
                    S = round(S, 2)
                    st.markdown(f"**Underlying Price (S):** {S}")
                    today = datetime.today().date()
                    t_days = (selected_expiry - today).days
                    if t_days <= 0:
                        st.error("The selected expiration date is in the past!")
                    else:
                        t = t_days / 365.0
                        st.markdown(f"**Time to Expiration (t in years):** {t:.4f}")
                        
                        def compute_vanna(row, flag):
                            sigma = row.get("impliedVolatility", None)
                            if sigma is None or sigma <= 0:
                                return None
                            try:
                                _, _, vanna_val = calculate_greeks(flag, S, row["strike"], t, sigma)
                                return vanna_val
                            except Exception:
                                return None
                        
                        calls = calls.copy()
                        puts = puts.copy()
                        calls["calc_vanna"] = calls.apply(lambda row: compute_vanna(row, "c"), axis=1)
                        puts["calc_vanna"] = puts.apply(lambda row: compute_vanna(row, "p"), axis=1)
                        
                        calls = calls.dropna(subset=["calc_vanna"])
                        puts = puts.dropna(subset=["calc_vanna"])
                        
                        calls["VEX"] = calls["calc_vanna"] * calls["openInterest"] * 100
                        puts["VEX"] = puts["calc_vanna"] * puts["openInterest"] * 100

                        
                        # Grouped Bar Chart for Vanna Exposure by Strike
                        def create_vex_bar_chart(calls, puts):
                            calls_df = calls[['strike', 'VEX']].copy()
                            calls_df['OptionType'] = 'Call'
                            puts_df = puts[['strike', 'VEX']].copy()
                            puts_df['OptionType'] = 'Put'
                            combined_chart = pd.concat([calls_df, puts_df], ignore_index=True)
                            combined_chart.sort_values(by='strike', inplace=True)
                            fig = px.bar(
                                combined_chart,
                                x='strike',
                                y='VEX',
                                color='OptionType',
                                title='Vanna Exposure by Strike',
                                barmode='group'
                            )
                            fig.update_layout(
                                xaxis_title='Strike Price',
                                yaxis_title='Vanna Exposure',
                                hovermode='x unified'
                            )
                            fig.update_xaxes(rangeslider=dict(visible=True))

                            fig = add_current_price_line(fig, S)  # Add price line
                            return fig
                        
                        fig_bar = create_vex_bar_chart(calls, puts)
                        st.plotly_chart(fig_bar, use_container_width=True)

elif page == "Calculated Greeks":
    st.write("This page calculates delta, gamma, and vanna based on market data.")
    
    user_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL", key="greek_ticker")
    ticker = format_ticker(user_ticker)
    
    if ticker:
        calls, puts = fetch_all_options(ticker)
        if calls.empty and puts.empty:
            st.warning("No options data available for this ticker.")
        else:
            combined = pd.concat([calls, puts])
            combined = combined.dropna(subset=['extracted_expiry'])
            unique_exps = sorted({d for d in combined['extracted_expiry'].unique() if d is not None})
            if not unique_exps:
                st.error("No expiration dates could be extracted from contract symbols.")
            else:
                unique_exps_str = [d.strftime("%Y-%m-%d") for d in unique_exps]
                expiry_date_str = st.selectbox("Select an Expiry Date (extracted from contract symbols):", options=unique_exps_str, key="greek_expiry")
                selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
                calls = calls[calls['extracted_expiry'] == selected_expiry]
                puts = puts[puts['extracted_expiry'] == selected_expiry]
                
                stock = yf.Ticker(ticker)
                S = stock.info.get("regularMarketPrice")
                if S is None:
                    S = stock.fast_info.get("lastPrice")
                if S is None:
                    st.error("Could not fetch underlying price.")
                else:
                    S = round(S, 2)
                    st.markdown(f"**Underlying Price (S):** {S}")
                    today = datetime.today().date()
                    t_days = (selected_expiry - today).days
                    if t_days <= 0:
                        st.error("The selected expiration date is in the past!")
                    else:
                        t = t_days / 365.0
                        st.markdown(f"**Time to Expiration (t in years):** {t:.4f}")
                        
                        def compute_row_greeks(row, flag_char):
                            sigma = row.get("impliedVolatility", None)
                            if sigma is None or sigma <= 0:
                                return pd.Series({"calc_delta": None, "calc_gamma": None, "calc_vanna": None})
                            try:
                                delta_val, gamma_val, vanna_val = calculate_greeks(flag_char, S, row["strike"], t, sigma)
                                return pd.Series({
                                    "calc_delta": delta_val,
                                    "calc_gamma": gamma_val,
                                    "calc_vanna": vanna_val
                                })
                            except Exception:
                                return pd.Series({"calc_delta": None, "calc_gamma": None, "calc_vanna": None})
                        
                        results = {}
                        if not calls.empty:
                            calls = calls.copy()
                            calls["calc_flag"] = "c"
                            greeks_calls = calls.apply(lambda row: compute_row_greeks(row, "c"), axis=1)
                            calls = pd.concat([calls, greeks_calls], axis=1)
                            results["Calls"] = calls
                        else:
                            st.warning("No call options data available.")
                        if not puts.empty:
                            puts = puts.copy()
                            puts["calc_flag"] = "p"
                            greeks_puts = puts.apply(lambda row: compute_row_greeks(row, "p"), axis=1)
                            puts = pd.concat([puts, greeks_puts], axis=1)
                            results["Puts"] = puts
                        else:
                            st.warning("No put options data available.")
                        
                        for typ, df in results.items():
                            st.write(f"### {typ} with Calculated Greeks")
                            st.dataframe(df[['contractSymbol', 'strike', 'impliedVolatility', 'calc_delta', 'calc_gamma', 'calc_vanna']])
                            fig = px.scatter(df, x="strike", y="calc_delta", title=f"{typ}: Delta vs. Strike",
                                             labels={"strike": "Strike", "calc_delta": "Calculated Delta"})
                            st.plotly_chart(fig, use_container_width=True)




# -----------------------------------------
# Auto-refresh: rerun the app every 60 seconds
# -----------------------------------------
time.sleep(60)
st.rerun()
