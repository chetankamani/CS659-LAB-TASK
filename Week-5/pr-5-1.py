"""
Lab Assignment: Gaussian Hidden Markov Model on Financial Time Series

How to run:
    pip install yfinance hmmlearn pandas numpy matplotlib
    python hmm_financial_lab5.py

You can change the TICKER and DATE RANGE in the CONFIG section below.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# ======================
# CONFIG
# ======================
TICKER = "^NSEI"          # e.g. "^NSEI", "^BSESN", "AAPL", "TSLA", "^GSPC"
START_DATE = "2010-01-01" # start date for historical data
END_DATE = "2025-01-01"   # end date for historical data
N_STATES_MAIN = 2         # main model with 2 hidden states (low/high volatility)
FIGURES_DIR = "figures"   # where plots will be saved

# ======================
# UTILITY FUNCTIONS
# ======================
def ensure_figures_dir(path: str):
    """Create figures directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Created directory: {path}")
    else:
        print(f"[INFO] Using existing directory: {path}")

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download close prices from Yahoo Finance."""
    print(f"[INFO] Downloading data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError("No data downloaded. Check ticker or date range.")
    df = df[["Close"]].dropna()
    print(f"[INFO] Downloaded {len(df)} rows.")
    return df

def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns and drop NaN."""
    df = df.copy()
    df["LogReturn"] = np.log(df["Close"]).diff()
    df = df.dropna()
    print(f"[INFO] Computed log returns; {len(df)} rows after dropping NaN.")
    return df

def fit_hmm(returns: np.ndarray, n_states: int, random_state: int = 42) -> GaussianHMM:
    """Fit a Gaussian HMM with given number of states."""
    print(f"[INFO] Fitting Gaussian HMM with {n_states} states...")
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=random_state
    )
    model.fit(returns)
    print(f"[INFO] HMM training completed for {n_states} states.")
    return model

def describe_states(data: pd.DataFrame, model: GaussianHMM, n_states: int):
    """Print state means, std devs, frequencies, and expected durations."""
    print("\n=== HMM State Parameters (from model) ===")
    print("Means (per state):")
    print(model.means_)
    print("\nCovariances (per state):")
    print(model.covars_)
    print("\n=== Empirical State Statistics (from data) ===")
    for s in range(n_states):
        state_data = data.loc[data["State"] == s, "LogReturn"]
        mean = state_data.mean()
        std = state_data.std()
        freq = len(state_data) / len(data)
        print(f"\nState {s}:")
        print(f"  Mean log return: {mean:.6f}")
        print(f"  Std dev        : {std:.6f}")
        print(f"  Time fraction  : {freq:.3f}")
    print("\n=== Transition Matrix ===")
    print(model.transmat_)
    print("\n=== Expected Duration in Each State (in days, approx) ===")
    for s in range(n_states):
        p_stay = model.transmat_[s, s]
        if p_stay < 1.0:
            expected_duration = 1.0 / (1.0 - p_stay)
            print(f"State {s}: ~{expected_duration:.2f} days")
        else:
            print(f"State {s}: infinite (p_stay = 1.0, degenerate case)")

def plot_price_with_states(data: pd.DataFrame, n_states: int, ticker: str, out_path: str):
    """Plot close price with points colored by hidden state."""
    plt.figure(figsize=(12, 5))
    for s in range(n_states):
        mask = data["State"] == s
        plt.plot(
            data.index[mask],
            data["Close"][mask],
            linestyle='',
            marker='.',
            label=f"State {s}"
        )
    plt.title(f"{ticker} Close Price with Hidden States (HMM)")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved price/state plot to {out_path}")
    plt.close()

def plot_returns_with_states(data: pd.DataFrame, n_states: int, ticker: str, out_path: str):
    """Plot log returns with points colored by hidden state."""
    plt.figure(figsize=(12, 5))
    for s in range(n_states):
        mask = data["State"] == s
        plt.plot(
            data.index[mask],
            data["LogReturn"][mask],
            linestyle='',
            marker='.',
            label=f"State {s}"
        )
    plt.title(f"{ticker} Log Returns with Hidden States (HMM)")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved returns/state plot to {out_path}")
    plt.close()

def compute_bic(model: GaussianHMM, returns: np.ndarray) -> float:
    """Compute a simple BIC value for the fitted HMM model."""
    logL = model.score(returns)  # log-likelihood
    k = model.n_components
    n_features = returns.shape[1]
    n_trans_params = k * (k - 1)
    n_mean_params = k * n_features
    n_cov_params = k * (n_features * (n_features + 1) / 2.0)
    n_params = n_trans_params + n_mean_params + n_cov_params
    n_samples = returns.shape[0]
    bic = -2 * logL + n_params * np.log(n_samples)
    return bic

# ======================
# MAIN
# ======================
def main():
    ensure_figures_dir(FIGURES_DIR)
    # 1. Download and preprocess data
    df_prices = download_data(TICKER, START_DATE, END_DATE)
    df = compute_log_returns(df_prices)
    # 2. Prepare data for HMM
    returns = df["LogReturn"].values.reshape(-1, 1)
    # 3. Fit main HMM with 2 states
    model_2 = fit_hmm(returns, N_STATES_MAIN, random_state=42)
    hidden_states_2 = model_2.predict(returns)
    df["State"] = hidden_states_2
    # 4. Describe states
    describe_states(df, model_2, N_STATES_MAIN)
    # 5. Plots
    price_plot_path = os.path.join(FIGURES_DIR, "price_states_2state.png")
    returns_plot_path = os.path.join(FIGURES_DIR, "returns_states_2state.png")
    plot_price_with_states(df, N_STATES_MAIN, TICKER, price_plot_path)
    plot_returns_with_states(df, N_STATES_MAIN, TICKER, returns_plot_path)
    # 6. Future state probabilities (next-day regime forecast)
    last_state = hidden_states_2[-1]
    next_state_probs = model_2.transmat_[last_state, :]
    print("\n=== Next-Day Regime Probabilities (2-state model) ===")
    print(f"Current (last) inferred state: {last_state}")
    for s in range(N_STATES_MAIN):
        print(
            f"P(State {s} tomorrow | today in State {last_state}) "
            f"= {next_state_probs[s]:.4f}"
        )
    # 7. OPTIONAL: Compare with 3- and 4-state models using BIC
    print("\n=== Model Comparison (BIC) for different number of states ===")
    for k in [2, 3, 4]:
        model_k = fit_hmm(returns, n_states=k, random_state=42)
        bic_k = compute_bic(model_k, returns)
        print(f"k = {k} states -> BIC = {bic_k:.2f}")
    print("\n[INFO] Script finished successfully.")
    print("[INFO] Use the printed numbers and saved plots in your lab report.")

if __name__ == "__main__":
    main()
