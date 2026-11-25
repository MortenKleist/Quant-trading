import numpy as np

def simulate_gbm(s0, m, sigma, n_steps, n_paths, dt, seed):
    """
    Inputs:
    s0:
    m: drift of the log prices
    sigma: is the volatility
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, size=(n_paths, n_steps))
    log_returns = m*dt + sigma*np.sqrt(dt)*Z
    log_paths = np.cumsum(log_returns, axis=1)
    prices = s0 * np.exp(log_paths)
    prices = np.hstack([np.full((n_paths,1), s0), prices])
    return prices