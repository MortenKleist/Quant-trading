import numpy as np
import pandas as pd
import pytest
from models import gbm

def test_shape_gbm():
    n_paths = 20
    n_steps = 50
    prices = gbm.simulate_gbm(s0=1, m=0.2, sigma=0.1, n_steps=n_steps, n_paths=n_paths, dt=1/250,seed=42)
    assert prices.shape == (n_paths, n_steps + 1)

def test_random_seed():
    n_paths = 20
    n_steps = 50
    prices1 = gbm.simulate_gbm(s0=1, m=0.2, sigma=0.1, n_steps=n_steps, n_paths=n_paths, dt=1/250,seed=42)
    prices2 = gbm.simulate_gbm(s0=1, m=0.2, sigma=0.1, n_steps=n_steps, n_paths=n_paths, dt=1/250,seed=44)
    prices3 = gbm.simulate_gbm(s0=1, m=0.2, sigma=0.1, n_steps=n_steps, n_paths=n_paths, dt=1/250,seed=44)
    assert not np.array_equal(prices1, prices2)
    assert np.array_equal(prices2, prices3)

def test_initial_value():
    n_paths = 20
    n_steps = 50
    prices1 = gbm.simulate_gbm(s0=1, m=0.2, sigma=0.1, n_steps=n_steps, n_paths=n_paths, dt=1/250,seed=42)
    prices2 = gbm.simulate_gbm(s0=1, m=0.2, sigma=0.1, n_steps=n_steps, n_paths=n_paths, dt=1/250,seed=44)
    assert np.array_equal(prices1[:, 0], prices2[:, 0])

def test_end_mean_value():
    n_paths = 10000
    n_steps = 500
    dt = 1/250
    s0=10
    m = 0.2
    sigma = 0.2
    prices1 = gbm.simulate_gbm(s0=s0, m=m, sigma=sigma, n_steps=n_steps, n_paths=n_paths, dt=dt,seed=42)

    T = n_steps * dt
    expected = s0 * np.exp(m*T + sigma**2*T/2)
    result = prices1[:, -1].mean()

    var = s0**2*np.exp(2*m*T + sigma**2*T)*(np.exp(sigma**2*T)-1)
    standard_error = np.sqrt(var)/np.sqrt(n_paths)
    assert result == pytest.approx(expected, abs = 4 * standard_error)