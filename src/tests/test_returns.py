import numpy as np
import pandas as pd
import pytest
from utils import returns

def test_simple_return():
    prices = pd.Series([100.0, 110.0, 99.0])
    result = returns.simple_returns(prices)

    expected = pd.Series([float('nan'), 0.10, -0.10])
    pd.testing.assert_series_equal(result, expected)

def test_log_return():
    prices = pd.Series([100.0, 110.0, 99.0])
    result = returns.log_returns(prices)

    expected = pd.Series([float('nan'), np.log(110)-np.log(100), np.log(99) - np.log(110)])
    pd.testing.assert_series_equal(result, expected)
   
def test_cumulative_returns():
    prices = pd.Series([100.0, 110.0, 99.0])
    simple_result = returns.simple_returns(prices)
    log_result = returns.log_returns(prices)

    simple_cum_return = returns.cumulative_returns(simple_result)
    log_cum_return = returns.cumulative_returns(log_result, log=True)

    expected = pd.Series([1.0, 1.10, 0.99]) 

    pd.testing.assert_series_equal(simple_cum_return, expected)
    pd.testing.assert_series_equal(log_cum_return, expected)

def test_missing_days():
    prices = pd.Series([100.0, np.nan, 99.0, 105.0])
    simple_result = returns.simple_returns(prices)
    log_result = returns.log_returns(prices)

    simple_cum_return = returns.cumulative_returns(simple_result)
    log_cum_return = returns.cumulative_returns(log_result, log=True)

    expected = pd.Series([1.0, 1.0, 0.99, 1.05])

    pd.testing.assert_series_equal(simple_cum_return, expected)
    pd.testing.assert_series_equal(log_cum_return, expected)

def test_annualized_return():
    return_ = pd.Series([0.001]*252)
    anualized_return = returns.annualized_return(return_, period=252)
    expected = (1.001**252) - 1
    assert anualized_return == pytest.approx(expected)

def test_annualized_volatility_exact():
    log_returns_ = pd.Series([0.01, -0.01]*252)
    expected = 0.01*np.sqrt(252*2/(252*2 - 1)) * np.sqrt(252)
    result = returns.annualized_volatility(log_returns_, period=252)
    assert result == pytest.approx(expected)

def test_annualized_volatility_statistically():
    rng = np.random.default_rng(1)
    log_returns_ = pd.Series(rng.normal(0, 0.01, size = 100_000))
    expected = 0.01 * np.sqrt(252)
    result = returns.annualized_volatility(log_returns_, period=252)
    assert result == pytest.approx(expected, rel=3.5/(np.sqrt(2*100_000)))

def test_max_drawdown():
    curve = pd.Series([1.0, 1.2, 0.6, 0.9])
    expected = -0.5
    result = returns.max_drawdown(curve)
    assert result == pytest.approx(expected)

def test_max_drawdown_NaN():
    curve = pd.Series([np.nan, 1.2, 0.6, 0.9])
    expected = -0.5
    result = returns.max_drawdown(curve)
    assert result == pytest.approx(expected)

def test_rising_curve():
    curve = pd.Series([1.0, 1.2, 2.0, 2.3])
    expected = 0
    result = returns.max_drawdown(curve)
    assert result == pytest.approx(expected)

def test_falling_curve():
    curve = pd.Series([1.0, 0.8, 0.4, 0.3])
    expected = -0.7
    result = returns.max_drawdown(curve)
    assert result == pytest.approx(expected)