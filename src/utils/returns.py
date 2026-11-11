import numpy as np
import pandas as pd

def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Daily simple returns: P_t / P_{t-1} - 1.

    prices are forward-filled first, so NaNs in the middle become zero returns, and the first row is NaN
    """
    filled_prices = prices.ffill()
    return filled_prices / filled_prices.shift(1) - 1 

def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Daily log returns: ln(P_t / P_{t-1}).
    prices are forward-filled first, so NaNs in the middle become zero returns, and the first row is NaN"""

    filled_prices = prices.ffill()
    return np.log(filled_prices) - np.log(filled_prices.shift(1))

def cumulative_returns(returns, log: bool = False):
    """output is a growth factor starting at 1.0, missing returns are treated as 0"""
    if log:
        cum_return = np.exp((returns.fillna(0)).cumsum()) 
        return cum_return
    else:
        cum_return = (returns.fillna(0)+1).cumprod()
        return cum_return 

def annualized_return(returns: pd.DataFrame | pd.Series, period: int = 250, log: bool = False):
    """Geometric average growth per year
    (gross return)^(1/n) - 1
    n: # years held, n = len(returns)/250 (250 becuase using copenhagen nasdaq)
    
    """
    n = returns.count() / period
    cum_returns = cumulative_returns(returns=returns, log=log)
    return (cum_returns.iloc[-1])**(1/n) -1

def annualized_volatility(log_return: pd.DataFrame | pd.Series, period: int = 250):
    """Annualized volatility
    Varince of sum of independent variables with var sigma^2:
        X = sum(X_i), i in [1, 2, 3, ..., n]
        Var(X) = Var(sum(X_i)) = sum(Var(X_i)) = sum(sigma^2) = n * sigma^2
        => std(X) = sqrt(n) * sigma
    
    Input: log_returns is the daily log returns
    Choosing n: # days per year gives annualized volatility ie. ddof = 0
    Output: sqrt(n)*log_returns.std(ddof=n) is the annualized volatility
    """
    n = period
    return np.sqrt(n) * log_return.std()

def drawdown_series(curve: pd.DataFrame | pd.Series):
    """
    expects a curve with base 1.0, returns values ≤ 0
    """
    running_max = curve.cummax()
    return curve/(running_max) - 1

def max_drawdown(curve: pd.DataFrame | pd.Series):
    '''
    Returns the maximum drawdown using negative magnitudes
    '''
    drawdown = drawdown_series(curve)
    return drawdown.min()