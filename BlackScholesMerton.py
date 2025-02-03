import numpy as np
from scipy.stats import norm

# implementation of the solutions to the Black-Scholes-Merton equations as presented in John C. Hull - Options, Futures, and Other Derivatives
# S0 = spot price of underlying security at t0
# K = strike price of european-style option
# T = time to maturity scaled to one year excluding business days (252 days/year)
# r = risk free rate
# vol = volatility of underlying security


# implementation of d1 intermediary calculation
def calculated1(S_0, K, T, r, vol):
    d1 = (np.log(S_0 / K) + (r + (vol**2) / 2) * T) / (vol * np.sqrt(T))
    return d1


# implementation of d2 intermediary calculation
def calculated2(S_0, K, T, r, vol):
    d2 = calculated1(S_0, K, T, r, vol) - vol * np.sqrt(T)
    return d2


# calculation of call price
def calculateCallPrice(S_0, K, T, r, vol):
    d1 = calculated1(S_0, K, T, r, vol)
    d2 = calculated2(S_0, K, T, r, vol)
    ND1 = norm.cdf(d1)  # cdf of normal distribution from scipy
    ND2 = norm.cdf(d2)  # cdf of normal distribution from scipy
    price = S_0 * ND1 - K * np.exp(-r * T) * ND2  # solution per Hull
    return price


# calculation of put price
def calculatePutPrice(S_0, K, T, r, vol):
    d1 = calculated1(S_0, K, T, r, vol)
    d2 = calculated2(S_0, K, T, r, vol)
    ND1 = norm.cdf(-d1)  # cdf of normal distribution from scipy
    ND2 = norm.cdf(-d2)  # cdf of normal distribution from scipy
    price = K * np.exp(-r * T) * ND2 - S_0 * ND1  # solution per Hull
    return price


# calculating volatility of actual stock prices.
def calculateVolatility(prices, annualize=True):
    secReturns = np.log(
        np.array(prices[1:]) / np.array(prices[:-1])
    )  # Compute log returns
    volatility = np.std(secReturns, ddof=1)  # Calculate volatility

    if annualize:
        return volatility * np.sqrt(
            252
        )  # in many applications volatility is annualized (252 trading days in the standard year)

    return volatility


# test function
def test():
    S_0 = 50
    K = 50
    T = 0.5
    r = 0.05
    vol = 0.2
    print(f"Call price: {calculateCallPrice(S_0, K, T, r, vol)}")
    print(f"Put price: {calculatePutPrice(S_0, K, T, r, vol)}")
    print(f"Calculated d1: {calculated1(S_0, K, T, r, vol)}")
    print(f"Calculated d2: {calculated2(S_0, K, T, r, vol)}")
