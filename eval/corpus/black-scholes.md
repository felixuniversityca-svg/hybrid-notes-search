# The Black-Scholes model

Black-Scholes is a closed-form way to price a European option. It takes the spot
price, the strike, the time to expiry, the risk-free rate, and the volatility of
the underlying, and returns a fair premium. The model assumes continuous trading,
no transaction costs, and log-normal returns with constant volatility. Those
assumptions rarely hold exactly, so traders quote the implied volatility that
makes the formula match the market price, and the famous volatility smile is the
market's way of admitting the constant-volatility assumption is wrong.
