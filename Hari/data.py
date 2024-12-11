import yfinance as yf
import pandas as pd

markets = {
    'NASDAQ': [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'NFLX', 'INTC',
        'PYPL', 'CSCO', 'CMCSA', 'PEP', 'AVGO', 'TXN', 'COST', 'QCOM', 'AMGN', 'CHTR',
        'SBUX', 'BIIB', 'MDLZ', 'INTU', 'ISRG', 'BKNG', 'VRTX', 'GILD', 'ADP', 'REGN'
    ],
    'NYSE': [
        'JNJ', 'XOM', 'GE', 'WMT', 'V', 'PG', 'RTX', 'GS', 'IBM', 'MMM',
        'T', 'BA', 'CVX', 'NKE', 'DIS', 'UNH', 'HD', 'MCD', 'LLY', 'DHR',
        'MRK', 'ACN', 'C', 'KO', 'ABBV', 'LIN', 'PLD', 'JD', 'UPS', 'BHP'
    ],
    'EuroNext': [
        'ABI.BR', 'MC.PA', 'SAN.PA', 'SU.PA', 'OR.PA', 'AI.PA', 'AIR.PA', 'BNP.PA', 'ENEL.MI', 'ENGI.PA',
        'KER.PA', 'SAF.PA', 'SGO.PA', 'ORA.PA', 'ENI.MI', 'RI.PA', 'DSY.PA', 'DG.PA', 'VIE.PA', 'GLE.PA',
        'CAP.PA', 'BN.PA', 'SW.PA', 'ML.PA', 'EL.PA', 'RMS.PA', 'LR.PA', 'FR.PA', 'STLA', 'VIV.PA'
    ],
    'Nikkei': [
        '7203.T', '9432.T', '6758.T', '9984.T', '8306.T', '8316.T', '8591.T', '8031.T', '8058.T', '4502.T',
        '4503.T', '4901.T', '6902.T', '7735.T', '6367.T', '6471.T', '6701.T', '6702.T', '6752.T', '7751.T',
        '7974.T', '9202.T', '7267.T', '7269.T', '7752.T', '8035.T', '6902.T', '4523.T', '8725.T', '8766.T', '7201.T'
    ]
}
start_date = '2023-01-01'  
end_date = '2023-05-20'   

def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

all_data = pd.DataFrame()

for market, companies in markets.items():
    market_data = fetch_data(companies, start_date, end_date)
    market_data.columns = [f"{market}_{ticker}" for ticker in market_data.columns]  # Prefix the market
    all_data = pd.concat([all_data, market_data], axis=1)

all_data.to_csv('stock_data.csv')

