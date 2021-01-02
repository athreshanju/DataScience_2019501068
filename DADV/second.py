import bs4 as bs
import requests
import datetime
import pandas_datareader.data as web

html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")

soup = bs.BeautifulSoup(html.text, features ='html.parser')

tickers = []
table = soup.find('table', {'class': 'wikitable sortable'})
# print(table)
# skip the first index
rows = table.findAll('tr')[1:]
for row in rows:
    ticker = row.findAll('td')[0].text
    # that -1 is to remove the new line character
    tickers.append(ticker[:-1])

print(tickers)    
# Start and end dates for historical data
start = datetime.datetime(2010, 4, 8)  # start date
end = datetime.datetime(2020, 12, 31) # end date


for i in range(0,len(tickers)): # for each key in the dictionary which represents a sector
    #myTickers = tickers[i] # find the tickers in that list
    for j in range(0,len(tickers)): # for each ticker
        myData = web.DataReader(tickers[j], 'yahoo', start, end) # query the pandas datareader to pull data from Yahoo! finance
        fileName = tickers[j] + '.csv' # create a file
        myData.to_csv(fileName) # save data to the file