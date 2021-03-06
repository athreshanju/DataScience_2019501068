import glob
import csv
import os
import pandas as pd
for files in glob.glob('G:\\DataScience_2019501068\\DADV\\*.csv'):
    f = open(files, 'r')
    name = os.path.basename(f.name).replace(".csv","")
    data = pd.read_csv(f)
    lis = []
    for each in range(0, len(data)-1):
        row2 = data.iloc[each]
        row1 = data.iloc[each+1]
        gl = (row2["Close"] + row1["Close"]) / row1["Close"]
        lis.append(gl)
    lis.insert(0,0)
    data.loc[:,'gainORloss'] = lis
    data.to_csv(r'G:\\DataScience_2019501068\\DADV\\'+name+'.csv')
    f.close()