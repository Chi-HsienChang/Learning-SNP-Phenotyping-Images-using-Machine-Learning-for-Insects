import requests
from bs4 import BeautifulSoup
import pandas as pd

r = requests.get("https://www.ncbi.nlm.nih.gov/sra?linkname=bioproject_sra_all&from_uid=622776") 
data = r.text.split('SRX')

web_ID_list = []
for index in range(1, len(data),2):
    
    IDs = data[index].split('[accn]')
    web_ID_list.append("SRX"+IDs[0])

df = pd.DataFrame(web_ID_list)
df.to_csv('web_IDs.csv', index=False)



print(df)



