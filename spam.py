def readCollection():
  import pandas as pd
  df = pd.read_csv('data/SMSSpamCollection',delimiter='\t',header = None)
  print df.head()


readCollection()