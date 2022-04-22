import pandas as pd
from rouge import Rouge
import sys
sys.setrecursionlimit(1800)

rouge = Rouge()

def transfer_arti(row):
  return len(row['article'])

def transfer_summ(row):
  return len(row['summary'])

def transfer_rougeScore(row):
  summary = ' '.join(list(row["summary"]))
  article = ' '.join(list(row["article"]))
  global count
  count += 1
  if count % 10000 == 0:
      print(count)  
  score = rouge.get_scores(summary, article)
  
  return score

path = 'data/train.csv'
df = pd.read_csv(path, names = ['summary', 'article'] )
df['article_length'] = df.apply(transfer_arti, axis = 1)
df['summary_length'] = df.apply(transfer_summ, axis = 1)

filter1 = df['article_length'] < 1500
df = df[filter1]
filter2 = df['summary_length'] < 500
df = df[filter2]
df = df.drop(df[df['summary_length']>=df['article_length']].index)

#df['RougeScore'] = rouge.get_scores(' '.join(list(df["summary"])), ' '.join(list(df["article"])))[0]["rouge-l"]["f"]
print(df.info())

count = 0
df['RougeScore'] = df.apply(transfer_rougeScore, axis = 1)
df.to_csv("Score_train.csv")
print(df['RougeScore'].head(5))
