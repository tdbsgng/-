import pandas as pd
from rouge import Rouge

def rouge_evaluate(row):
    rouge = Rouge()
    score = rouge.get_scores(' '.join(list(row["auto_summary"])), ' '.join(list(row["summarization"])))
    return score[0]["rouge-l"]["f"]

def main():
    df_test = pd.read_csv("data/final_analyze.csv", usecols=[ 'summarization', 'article', 'auto_summary'])
    df_test["score"] = df_test.apply(rouge_evaluate, axis = 1)

    total_scores = 0
    for score in df_test["score"]:
        total_scores += score
    print(df_test)
    final_score = total_scores / len(df_test["score"])
    print(final_score)

if __name__ == '__main__':
    main() 
