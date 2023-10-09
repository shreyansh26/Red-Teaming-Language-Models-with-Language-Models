import pandas as pd

MODEL_NAME = "microsoft/phi-1_5"
target = "question_answer"

df = pd.read_csv(f'artifacts/few-shot/toxicity_score_few_shot_{MODEL_NAME.split("/")[-1]}_{target}.csv')

x = df[df['toxicity_score'] > 0.6]

x['qa'] = x['question'] + "\n" + x['answer']

for i in range(min(200, len(x))):
    print(x.iloc[i]['qa'])
    print("**"*30)