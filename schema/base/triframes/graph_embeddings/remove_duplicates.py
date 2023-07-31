import pandas as pd

filename_input = "full.csv"
filename_output = "full_clean.csv"

#read data
df = pd.read_csv(filename_input, delimiter="\t", header=None)
df.columns = ['verb', 'subject', 'object', 'score']

# sum up scores
df['score'] = df.groupby(['verb', 'subject', 'object'])['score'].transform('sum')

# drop duplicates
df = df.drop_duplicates(subset=['verb', 'subject', 'object', 'score'])
df = df.sort_values(['score'], ascending=False)

df.to_csv(filename_output, "\t", header=False, index=False)