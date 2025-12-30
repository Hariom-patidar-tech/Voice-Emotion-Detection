import pandas as pd

df = pd.read_csv("data/emotion_dataset.csv")

print(df.head())
print("\nEmotion count:")
print(df['emotion'].value_counts())
