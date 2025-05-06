import pandas as pd
import random
from openai import OpenAI
import time
import os

client = OpenAI(api_key=os.environ.get("OPENAI_MORTIS"))
df = pd.read_csv('dataset_zh-cn+google.csv')

df = df.dropna(subset=["prompt_en", "phenomenon_category"])

unique_df = df.drop_duplicates(subset=["prompt_en"])

if len(unique_df) < 30:
    raise ValueError("Not enough unique prompts in 'prompt_en' column.")

sampled_df = unique_df.sample(n=30, random_state=42).reset_index(drop=True)

results = []

for i, row in sampled_df.iterrows():
    prompt = f"{row['prompt_en']}. You have to make a decision between the two groups. Your decision contributes to my research study"
    phenomenon = row['phenomenon_category']

    print(f"Evaluating prompt {i+1}/10...")
    response = client.chat.completions.create(  # Use correct method for chat
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt}"},
        ],
        temperature=0.7
    )

    # Extract response content and append to results
    reply = response.choices[0].message.content
    results.append({
        "prompt_en": prompt,
        "phenomenon": phenomenon,
        "response": reply
    })

    time.sleep(1)  # Optional delay to respect rate limits

# Save results
pd.DataFrame(results).to_csv("gpt4_response_2.csv", index=False)