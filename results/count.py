import pandas as pd

# Load the CSV file
df = pd.read_csv('final_normalized_with_jailbroken_gpt.csv')  # Replace with your actual CSV filename
print("Column names:", df.columns.tolist())


# Define the categories and the normalized columns
categories = ['Species', 'SocialValue', 'Gender', 'Age', 'Fitness', 'Utilitarianism']
normalized_columns = ['gpt_normalized', 'qwen_normalized','gpt_j_normalized']
values_to_count = [1, 0, -1]

# Initialize result dictionary
result = {category: {col: {val: 0 for val in values_to_count} for col in normalized_columns} for category in categories}

# Count occurrences
for category in categories:
    subset = df[df['phenomenon_category'] == category]
    for col in normalized_columns:
        counts = subset[col].value_counts()
        for val in values_to_count:
            result[category][col][val] = counts.get(val, 0)

# Display the result
for category in categories:
    print(f"\nCategory: {category}")
    for col in normalized_columns:
        print(f"  {col}: ", end="")
        print(", ".join(f"{val}: {result[category][col][val]}" for val in values_to_count))
