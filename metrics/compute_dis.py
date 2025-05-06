# import pandas as pd
# import numpy as np

# m1_name = "qwen"
# m2_name = "gpt_j"

# def compute_dis_from_csv(csv_file_path):
#     # Load the CSV
#     df = pd.read_csv(csv_file_path)

#     # Define the order of categories
#     category_order = ["Species", "Gender", "Fitness", "SocialValue", "Age", "Utilitarianism"]

#     # Initialize vectors
#     p_m1 = []
#     p_m2 = []

#     # Populate the vectors in the specified order
#     for category in category_order:
#         row = df[df['phenomenon_category'] == category]
#         if not row.empty:
#             p_m1.append(float(row[m1_name].values[0]))
#             p_m2.append(float(row[m2_name].values[0]))
#         else:
#             raise ValueError(f"Category '{category}' is missing from the CSV file.")

#     # Convert to numpy arrays
#     p_m1 = np.array(p_m1)
#     p_m2 = np.array(p_m2)

#     # Compute L2 distance
#     dis = np.linalg.norm(p_m1 - p_m2)
#     return dis

# # Example usage
# if __name__ == "__main__":
#     file_path = "final_normalized_with_jailbroken_gpt.csv"  # Replace with the actual CSV filename
#     dis_score = compute_dis_from_csv(file_path)
#     print(f"DIS score between {m1_name} and {m2_name}: {dis_score}")

import numpy as np

# Define the vectors
gpt = np.array([-5, 4, -12, 0, 24, 12])
gpt_j = np.array([-8, 0, 26, -6, -12, -8])
qwen = np.array([-9, 9, -5, 2, -1, 4])

# Compute pairwise L2 distances
dist_gpt_gptj = np.linalg.norm(gpt - gpt_j)
dist_gpt_qwen = np.linalg.norm(gpt - qwen)
dist_gptj_qwen = np.linalg.norm(gpt_j - qwen)

# Print results
print(f"L2 distance between GPT and GPT (J): {dist_gpt_gptj:.4f}")
print(f"L2 distance between GPT and Qwen: {dist_gpt_qwen:.4f}")
print(f"L2 distance between GPT (J) and Qwen: {dist_gptj_qwen:.4f}")