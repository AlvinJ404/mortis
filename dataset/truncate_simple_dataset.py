import os
import pandas as pd

def simplify_gpt_response(text):
    if pd.isna(text):
        return text
    return text.split('.', 1)[0] + '.' if '.' in text else text

def process_file(input_path, output_dir):
    file_name = os.path.basename(input_path).replace('.csv', '')
    output_path = os.path.join(output_dir, f"{file_name}_simple.csv")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return

    if "gpt_response" in df.columns:
        df["gpt_response"] = df["gpt_response"].apply(simplify_gpt_response)
    else:
        print(f"'gpt_response' column not found in {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def main():
    input_dir = "./data_truncated"
    output_dir = "./data_truncated_simple"

    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            process_file(input_path, output_dir)

if __name__ == "__main__":
    main()
