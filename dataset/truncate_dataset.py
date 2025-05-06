import os
import pandas as pd

def process_csv_file(input_path, output_dir):
    file_name = os.path.basename(input_path).replace('.csv', '')
    output_path = os.path.join(output_dir, f"{file_name}_truncated.csv")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Failed to read {input_path}: {e}")
        return

    keep_columns = ["Prompt", "two_choices", "phenomenon_category", "gpt_response"]
    df = df[[col for col in keep_columns if col in df.columns]]

    if "Prompt" in df.columns:
        cols = [col for col in df.columns if col != "Prompt"] + ["Prompt"]
        df = df[cols]

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def main():
    input_dir = "./data"
    output_dir = "./data_truncated"

    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            process_csv_file(input_path, output_dir)

if __name__ == "__main__":
    main()
