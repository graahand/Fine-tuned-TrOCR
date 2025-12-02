import pandas as pd
import os

def count_metrics(file_path, limit):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Slice the dataframe up to the requested row limit
        # If limit is greater than total rows, it just takes all rows
        df_subset = df.iloc[:limit]
        
        print(f"\nAnalysis for the first {len(df_subset)} rows:")
        print("-" * 85)
        print(f"{'Column Name':<30} | {'True':<8} | {'False':<8} | {'Blanks':<8} | {'Accuracy %':<10}")
        print("-" * 85)
        
        # Accumulators for overall statistics
        grand_total_true = 0
        grand_total_false = 0

        for col in df_subset.columns:
            # Skip unnamed columns that might appear due to trailing commas
            if "Unnamed" in col:
                continue

            series = df_subset[col]
            
            # Normalize data to string, strip whitespace, and convert to uppercase
            # This handles boolean types, mixed types, and string variations
            normalized = series.astype(str).str.strip().str.upper()
            
            # Count True
            num_true = (normalized == 'TRUE').sum()
            
            # Count False
            num_false = (normalized == 'FALSE').sum()
            
            # Count Blanks
            # In pandas, empty CSV fields are read as NaN (Not a Number)
            # We count actual NaNs and empty strings
            num_blanks = series.isna().sum()
            
            # If data was read as object (strings), check for empty strings explicitly
            if series.dtype == 'object':
                num_blanks += (series == '').sum()

            # Calculate Accuracy (neglecting blanks)
            total_valid = num_true + num_false
            if total_valid > 0:
                accuracy = (num_true / total_valid) * 100
            else:
                accuracy = 0.0

            # Update grand totals
            grand_total_true += num_true
            grand_total_false += num_false

            print(f"{col:<30} | {num_true:<8} | {num_false:<8} | {num_blanks:<8} | {accuracy:<10.2f}")
        
        print("-" * 85)
        
        # Calculate Overall Accuracy (Whole Subset)
        grand_total_valid = grand_total_true + grand_total_false
        if grand_total_valid > 0:
            overall_accuracy = (grand_total_true / grand_total_valid) * 100
        else:
            overall_accuracy = 0.0
            
        print(f"\nOverall Statistics:")
        print(f"Total True  : {grand_total_true}")
        print(f"Total False : {grand_total_false}")
        print(f"Total Valid : {grand_total_valid}")
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define the path to your specific file
    # We check the absolute path first, then relative to current dir
    target_file = '/home/museum/Downloads/Accuracy_vlm(minicpm_scanned).csv'
    
    if not os.path.exists(target_file):
        # Fallback to checking current directory if file was moved
        target_file = '/home/museum/Fine-tuned-TrOCR/Finalized_flow/Accuracy_vlm(Qwen) (2).csv'
    print(f"Target File: {target_file}")
    
    try:
        user_input = input("Enter the row number up to which the count is required: ")
        row_limit = int(user_input)
        
        if row_limit < 0:
            print("Please enter a positive number.")
        else:
            count_metrics(target_file, row_limit)
            
    except ValueError:
        print("Invalid input. Please enter an integer number.")