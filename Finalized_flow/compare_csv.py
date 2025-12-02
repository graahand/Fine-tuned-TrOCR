import pandas as pd
import numpy as np

def create_diff_csv(file1_path, file2_path, output_path):
    """
    Compares two CSV files based on the 'image_file_path' column and generates 
    a new CSV report highlighting the differences.
    """
    key_col = 'image_file_path'

    try:
        # Read CSVs, filling empty cells with an empty string for consistent comparison
        df1 = pd.read_csv(file1_path).fillna('')
        df2 = pd.read_csv(file2_path).fillna('')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
        return

    # Check if key column exists
    if key_col not in df1.columns or key_col not in df2.columns:
        print(f"Error: Key column '{key_col}' not found in one or both files.")
        return

    # Perform an outer merge on the key column
    # This aligns rows based on the image filename regardless of their order in the files
    merged_df = pd.merge(
        df1,
        df2,
        on=key_col,
        how='outer',
        suffixes=('_f1', '_f2')
    )

    # Prepare the final DataFrame for the report
    diff_report = pd.DataFrame()
    diff_report[key_col] = merged_df[key_col]

    # Identify columns to compare (intersection of columns from both files, excluding key)
    common_cols = [c for c in df1.columns if c in df2.columns and c != key_col]

    # Iterate through each common column to compare values
    for col in common_cols:
        col_f1 = f'{col}_f1'
        col_f2 = f'{col}_f2'

        # Fill NaN values that result from the outer merge (indicating a missing row in one file)
        # We use a placeholder to distinguish missing rows from empty strings
        val_f1 = merged_df[col_f1].fillna('[MISSING ROW]')
        val_f2 = merged_df[col_f2].fillna('[MISSING ROW]')

        # Compare the columns and create the diff string where they don't match
        diff_report[col] = np.where(
            val_f1 == val_f2,
            val_f1,  # If same, use the value
            val_f1.astype(str) + ' | ' + val_f2.astype(str)
        )

    # Identify rows that have at least one difference
    # A difference exists if the formatted string contains our separator '|'
    # We exclude the key column from this check
    cols_to_check = [c for c in diff_report.columns if c != key_col]
    
    if not cols_to_check:
        print("No common columns found to compare.")
        return

    has_difference = diff_report[cols_to_check].apply(
        lambda x: x.astype(str).str.contains(' | ', regex=False)
    ).any(axis=1)

    # Filter the report to only include rows with at least one difference
    final_report = diff_report[has_difference].reset_index(drop=True)

    if final_report.empty:
        print("No differences found between the two files. No report generated.")
        return

    # Save the final report to a new CSV file
    try:
        final_report.to_csv(output_path, index=False)
        print(f"\nComparison complete. Difference report saved to: {output_path}")
        print(f"Found differences in {len(final_report)} row(s).")
    except Exception as e:
        print(f"Error saving the output file: {e}")


def main():
    """Main function to get user input and run the comparison."""
    print("--- CSV Difference Report Generator ---")
    file1 = input("Enter the path to the first CSV file: ").strip()
    file2 = input("Enter the path to the second CSV file: ").strip()
    output_file = input("Enter the path for the output CSV report (e.g., 'diff_report.csv'): ").strip()
    
    create_diff_csv(file1, file2, output_file)

if __name__ == "__main__":
    main()