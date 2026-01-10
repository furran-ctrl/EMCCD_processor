import pandas as pd
import sys
import os

def convert_parquet_to_csv(input_path: str, output_path: str, index: bool = False):
    """
    Converts a Parquet file to a CSV file using the pandas library.

    Args:
        input_path (str): The full path to the input .parquet file.
        output_path (str): The full path for the output .csv file.
        index (bool): Whether to write the DataFrame index as a column in the CSV.
    """
    print(f"Starting conversion of: {input_path}")
    
    try:
        # Read the Parquet file into a pandas DataFrame
        df = pd.read_parquet(input_path)
        
        # Write the DataFrame to a CSV file
        df.to_csv(output_path, index=index)
        
        print(f"Successfully converted and saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}", file=sys.stderr)

# --- Example Usage ---
# To test this code, ensure you have a 'data.parquet' file in the same directory.
if __name__ == '__main__':
    
    # Define placeholder paths
    input_file = r'C:\Users\ab177\Desktop\diffraction_results\1012long\analysis_parallel_time\xps_198.00499\xps_198.00499.parquet'
    output_file = r'C:\Users\ab177\Desktop\analyze.csv'

    convert_parquet_to_csv(input_file, output_file, index=False)