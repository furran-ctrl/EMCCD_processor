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

# NOTE: You will need to replace these placeholder paths with your actual file paths.
# To test this code, ensure you have a 'data.parquet' file in the same directory.
if __name__ == '__main__':
    
    # Define placeholder paths
    input_file = r'C:\Users\86177\Desktop\0809\analysis_parallel_time\xps_189.84000\filtered_xps_189.84000.parquet'
    output_file = r'C:\Users\86177\Desktop\analyze.csv'
    
    # ----------------------------------------------------------------------------------
    # Dummy data creation for testing (Remove this block for real use)
    # This creates a dummy 'input_data.parquet' file if it doesn't exist
    if not os.path.exists(input_file):
        print(f"Creating dummy Parquet file for demonstration: {input_file}")
        dummy_df = pd.DataFrame({
            'A': [1, 2, 3, 4], 
            'B': ['x', 'y', 'z', 'w'],
            'C': [0.1, 0.2, 0.3, 0.4]
        })
        dummy_df.to_parquet(input_file)
    # ----------------------------------------------------------------------------------
    
    convert_parquet_to_csv(input_file, output_file, index=False)