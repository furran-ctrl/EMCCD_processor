from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import csv

from src.core.tiff_objects import EMCCDimage
from src.io.tiff_import import TiffLoader
from src.core.reslut_class import ProcessedResult, XPSGroupResult

class XPSGroupProcessor:
    """
    Processor for handling all files within a single XPS group.
    
    This class loads background images, processes individual TIFF files by 
    removing background, finding diffraction centers, and calculating 
    azimuthal averages, then aggregates results for the entire XPS group.
    """
    
    def __init__(self, 
                 xps_value: float,
                 file_list: List[str],
                 bkg_directory: str,
                 bkg_filename: str,
                 initial_center_guess: Tuple[float, float] = (512, 512)):
        """
        Initialize XPS group processor.
        
        Args:
            xps_value: XPS value for this group
            file_list: List of file paths to process
            bkg_directory: Directory containing background image
            bkg_filename: Background image filename
            initial_center_guess: Initial guess for diffraction center (x, y)
        """
        self.xps_value = xps_value
        self.file_list = file_list
        self.bkg_directory = bkg_directory
        self.bkg_filename = bkg_filename
        self.initial_center_guess = initial_center_guess
        self.results: Optional[XPSGroupResult] = None
        self.bkg_image: Optional[EMCCDimage] = None
    
    def load_background(self) -> None:
        """
        Load and prepare background image.
        
        Raises:
            FileNotFoundError: If background file cannot be found
            ValueError: If background image cannot be loaded
        """
        try:
            print(f"Loading background image: {self.bkg_filename}")
            bkg_image = EMCCDimage(TiffLoader(self.bkg_directory, self.bkg_filename))
            bkg_image.copy_as_processed()
            self.bkg_image = bkg_image
            print("Background loaded successfully")
        except Exception as e:
            print(f"Error loading background: {e}")
            raise
    
    def process_single_file(self, filepath: str) -> Optional[ProcessedResult]:
        """
        Process a single TIFF file.
        
        Args:
            filepath: Path to the TIFF file to process
            
        Returns:
            ProcessedResult containing center and radial profile, or None if processing fails
            
        Steps:
            1. Load image file
            2. Remove background using pre-loaded background image
            3. Find diffraction center using iterative ring centroid
            4. Calculate azimuthal average radial profile
        """
        try:
            # Load image file
            image_file = EMCCDimage(TiffLoader(Path(filepath).parent, Path(filepath).name))
            
            # Remove background
            image_file.remove_background(self.bkg_image.get_processed_data())
            
            # Find diffraction center
            center = image_file.iterative_ring_centroid(self.initial_center_guess)
            
            # Calculate azimuthal average
            bin_centers, radial_average = image_file.azimuthal_average()
            
            return ProcessedResult(
                center=center,
                radial_profile=radial_average,
            )
            
        except Exception as e:
            print(f"Error processing {Path(filepath).name}: {e}")
            return None
    
    def process_xps_group(self, verbose: bool = False) -> XPSGroupResult:
        """
        Process all files in the XPS group.
        
        Args:
            verbose: If True, display progress updates during processing
            
        Returns:
            XPSGroupResult containing aggregated results for the entire group
            
        Raises:
            ValueError: If background is not loaded or no files processed successfully
        """
        # Ensure background is loaded
        if self.bkg_image is None:
            self.load_background()
        
        print(f"Processing XPS group {self.xps_value:.5f} with {len(self.file_list)} files...")
        
        # Process all files
        all_radial_profiles = []
        centers = []
        total_counts = []
        successful_files = 0
        
        for i, filepath in enumerate(self.file_list):
            if verbose and i % 5 == 0:
                print(f"Progress: {i}/{len(self.file_list)} files processed")
            
            result = self.process_single_file(filepath)
            if result is not None:
                all_radial_profiles.append(result.radial_profile)
                centers.append(result.center)
                total_counts.append(np.sum(result.radial_profile))
                successful_files += 1
        
        if successful_files == 0:
            raise ValueError(f"No files processed successfully in XPS group {self.xps_value}")
        
        # Calculate average and standard deviation
        avg_radial_profile, std_radial_profile = self.calculate_group_statistics(all_radial_profiles)
        
        # Create result object
        self.results = XPSGroupResult(
            xps_value=self.xps_value,
            all_radial_profile=all_radial_profiles,
            avg_radial_profile=avg_radial_profile,
            std_radial_profile=std_radial_profile,
            file_count=successful_files,
            centers=centers,
            total_counts=total_counts
        )
        
        print(f"XPS group processing completed: {successful_files}/{len(self.file_list)} files successful")
        
        return self.results
    
    def calculate_group_statistics(self, radial_profiles: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate average and standard deviation of radial profiles.
        
        Args:
            radial_profiles: List of radial profile arrays from all successfully processed files
            
        Returns:
            Tuple of (average_radial_profile, std_radial_profile)
            
        Raises:
            ValueError: If radial_profiles list is empty
        """
        if not radial_profiles:
            raise ValueError("No radial profiles provided for statistics calculation")
        
        # Convert to numpy array for vectorized operations
        profiles_array = np.array(radial_profiles)
        
        # Calculate average and standard deviation
        avg_profile = np.mean(profiles_array, axis=0)
        std_profile = np.std(profiles_array, axis=0)
        
        return avg_profile, std_profile
    
    def save_to_csv(self, output_dir: str, filename: str = None) -> None:
        """
        Save XPS group results to CSV file.
        
        Args:
            output_dir: Directory to save the CSV file
            filename: Optional filename, defaults to f"xps_{xps_value:.5f}_results.csv"
            
        Raises:
            ValueError: If results are not available (process_xps_group not called)
            IOError: If file cannot be written
        """
        if self.results is None:
            raise ValueError("No results available. Call process_xps_group() first.")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            filename = f"xps_{self.xps_value:.5f}_results.csv"
        
        csv_filepath = output_path / filename
        
        try:
            with open(csv_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['XPS_Group_Results'])
                writer.writerow(['XPS_Value', f'{self.xps_value:.6f}'])
                writer.writerow(['Files_Processed', f'{self.results.file_count}'])
                writer.writerow(['Total_Files', f'{len(self.file_list)}'])
                writer.writerow([])  # Empty row for separation
                
                # Write centers
                writer.writerow(['Diffraction_Centers'])
                writer.writerow(['File_Index', 'Center_X', 'Center_Y'])
                for i, (center_x, center_y) in enumerate(self.results.centers):
                    writer.writerow([i, f'{center_x:.3f}', f'{center_y:.3f}'])
                writer.writerow([])
                
                # Write total counts
                writer.writerow(['Total_Counts'])
                writer.writerow(['File_Index', 'Total_Count'])
                for i, count in enumerate(self.results.total_counts):
                    writer.writerow([i, f'{count:.3f}'])
                writer.writerow([])
                
                # Write radial profiles statistics
                writer.writerow(['Radial_Profile_Statistics'])
                writer.writerow(['Bin_Index', 'Average_Intensity', 'Standard_Deviation'])
                for i, (avg, std) in enumerate(zip(self.results.avg_radial_profile, 
                                                 self.results.std_radial_profile)):
                    writer.writerow([i, f'{avg:.6f}', f'{std:.6f}'])
            
            print(f"Results saved to: {csv_filepath}")
            
        except Exception as e:
            raise IOError(f"Error writing CSV file {csv_filepath}: {e}")
    
    def get_processing_summary(self) -> dict:
        """
        Get summary statistics of the processing results.
        
        Returns:
            Dictionary containing processing summary statistics
        """
        if self.results is None:
            return {"status": "Not processed"}
        
        centers_array = np.array(self.results.centers)
        total_counts_array = np.array(self.results.total_counts)
        
        return {
            "xps_value": self.xps_value,
            "files_processed": self.results.file_count,
            "total_files": len(self.file_list),
            "success_rate": self.results.file_count / len(self.file_list),
            "center_mean": np.mean(centers_array, axis=0).tolist(),
            "center_std": np.std(centers_array, axis=0).tolist(),
            "total_counts_mean": np.mean(total_counts_array),
            "total_counts_std": np.std(total_counts_array),
            "avg_profile_max": np.max(self.results.avg_radial_profile),
            "avg_profile_min": np.min(self.results.avg_radial_profile)
        }
