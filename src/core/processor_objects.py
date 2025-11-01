import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import List, Tuple, Dict, Optional
import logging

from src.core.tiff_objects import EMCCDimage
from src.io.tiff_import import TiffLoader
from src.utils.timer import timer

class XPSGroupProcessor:
    def __init__(self, 
                 background_data: np.ndarray,
                 X_ray_config: List,
                 center_config: List,
                 azimuthal_config: List,
                 xps_value: float,
                 filelist: List[str],
                 resultdir: str):
        """
        Initialize XPS Group Processor for batch processing of images.
        
        Args:
            background_data: Background data for subtraction
            X_ray_config: [chunk_size, sigma_threshold, beam_threshold]
            center_config: [ring_mask, initial_guess]
            azimuthal_config: [radial_masks, azimuthal_mask_dict]
            xps_value: XPS value for this group
            filelist: List of file paths to process
            resultdir: Directory to save results
        """
        self.background_data = background_data
        self.X_ray_config = X_ray_config  # [chunk_size, sigma_threshold, beam_threshold]
        self.center_config = center_config  # [ring_mask, initial_guess]
        self.azimuthal_config = azimuthal_config  # [radial_masks, azimuthal_mask_dict]
        self.xps_value = xps_value
        self.filelist = filelist
        self.resultdir = Path(resultdir)
        
        # Create result directory if it doesn't exist
        self.resultdir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = []
        self.processed_files = 0
        self.failed_files = []
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def process_single(self, filepath: str) -> Optional['ProcessedResult']:
        """
        Process a single image file.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            ProcessedResult object if successful, None if failed
        """
        try:
            #self.logger.info(f"Processing: {Path(filepath).name}")
            
            # Load image file
            image_file = EMCCDimage(TiffLoader(Path(filepath).parent, Path(filepath).name))

            # Remove background
            with timer('bkg_removal'):
                image_file.remove_background(
                    self.background_data,
                    self.X_ray_config[0],  # chunk_size
                    self.X_ray_config[1],  # sigma_threshold
                    self.X_ray_config[2]   # beam_threshold
                )

            # Find diffraction center
            with timer('center_finding'):
                center = image_file.iterative_ring_centroid(
                    self.center_config[0],  # ring_mask
                    self.center_config[1]   # initial_guess
                )

            # Calculate azimuthal average
            with timer('azimuthal_avg'):
                bin_centers, radial_average = image_file.azimuthal_average_bincount(
                    self.azimuthal_config[0],  # radial_masks
                    self.azimuthal_config[1]   # azimuthal_mask dict
                )

            # Create result object
            result = ProcessedResult(
                filename=Path(filepath).name,
                center=center,
                total_count=image_file.total_count,
                radial_profile=radial_average,
                xps_value=self.xps_value
            )

            #self.logger.info(f"Successfully processed: {Path(filepath).name}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to process {filepath}: {str(e)}")
            self.failed_files.append((filepath, str(e)))
            return None

    def process_group(self, batch_size: int = 100) -> None:
        """
        Process all files in the filelist.
        
        Args:
            batch_size: Number of files to process before printing progress and saving
        """
        total_files = len(self.filelist)
        self.logger.info(f"Starting batch processing of {total_files} files")
        
        start_time = time.time()
        batch_results = []
        
        for i, filepath in enumerate(self.filelist, 1):
            # Process single file
            result = self.process_single(filepath)
            
            if result is not None:
                self.results.append(result)
                batch_results.append(result)
                self.processed_files += 1
            
            # Print progress and save every batch_size files
            if i % batch_size == 0 or i == total_files:
                batch_time = time.time() - start_time
                self.logger.info(f"Processed {i}/{total_files} files. "
                               f"Batch time: {batch_time:.2f} seconds. "
                               f"Success rate: {self.processed_files}/{i} "
                               f"({self.processed_files/i*100:.1f}%)")
                
                # Save current batch results
                if batch_results:
                    self.save_results(batch_results, batch_number=i//batch_size)
                    batch_results = []  # Clear batch results after saving
                
                # Reset timer for next batch
                start_time = time.time()
        
        # Final summary
        self.logger.info(f"Processing completed. "
                       f"Successfully processed: {self.processed_files}/{total_files} "
                       f"({self.processed_files/total_files*100:.1f}%)")
        
        if self.failed_files:
            self.logger.warning(f"Failed files: {len(self.failed_files)}")
            # Save failed files list
            self.save_failed_files()

    def save_results(self, results: List['ProcessedResult'], batch_number: int = None) -> None:
        """
        Save results to CSV file.
        
        Args:
            results: List of ProcessedResult objects
            batch_number: Batch number for filename
        """
        if not results:
            self.logger.warning("No results to save")
            return
        
        # Create DataFrame
        data = []
        for result in results:
            row = {
                'filename': result.filename,
                'center_x': result.center[0],
                'center_y': result.center[1],
                'total_count': result.total_count,
                'xps_value': result.xps_value
            }
            
            # Add radial profile columns
            for i, intensity in enumerate(result.radial_profile):
                row[f'radial_bin_{i}'] = intensity
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Generate filename
        if batch_number is not None:
            filename = self.resultdir / f"results_batch_{batch_number:03d}.csv"
        else:
            filename = self.resultdir / "results_final.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False)
        self.logger.info(f"Saved {len(results)} results to {filename}")

    def save_failed_files(self) -> None:
        """Save list of failed files with error messages."""
        if not self.failed_files:
            return
            
        failed_df = pd.DataFrame(self.failed_files, columns=['filename', 'error'])
        failed_filepath = self.resultdir / "failed_files.csv"
        failed_df.to_csv(failed_filepath, index=False)
        self.logger.info(f"Saved {len(self.failed_files)} failed files to {failed_filepath}")

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of processing."""
        if not self.results:
            return {}
        
        centers = np.array([result.center for result in self.results])
        total_counts = np.array([result.total_count for result in self.results])
        
        return {
            'total_files_processed': self.processed_files,
            'total_files_failed': len(self.failed_files),
            'mean_center_x': np.mean(centers[:, 0]),
            'mean_center_y': np.mean(centers[:, 1]),
            'std_center_x': np.std(centers[:, 0]),
            'std_center_y': np.std(centers[:, 1]),
            'mean_total_count': np.mean(total_counts),
            'std_total_count': np.std(total_counts),
        }


class ProcessedResult:
    """Container for processed image results."""
    def __init__(self, 
                 filename: str,
                 center: Tuple[float, float],
                 total_count: float,
                 radial_profile: np.ndarray,
                 xps_value: float):
        self.filename = filename
        self.center = center
        self.total_count = total_count
        self.radial_profile = radial_profile
        self.xps_value = xps_value

'''
define a class:XPSGroupProcessor for processsing large batch of image:
the class should have these self variables:
background_data : ndarray
X_ray_config: List[chunk_size:int ; sigma_threshold:float ; beam_threshold:float]
Center_config: List[ring_mask:RingMask, initial_guess:Tuple[float,float]]
Azimuthal_config: List[Radialmasks:RadialMasks, azimuthal_mask:Dict]
(RingMask and RadialMasks were customized class)
xps_value:float  filelist:List[str]  resultdir:str
for every file in filelist, define a function process_single as follows:
try:
            # Load image file
            image_file = EMCCDimage(TiffLoader(Path(filepath).parent, Path(filepath).name))

            # Remove background
            image_file.remove_background(self.background_data,self.X_ray_config[0],self.X_ray_config[1],self.X_ray_config[2])

            # Find diffraction center
            center = image_file.iterative_ring_centroid(self.center_config[0], self.center_config[1])

            # Calculate azimuthal average
            bin_centers, radial_average = image_file.azimuthal_average_bincount(self.azimuthal_config[0],
                                                                            self.azimuthal_config[1])

            return ProcessedResult(
                center=center,
                total_count=image_file.total_count,
                radial_profile=radial_average,
            )

define a function process_group to process all files in self.filelist using process_single
save the results in self, also have an option to use time module to print the time used for every 100 files processed
calls save_results to save the results(center, total_count and radial profile) in csv format for every 100 files 

define function save_results
'''