import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import List, Tuple, Dict, Optional
import logging

from src.core.tiff_objects import EMCCDimage
from src.io.tiff_import import TiffLoader
from src.utils.timer import timer
from src.io.quick_plot import plot_azimuthal_average, plot_ndarray

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
            X_ray_config: [sigma_threshold, expansion_threshold_ratio]
            center_config: [ring_mask, initial_guess]
            azimuthal_config: [radial_masks, azimuthal_mask_dict]
            xps_value: XPS value for this group
            filelist: List of file paths to process
            resultdir: Directory to save results
        """
        self.background_data = background_data
        self.X_ray_config = X_ray_config  # [sigma_threshold, expansion_threshold_ratio]
        self.center_config = center_config  # [ring_mask, initial_guess]
        self.azimuthal_config = azimuthal_config  # [radial_masks, azimuthal_mask_dict]
        self.xps_value = xps_value
        self.filelist = filelist
        self.resultdir = Path(resultdir)
        
        #Create list for storing median and mad ndarray to sort Xray spots
        self.X_ray_precompute = None
        
        # Create result directory if it doesn't exist
        self.resultdir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = []
        self.processed_files = 0
        self.failed_files = []
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def precompute_xray_statistics(self, center_region_size: int = 40, sample_size: int = 50) -> None:
        """
        Precompute median and MAD arrays for X-ray removal using random sampling.
        
        Args:
            center_region_size: Size of center region to ignore (square around center)
            sample_size: Number of random images to use for statistics
        """
        if not self.filelist:
            raise ValueError("File list is empty")
        
        # Determine how many files to sample
        if len(self.filelist) <= sample_size:
            sample_files = self.filelist
            self.logger.info(f"Using all {len(sample_files)} files for X-ray statistics precomputation")
        else:
            # Randomly sample files
            import random
            sample_files = random.sample(self.filelist, sample_size)
            self.logger.info(f"Randomly sampled {len(sample_files)} files for X-ray statistics precomputation")
        
        all_processed_images = []
        
        # Load and process sample images
        for i, filepath in enumerate(sample_files):
            try:
                #self.logger.info(f"Processing sample {i+1}/{len(sample_files)}: {Path(filepath).name}")
                
                # Load image
                image = EMCCDimage(TiffLoader(Path(filepath).parent, Path(filepath).name))
                
                # Remove only background, no Xray filter 
                image.remove_background_legacy(
                    self.background_data
                )
                
                # Store the processed data
                all_processed_images.append(image.processed_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to process sample {filepath}: {str(e)}")
                continue
        
        if not all_processed_images:
            raise ValueError("No images were successfully processed for X-ray statistics")
        
        # Stack images and compute statistics
        image_stack = np.stack(all_processed_images, axis=0)
        
        # Compute median across the sample
        median_array = np.median(image_stack, axis=0)
        
        # Compute MAD (Median Absolute Deviation)
        abs_deviation = np.abs(image_stack - median_array)
        mad_array = np.median(abs_deviation, axis=0)
        
        # Get center position from center_config
        center_x, center_y = self.center_config[1]  # initial_guess tuple
        
        # Set center region to NaN to ignore bright diffraction center
        height, width = median_array.shape
        half_size = center_region_size // 2
        
        row_start = int(round(center_y - half_size))
        row_end = int(round(center_y + half_size))
        col_start = int(round(center_x - half_size))
        col_end = int(round(center_x + half_size))
        
        # Ensure indices are within bounds
        row_start = max(0, row_start)
        row_end = min(height, row_end)
        col_start = max(0, col_start)
        col_end = min(width, col_end)
        
        # Set center region to NaN
        median_array[row_start:row_end, col_start:col_end] = np.nan
        mad_array[row_start:row_end, col_start:col_end] = np.nan
        
        # Store precomputed statistics
        self.X_ray_precompute = [median_array, mad_array]
        
        self.logger.info(f"X-ray statistics precomputation completed. Center region ({center_region_size}x{center_region_size}) excluded.")

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
            #with timer('bkg_removal'):
            image_file.remove_background(
                self.background_data,
                self.X_ray_precompute[0],  # median_array
                self.X_ray_precompute[1],  # mad_array
                self.X_ray_config[0],  # sigma_threshold
                self.X_ray_config[1]   # expansion_threshold_ratio
            )

            # Find diffraction center
            #with timer('center_finding'):
            center = image_file.iterative_ring_centroid(
                self.center_config[0],  # ring_mask
                self.center_config[1]   # initial_guess
            )

            # Calculate azimuthal average
            #with timer('azimuthal_avg'):
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

    def process_single_debug(self, filepath: str, plot_min: float = None, plot_max: float = None) -> None:
        """
        Debug version: Process a single image file with extensive plotting and printing.
        
        Args:
            filepath: Path to the image file
            plot_min: Minimum value for plot display (optional)
            plot_max: Maximum value for plot display (optional)
        """
        try:
            print(f"\n{'='*60}")
            print(f"DEBUG PROCESSING: {Path(filepath).name}")
            print(f"{'='*60}")
            
            # Load image file
            image_file = EMCCDimage(TiffLoader(Path(filepath).parent, Path(filepath).name))
            
            # Plot original data before background removal
            print(f"\n1. ORIGINAL DATA:")
            print(f"Data range: [{np.nanmin(image_file.processed_data):.3f}, {np.nanmax(image_file.processed_data):.3f}]")
            plot_ndarray(image_file.processed_data, plot_min, plot_max)

            # Remove background
            print(f"\n2. BACKGROUND REMOVAL:")
            image_file.remove_background(
                self.background_data,
                self.X_ray_precompute[0],  # median_array
                self.X_ray_precompute[1],  # mad_array
                self.X_ray_config[0],  # sigma_threshold
                self.X_ray_config[1]   # expansion_threshold_ratio
            )
            
            # Plot after background removal
            print(f"After background removal:")
            print(f"Data range: [{np.nanmin(image_file.processed_data):.3f}, {np.nanmax(image_file.processed_data):.3f}]")
            print(f"NaN count: {np.sum(np.isnan(image_file.processed_data))}")
            plot_ndarray(image_file.processed_data, plot_min, plot_max)

            # Find diffraction center
            print(f"\n3. CENTER FINDING:")
            center = image_file.iterative_ring_centroid(
                self.center_config[0],  # ring_mask
                self.center_config[1]   # initial_guess
            )
            print(f"Found center: ({center[0]:.2f}, {center[1]:.2f})")
            print(f"Total count: {image_file.total_count:.2f}")

            # Calculate azimuthal average
            print(f"\n4. AZIMUTHAL AVERAGE:")
            bin_centers, radial_average = image_file.azimuthal_average_bincount(
                self.azimuthal_config[0],  # radial_masks
                self.azimuthal_config[1]   # azimuthal_mask dict
            )
            
            # Plot radial average
            print(f"Radial profile range: [{np.nanmin(radial_average):.3f}, {np.nanmax(radial_average):.3f}]")
            print(f"Non-NaN bins: {np.sum(~np.isnan(radial_average))}/{len(radial_average)}")
            plot_azimuthal_average(bin_centers, radial_average)
            
            print(f"\n✓ SUCCESSFULLY PROCESSED: {Path(filepath).name}")
            print(f"{'='*60}")

        except Exception as e:
            print(f"\n✗ FAILED TO PROCESS {filepath}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.failed_files.append((filepath, str(e)))

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
        
        # Precompute X-ray statistics once for the entire group
        self.precompute_xray_statistics()

        for i, filepath in enumerate(self.filelist, 1):
            # Process single file
            #with timer('single'):
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