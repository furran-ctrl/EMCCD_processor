import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_truncated_sem_with_variable1_n(max_k, n_original=1000):
    """
    Plots the SEM of a truncated normal distribution as a function of k-sigma,
    correctly accounting for the reduction in sample size (n) due to truncation.
    
    Args:
        max_k (float): The maximum number of standard deviations (k) to plot.
        n_original (int): The starting, original sample size before truncation.
    """
    k_values = np.linspace(0.1, max_k, 100)
    sem_values = []
    
    # Pre-calculate the theoretical SEM for the full, untruncated sample
    sem_untruncated = 1 / np.sqrt(n_original) 

    for k in k_values:
        # --- Step 1: Calculate the Standard Deviation of the Truncated Distribution (sigma_k) ---
        # Probability density function (PDF) and Cumulative distribution function (CDF) at k
        pdf_k = norm.pdf(k)
        cdf_k = norm.cdf(k)
        
        # Denominator of the variance formula is the normalization factor (P_k)
        # P_k = cdf(k) - cdf(-k)
        P_k = cdf_k - norm.cdf(-k)
        
        # Variance of truncated distribution (sigma_k^2):
        # var_k = 1 - (2 * k * pdf(k)) / P_k
        truncated_var = 1 - (2 * k * pdf_k / P_k)
        truncated_std = np.sqrt(truncated_var)
        
        # --- Step 2: Calculate the Reduced Sample Size (n_k) ---
        # n_k = N_original * P_k
        n_k = n_original * P_k
        
        # --- Step 3: Calculate the SEM for the Truncated Data ---
        # SEM_k = sigma_k / sqrt(n_k)
        sem_k = truncated_std / np.sqrt(n_k)
        sem_values.append(sem_k)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sem_values, label=f'SEM ($\sigma_k / \sqrt{{n_k}}$)', color='darkorange', lw=2)
    
    # Add a horizontal line for the theoretical SEM of the original sample
    plt.axhline(sem_untruncated, color='red', linestyle='--', label=f'Theoretical SEM (Untruncated $N={n_original}$)')
    
    # Add a line for the Standard Deviation of the Truncated Data (to show the components)
    plt.plot(k_values, np.array(sem_values) * np.sqrt(np.array(n_original * (norm.cdf(k_values) - norm.cdf(-k_values)))), 
             label='Standard Deviation ($\sigma_k$)', color='darkgreen', linestyle=':', alpha=0.7)

    plt.title(f'SEM of Truncated Normal Distribution Accounting for Reduced Sample Size')
    plt.xlabel('Truncation Point ($k\sigma$)')
    plt.ylabel('Standard Error of the Mean ($\text{SEM}_k$)')
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.legend()
    plt.show()

#plot_truncated_sem_with_variable_n(3)

def plot_truncated_sem_with_variable_n(max_k, n_original=1000):
    """
    Plots the SEM of a truncated normal distribution as a function of k-sigma,
    correctly accounting for the reduction in sample size (n) due to truncation.
    """
    # Start k very close to 0 (but not 0 to avoid division by zero)
    k_values = np.linspace(0.01, max_k, 200)
    sem_values = []
    
    # Pre-calculate the theoretical SEM for the full, untruncated sample
    sem_untruncated = 1 / np.sqrt(n_original) 

    for k in k_values:
        # --- 1. Truncated Standard Deviation (sigma_k) ---
        pdf_k = norm.pdf(k)
        
        # P_k = cdf(k) - cdf(-k)
        P_k = norm.cdf(k) - norm.cdf(-k)
        
        # Variance of truncated distribution (sigma_k^2)
        truncated_var = 1 - (2 * k * pdf_k / P_k)
        truncated_std = np.sqrt(truncated_var)
        
        # --- 2. Reduced Sample Size (n_k) ---
        n_k = n_original * P_k
        
        # --- 3. SEM for the Truncated Data (SEM_k) ---
        sem_k = truncated_std / np.sqrt(n_k)
        sem_values.append(sem_k)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot the calculated SEM
    plt.plot(k_values, sem_values, 
             label=f'Calculated SEM ($\sigma_k / \sqrt{{n_k}}$)', 
             color='darkorange', lw=3)
    
    # Plot the theoretical sqrt(k) approximation for small k
    # We fit a constant C to the calculated SEM at a small k for the visual guide
    k_small = k_values[0] # Use the smallest k for the scaling constant
    C = sem_values[0] / np.sqrt(k_small)
    
    plt.plot(k_values, C * np.sqrt(k_values), 
             label=f'Theoretical Approximation ($\propto \sqrt{{k}}$)', 
             color='black', linestyle='--', alpha=0.7)

    # Add a horizontal line for the theoretical SEM of the original sample
    plt.axhline(sem_untruncated, color='red', linestyle=':', 
                label=f'Theoretical SEM (Untruncated $N={n_original}$)')

    plt.title(f'SEM of Truncated Normal Distribution: $\sqrt{{k}}$ Scaling near Zero')
    plt.xlabel('Truncation Point ($k\sigma$)')
    plt.ylabel('Standard Error of the Mean ($\text{SEM}_k$)')
    plt.ylim(0, max(sem_values) * 1.05)
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.legend()
    plt.savefig('truncated_sem_plot.png')
    plt.show()

# Execute the function with a large N to better approximate the theoretical values
plot_truncated_sem_with_variable_n(max_k=4, n_original=1000)