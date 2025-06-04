import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import invgauss, levy
import math


def analyze_event_temporal_distribution(events, num_bins=50, time_index=0, title_suffix=""):
    """
    Analyze the temporal distribution of events using histogram
    
    Parameters:
        events (ndarray): Event array [t, x, y, ...]
        num_bins (int): Number of histogram bins
        time_index (int): Index of timestamp in event array
        title_suffix (str): Additional text for plot title
    """
    # Check if input is PyTorch tensor, convert to numpy array if it is
    if hasattr(events, 'cpu'):
        events = events.cpu().numpy()
    
    # Extract timestamp data
    timestamps = events[:, time_index]
    if len(timestamps) < 2:
        print("Warning: Not enough events for analysis")
        return None
    
    # Calculate time range and convert to milliseconds
    t_min, t_max = np.min(timestamps), np.max(timestamps)
    timestamps_ms = (timestamps - t_min) * 1e-3  # Subtract minimum timestamp, unit in milliseconds
    
    # Create histogram data
    hist, bin_edges = np.histogram(timestamps_ms, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.7)
    plt.xlabel('Time (ms, from start)', fontsize=12)
    plt.ylabel('Event Count', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title(f'Event Temporal Distribution {title_suffix}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return {
        'histogram': {
            'bins': bin_centers,  # Bin centers in milliseconds
            'counts': hist,       # Event count per bin
            'edges': bin_edges * 1e-3  # Bin edges in milliseconds
        }
    }


def analyze_per_pixel_event_intervals_combined(
    events,                      # Input event data
    min_events_per_pixel=5,      # Minimum events per pixel
    max_dt_us_for_plot=100000,   # Maximum time interval for plotting
    plot_bins=100,               # Number of histogram bins
    mean_threshold=1e-6,         # Drift rate threshold
    type='RGB'                   # Data type
):
    """Analyze event time intervals per pixel and fit distribution"""
    # Convert to numpy array
    if hasattr(events, 'cpu'):
        events = events.cpu().numpy()
    
    # Extract coordinates and create pixel timestamp dictionary
    x_coords = events[:, 1].astype(int)
    y_coords = events[:, 2].astype(int)
    pixel_timestamps = defaultdict(list)
    for i in range(events.shape[0]):
        pixel_timestamps[(x_coords[i], y_coords[i])].append(events[i, 0])
    
    # Filter valid pixels and calculate time intervals
    eligible_pixels_timestamps = [
        np.sort(ts_list) for ts_list in pixel_timestamps.values() 
        if len(ts_list) >= min_events_per_pixel
    ]
    
    # Calculate all time intervals
    all_dt = np.concatenate([
        np.diff(ts_list) for ts_list in eligible_pixels_timestamps if len(ts_list) > 1
    ])
    all_dt = all_dt[all_dt > 0]
    
    # Prepare fitting data
    tau_for_fit = all_dt.astype(np.float64)
    num_pixels = len(eligible_pixels_timestamps)
    
    # Try fitting inverse Gaussian distribution
    try:
        ig_fit_shape, ig_fit_loc, ig_fit_scale = invgauss.fit(tau_for_fit, floc=0)
        ig_mu = ig_fit_shape * ig_fit_scale
        ig_lambda = ig_fit_scale
        mu = 1.0 / ig_mu if ig_mu > 0 else 0
        sigma = 1.0 / math.sqrt(ig_lambda) if ig_lambda > 0 else 0
        dist_name = "Inverse Gaussian"
    except:
        # If inverse Gaussian fit fails, try Lévy distribution
        try:
            levy_fit_loc, levy_fit_scale = levy.fit(tau_for_fit, floc=0)
            mu = 0.0
            sigma = 1.0 / math.sqrt(levy_fit_scale) if levy_fit_scale > 0 else 0
            final_scale_param = levy_fit_scale
            dist_name = "Lévy"
        except:
            return num_pixels, all_dt, np.nan, np.nan
    
    # Plot distribution
    dt_for_plot = all_dt[all_dt <= max_dt_us_for_plot]
    if len(dt_for_plot) > 0:
        plt.figure(figsize=(12, 6))
        plt.hist(dt_for_plot, bins=plot_bins, density=True, alpha=0.6, label="Histogram")
        
        # Generate fit curve
        x_plot = np.linspace(dt_for_plot.min(), dt_for_plot.max(), 500)
        if dist_name == "Inverse Gaussian":
            y_plot = invgauss.pdf(x_plot, ig_fit_shape, loc=0, scale=ig_fit_scale)
            plt.plot(x_plot, y_plot, 'r-', linewidth=2, 
                    label=f'IG Fit (ig_u={ig_mu:.5g}, ig_lambda={ig_lambda:.5g}, μ={mu:.5g}, σ={sigma:.5g})')
        else:
            y_plot = levy.pdf(x_plot, loc=0, scale=final_scale_param)
            plt.plot(x_plot, y_plot, 'r-', linewidth=2, 
                    label=f'Lévy Fit (scale={levy_fit_scale:.5g}, μ={mu:.5g}, σ={sigma:.5g})')
        
        # Set plot properties
        plt.xlabel(f"Time Interval Δt (μs)")
        plt.ylabel("Probability Density")
        plt.title(f"{type} Interval Distribution")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    return num_pixels, all_dt, mu, sigma


def analyze_event_frequency_spectrum(events, max_freq_hz=100, bins=50, time_index=0, min_events_per_pixel=2):
    """
    Analyze frequency spectrum of event data
    
    Parameters:
        events (ndarray): Event array [t, x, y, ...]
        max_freq_hz (float): Maximum analysis frequency (Hz)
        bins (int): Number of histogram bins
        time_index (int): Index of timestamp in event array
        min_events_per_pixel (int): Minimum events per pixel
    
    Returns:
        dict: Dictionary containing frequency analysis results
    """
    if hasattr(events, 'cpu'):  # Convert PyTorch tensor to numpy array
        events = events.cpu().numpy()
    
    # Extract coordinate information
    x_coords = events[:, 1].astype(int)  # Extract x coordinates
    y_coords = events[:, 2].astype(int)  # Extract y coordinates
    
    # Collect timestamps for each pixel
    pixel_timestamps = {}
    for i in range(events.shape[0]):
        key = (x_coords[i], y_coords[i])
        if key not in pixel_timestamps:
            pixel_timestamps[key] = []
        pixel_timestamps[key].append(events[i, time_index])
    
    # Calculate frequency for each pixel
    pixel_frequencies = {}
    for pixel, timestamps in pixel_timestamps.items():
        if len(timestamps) >= min_events_per_pixel:
            ts = np.sort(timestamps)  # Sort timestamps
            intervals = np.diff(ts)   # Calculate intervals
            # Calculate frequency: 1/interval (convert microseconds to seconds)
            freqs = 1.0 / (intervals * 1e-6)
            freqs = freqs[freqs <= max_freq_hz]  # Limit maximum frequency
            if len(freqs) > 0:
                pixel_frequencies[pixel] = freqs
    
    # Combine frequency data from all pixels
    if not pixel_frequencies:
        print("Warning: No valid pixel frequency data found")
        return None
    
    all_frequencies = np.concatenate(list(pixel_frequencies.values()))
    
    # Calculate basic statistics
    freq_stats = {
        'mean': np.mean(all_frequencies),
        'median': np.median(all_frequencies),
        'std': np.std(all_frequencies),
        'min': np.min(all_frequencies),
        'max': np.max(all_frequencies),
        'count': len(all_frequencies),
        'pixels': len(pixel_frequencies)
    }
    
    # Calculate frequency distribution histogram
    hist, bin_edges = np.histogram(all_frequencies, bins=bins, range=(0, max_freq_hz))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.bar(bin_centers, hist, width=(max_freq_hz/bins)*0.8, alpha=0.7)
    plt.axvline(30, color='r', linestyle='--', label=f"Mean: {freq_stats['mean']:.2f} Hz")
    plt.axvline(freq_stats['median'], color='g', linestyle='--', label=f"Median: {freq_stats['median']:.2f} Hz")
    
    # Add statistics text
    info_text = (f"Pixel Count: {freq_stats['pixels']}\n"
                f"Event Frequency: {freq_stats['count']}\n"
                f"Mean: {freq_stats['mean']:.2f} Hz\n"
                f"Standard Deviation: {freq_stats['std']:.2f} Hz\n"
                f"Range: [{freq_stats['min']:.2f}, {freq_stats['max']:.2f}] Hz")
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set plot properties
    plt.xlabel('Frequency (Hz)(log scale)')
    plt.ylabel('Count')
    plt.title('Event Frequency Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return {
        'frequencies': all_frequencies,
        'histogram': {'bins': bin_centers, 'counts': hist},
        'stats': freq_stats
    }

def analyze_event_fft_spectrum(events, sampling_rate=1000, time_index=0, max_freq_hz=100):
    """Analyze FFT spectrum of event data"""
    if hasattr(events, 'cpu'):  # Convert PyTorch tensor to numpy array
        events = events.cpu().numpy()
    
    # Extract timestamps
    timestamps = events[:, time_index]
    if len(timestamps) < 10:  # Not enough events for analysis
        print("Warning: Insufficient events for FFT analysis")
        return None
    
    # Calculate event rate signal
    t_min, t_max = np.min(timestamps), np.max(timestamps)
    duration = (t_max - t_min) * 1e-6  # Convert to seconds
    
    # Create uniform time axis
    num_bins = int(sampling_rate * duration)  # Determine number of bins based on sampling rate
    num_bins = max(1024, num_bins)  # Ensure at least 1024 points for sufficient resolution
    
    # Calculate event rate time series
    hist, bin_edges = np.histogram(timestamps, bins=num_bins, range=(t_min, t_max))
    bin_width = (t_max - t_min) / num_bins  # μs
    event_rate = hist / (bin_width * 1e-6)  # Convert to events/second
    
    # Calculate FFT
    fft_result = np.fft.rfft(event_rate)  # Real FFT
    fft_magnitude = np.abs(fft_result) * 2.0 / num_bins  # Normalize magnitude
    fft_magnitude[0] /= 2  # DC component doesn't need ×2
    
    # Calculate frequency axis
    sample_spacing = duration / num_bins  # seconds
    fft_freqs = np.fft.rfftfreq(num_bins, d=sample_spacing)  # Frequency axis (Hz)
    
    # Modify frequency limit section
    valid_idx = (fft_freqs > 0) & (fft_freqs <= max_freq_hz)  # Use & for logical AND
    fft_freqs = fft_freqs[valid_idx]
    fft_magnitude = fft_magnitude[valid_idx]
    
    # Modify dominant frequency calculation section
    dominant_freq = fft_freqs[np.argmax(fft_magnitude)] if len(fft_magnitude) > 0 else 0
    
    # Calculate statistics
    fft_stats = {
        'dominant_freq': dominant_freq,
        'mean_magnitude': np.mean(fft_magnitude),
        'max_magnitude': np.max(fft_magnitude),
        'total_power': np.sum(fft_magnitude**2)
    }
    
    # Plot FFT
    plt.figure(figsize=(12, 6))
    plt.plot(fft_freqs, fft_magnitude, linewidth=1.5)
    # plt.axvline(dominant_freq, color='r', linestyle='--', 
    #             label=f'Main Frequency: {dominant_freq:.2f} Hz')
    
    # # Add information text
    # info_text = (f"main frequency: {dominant_freq:.2f} Hz\n"
    #              f"mean magnitude: {fft_stats['mean_magnitude']:.2f}\n"
    #              f"max magnitude: {fft_stats['max_magnitude']:.2f}\n"
    #              f"total power: {fft_stats['total_power']:.2f}")
    # plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, 
    #          verticalalignment='top', horizontalalignment='right',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set plot properties
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title('Event FFT Spectrum Analysis')
    plt.grid(True, alpha=0.3)
    # plt.legend()
    plt.tight_layout()
    
    return {
        'freqs': fft_freqs,
        'magnitude': fft_magnitude,
        'stats': fft_stats,
        'event_rate': event_rate,
        'time_bins': bin_edges
    }

