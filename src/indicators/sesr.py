"""
FLASHCAT - Flash Drought Analysis and Toolkit
SESR (Standardized Evaporative Stress Ratio) Module

Author: Kumar Puran Tripathy
Email: tripathypuranbdk@gmail.com
Date: Sept 04, 2025
Version: 1.0.0

This module implements the SESR index for flash drought detection with 
flexible timescale options (pentad, weekly, biweekly, etc.)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple, Dict, Union
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__author__ = "Kumar Puran Tripathy"

class SESR:
    """
    Standardized Evaporative Stress Ratio (SESR) Index Calculator
    
    A component of FLASHCAT (Flash Drought Analysis and Toolkit) for 
    detecting and analyzing flash drought events using evaporative stress.
    
    Attributes
    ----------
    timescale : int
        Time aggregation period in days (5=pentad, 7=weekly, 14=biweekly)
    data : pd.DataFrame
        Input dataframe with AET and PET values
    results : pd.DataFrame
        Processed results with SESR calculations
    events : pd.DataFrame
        Detected flash drought events
    
    Methods
    -------
    fit(data)
        Fit the SESR model to input data
    detect_events()
        Detect flash drought events
    get_intensity()
        Calculate flash drought intensity metrics
    """
    
    def __init__(self, 
                 timescale: int = 5,
                 sesr_threshold_percentile: float = 20.0,
                 delta_threshold_percentile: float = 40.0,
                 min_duration_periods: Optional[int] = None,
                 standardization_window: int = 30):
        """
        Initialize SESR calculator with configurable parameters.
        
        Parameters
        ----------
        timescale : int, default=5
            Time aggregation period in days:
            - 5 for pentad (5-day periods)
            - 7 for weekly
            - 14 for biweekly
            - 10 for dekad (10-day periods)
            - Any custom period
            
        sesr_threshold_percentile : float, default=20.0
            Percentile threshold for identifying drought conditions
            
        delta_threshold_percentile : float, default=40.0
            Percentile threshold for rate of change (ΔSESR)
            
        min_duration_periods : int, optional
            Minimum number of periods for flash drought (default: 6 periods)
            
        standardization_window : int, default=30
            Rolling window size in days for standardization
        """
        self.timescale = timescale
        self.sesr_threshold_percentile = sesr_threshold_percentile
        self.delta_threshold_percentile = delta_threshold_percentile
        self.min_duration_periods = min_duration_periods or (30 // timescale)
        self.standardization_window = standardization_window
        
        # Validate parameters
        self._validate_parameters()
        
        # Initialize storage
        self.data = None
        self.results = None
        self.events = None
        self._fitted = False
        
        logger.info(f"SESR initialized with timescale={timescale} days")
        
    def _validate_parameters(self):
        """Validate input parameters"""
        if self.timescale <= 0:
            raise ValueError("timescale must be positive")
        if not 0 <= self.sesr_threshold_percentile <= 100:
            raise ValueError("sesr_threshold_percentile must be between 0 and 100")
        if not 0 <= self.delta_threshold_percentile <= 100:
            raise ValueError("delta_threshold_percentile must be between 0 and 100")
            
    def fit(self, 
            data: pd.DataFrame,
            aet_col: str = 'AET',
            pet_col: str = 'PET',
            date_col: str = 'date') -> 'SESR':
        """
        Fit SESR model to input data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe containing evapotranspiration data
        aet_col : str, default='AET'
            Column name for Actual Evapotranspiration
        pet_col : str, default='PET'
            Column name for Potential Evapotranspiration
        date_col : str, default='date'
            Column name for date/time
            
        Returns
        -------
        self : SESR
            Fitted SESR object
        """
        logger.info("Fitting SESR model to data...")
        
        # Prepare data
        self.data = data.copy()
        self.aet_col = aet_col
        self.pet_col = pet_col
        self.date_col = date_col
        
        # Ensure date column is datetime
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        self.data = self.data.sort_values(date_col).reset_index(drop=True)
        
        # Calculate SESR components
        self._calculate_esr()
        self._aggregate_by_timescale()
        self._standardize_esr()
        self._calculate_changes()
        
        self._fitted = True
        logger.info("SESR model fitted successfully")
        
        return self
    
    def _calculate_esr(self):
        """Calculate Evaporative Stress Ratio (ESR = AET/PET)"""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            self.data['ESR'] = self.data[self.aet_col] / self.data[self.pet_col]
            
            # Handle invalid values
            self.data['ESR'] = self.data['ESR'].replace([np.inf, -np.inf], np.nan)
            self.data['ESR'] = self.data['ESR'].clip(0, 1)
            
        logger.debug(f"ESR calculated: mean={self.data['ESR'].mean():.3f}")
    
    def _aggregate_by_timescale(self):
        """Aggregate ESR values by specified timescale"""
        # Create period groups based on timescale
        self.data['period'] = self.data.index // self.timescale
        
        # Calculate mean ESR for each period
        period_stats = self.data.groupby('period').agg({
            'ESR': 'mean',
            self.date_col: 'first'
        }).rename(columns={'ESR': 'ESR_period_mean'})
        
        # Map back to original data
        self.data = self.data.merge(
            period_stats, 
            left_on='period', 
            right_index=True,
            suffixes=('', '_period')
        )
        
        logger.debug(f"Data aggregated into {self.data['period'].nunique()} periods")
    
    def _standardize_esr(self):
        """
        Standardize ESR values to calculate SESR
        SESR = (ESR_period - ESR_mean) / ESR_std
        """
        # Calculate rolling statistics
        window = self.standardization_window
        
        # Group by period for standardization
        period_data = self.data.groupby('period')['ESR_period_mean'].first()
        
        # Rolling statistics
        rolling_mean = period_data.rolling(
            window=window // self.timescale,
            center=True,
            min_periods=window // (2 * self.timescale)
        ).mean()
        
        rolling_std = period_data.rolling(
            window=window // self.timescale,
            center=True,
            min_periods=window // (2 * self.timescale)
        ).std()
        
        # Calculate SESR for periods
        sesr_period = (period_data - rolling_mean) / rolling_std
        
        # Map back to original data
        self.data['ESR_mean'] = self.data['period'].map(rolling_mean)
        self.data['ESR_std'] = self.data['period'].map(rolling_std)
        self.data['SESR'] = self.data['period'].map(sesr_period)
        
        # Handle NaN and inf values
        self.data['SESR'] = self.data['SESR'].replace([np.inf, -np.inf], np.nan)
        
        logger.debug(f"SESR standardized: range=[{self.data['SESR'].min():.2f}, {self.data['SESR'].max():.2f}]")
    
    def _calculate_changes(self):
        """
        Calculate standardized changes in SESR (ΔSESR)
        """
        # Calculate changes at period level
        period_sesr = self.data.groupby('period')['SESR'].first()
        period_change = period_sesr.diff()
        
        # Standardize changes
        change_mean = period_change.mean()
        change_std = period_change.std()
        
        if change_std > 0:
            period_delta_sesr = (period_change - change_mean) / change_std
        else:
            period_delta_sesr = period_change - change_mean
        
        # Map back to original data
        self.data['SESR_change'] = self.data['period'].map(period_change)
        self.data['ΔSESR'] = self.data['period'].map(period_delta_sesr)
        
        logger.debug("ΔSESR calculated for change detection")
    
    def detect_events(self, 
                     custom_criteria: Optional[Dict] = None) -> pd.DataFrame:
        """
        Detect flash drought events based on SESR criteria.
        
        Parameters
        ----------
        custom_criteria : dict, optional
            Custom criteria for event detection, overrides defaults
            
        Returns
        -------
        pd.DataFrame
            Dataframe with detected flash drought events
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first. Call .fit() method.")
        
        logger.info("Detecting flash drought events...")
        
        # Calculate thresholds
        sesr_threshold = np.nanpercentile(self.data['SESR'], self.sesr_threshold_percentile)
        delta_threshold = np.nanpercentile(self.data['ΔSESR'], self.delta_threshold_percentile)
        mean_delta_threshold = np.nanpercentile(self.data['ΔSESR'], 25)
        
        # Initialize event tracking
        self.data['is_flash_drought'] = False
        self.data['event_id'] = 0
        
        # Group by periods for event detection
        period_data = self.data.groupby('period').first().reset_index()
        
        events = []
        event_id = 0
        i = 0
        
        min_periods = self.min_duration_periods
        
        while i < len(period_data) - min_periods:
            window = period_data.iloc[i:i + min_periods]
            
            # Check criteria
            delta_below = (window['ΔSESR'].dropna() < delta_threshold).sum()
            
            if delta_below >= min_periods - 1:
                final_sesr = window['SESR'].iloc[-1]
                
                if final_sesr < sesr_threshold:
                    mean_delta = window['ΔSESR'].mean()
                    
                    if mean_delta < mean_delta_threshold:
                        # Flash drought detected
                        event_id += 1
                        
                        # Get date range for this event
                        start_period = window['period'].iloc[0]
                        end_period = window['period'].iloc[-1]
                        
                        # Mark in original data
                        mask = (self.data['period'] >= start_period) & (self.data['period'] <= end_period)
                        self.data.loc[mask, 'is_flash_drought'] = True
                        self.data.loc[mask, 'event_id'] = event_id
                        
                        # Store event info
                        event_dates = self.data[mask]
                        events.append({
                            'event_id': event_id,
                            'start_date': event_dates[self.date_col].min(),
                            'end_date': event_dates[self.date_col].max(),
                            'duration_days': (event_dates[self.date_col].max() - 
                                            event_dates[self.date_col].min()).days + 1,
                            'duration_periods': len(window),
                            'min_sesr': window['SESR'].min(),
                            'final_sesr': final_sesr,
                            'mean_delta_sesr': mean_delta
                        })
                        
                        i += min_periods
                        continue
            
            i += 1
        
        self.events = pd.DataFrame(events) if events else pd.DataFrame()
        self.results = self.data.copy()
        
        n_events = len(events)
        logger.info(f"Detected {n_events} flash drought event(s)")
        
        return self.events
    
    def get_intensity(self, method: str = 'deficit') -> pd.DataFrame:
        """
        Calculate flash drought intensity metrics.
        
        Parameters
        ----------
        method : str, default='deficit'
            Method for intensity calculation:
            - 'deficit': Based on deficit from threshold
            - 'magnitude': Based on absolute SESR values
            - 'rate': Based on rate of intensification
            
        Returns
        -------
        pd.DataFrame
            Intensity metrics for each event
        """
        if self.events is None or len(self.events) == 0:
            logger.warning("No events detected. Run detect_events() first.")
            return pd.DataFrame()
        
        intensity_data = self.events.copy()
        sesr_threshold = np.nanpercentile(self.data['SESR'], self.sesr_threshold_percentile)
        
        for idx, event in self.events.iterrows():
            event_mask = self.data['event_id'] == event['event_id']
            event_data = self.data[event_mask]
            
            if method == 'deficit':
                # Deficit-based intensity
                intensity = (sesr_threshold - event_data['SESR']).mean()
            elif method == 'magnitude':
                # Magnitude-based intensity
                intensity = abs(event_data['SESR'].mean())
            elif method == 'rate':
                # Rate-based intensity
                intensity = abs(event_data['ΔSESR'].mean())
            else:
                raise ValueError(f"Unknown method: {method}")
            
            intensity_data.loc[idx, f'intensity_{method}'] = intensity
            
            # Additional metrics
            intensity_data.loc[idx, 'peak_severity'] = event_data['SESR'].min()
            intensity_data.loc[idx, 'mean_sesr'] = event_data['SESR'].mean()
            intensity_data.loc[idx, 'total_deficit'] = (sesr_threshold - event_data['SESR']).sum()
        
        return intensity_data
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics of the analysis.
        
        Returns
        -------
        dict
            Dictionary containing analysis statistics
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first. Call .fit() method.")
        
        stats = {
            'timescale': self.timescale,
            'timescale_name': self._get_timescale_name(),
            'total_days': len(self.data),
            'total_periods': self.data['period'].nunique(),
            'flash_drought_days': self.data['is_flash_drought'].sum(),
            'number_of_events': len(self.events) if self.events is not None else 0,
            'mean_esr': self.data['ESR'].mean(),
            'mean_sesr': self.data['SESR'].mean(),
            'min_sesr': self.data['SESR'].min(),
            'max_sesr': self.data['SESR'].max(),
            'sesr_threshold': np.nanpercentile(self.data['SESR'], self.sesr_threshold_percentile),
            'mean_delta_sesr': self.data['ΔSESR'].mean(),
        }
        
        if self.events is not None and len(self.events) > 0:
            stats.update({
                'mean_event_duration_days': self.events['duration_days'].mean(),
                'max_event_duration_days': self.events['duration_days'].max(),
                'mean_event_severity': self.events['min_sesr'].mean(),
            })
        
        return stats
    
    def _get_timescale_name(self) -> str:
        """Get descriptive name for timescale"""
        names = {
            5: 'pentad',
            7: 'weekly',
            10: 'dekad',
            14: 'biweekly',
            30: 'monthly'
        }
        return names.get(self.timescale, f'{self.timescale}-day')
    
    def plot_results(self, 
                    figsize: Tuple[int, int] = (15, 10),
                    save_path: Optional[str] = None):
        """
        Generate diagnostic plots for SESR analysis.
        
        Parameters
        ----------
        figsize : tuple, default=(15, 10)
            Figure size for plots
        save_path : str, optional
            Path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.error("matplotlib required for plotting. Install with: pip install matplotlib")
            return
        
        if not self._fitted:
            raise ValueError("Model must be fitted first. Call .fit() method.")
        
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot 1: ESR time series
        axes[0].plot(self.data[self.date_col], self.data['ESR'], 
                    label='Daily ESR', alpha=0.5, color='gray')
        axes[0].plot(self.data[self.date_col], self.data['ESR_period_mean'], 
                    label=f'{self._get_timescale_name()} mean', color='blue', linewidth=2)
        axes[0].set_ylabel('ESR (AET/PET)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f'SESR Flash Drought Analysis (Timescale: {self._get_timescale_name()})')
        
        # Plot 2: SESR
        axes[1].plot(self.data[self.date_col], self.data['SESR'], 
                    color='darkblue', linewidth=1.5)
        axes[1].axhline(y=np.nanpercentile(self.data['SESR'], self.sesr_threshold_percentile),
                       color='red', linestyle='--', alpha=0.5, 
                       label=f'{self.sesr_threshold_percentile}th percentile')
        axes[1].set_ylabel('SESR (standardized)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: ΔSESR
        axes[2].plot(self.data[self.date_col], self.data['ΔSESR'], 
                    color='darkgreen', linewidth=1.5)
        axes[2].axhline(y=np.nanpercentile(self.data['ΔSESR'], self.delta_threshold_percentile),
                       color='orange', linestyle='--', alpha=0.5,
                       label=f'{self.delta_threshold_percentile}th percentile')
        axes[2].set_ylabel('ΔSESR (change rate)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Flash drought events
        axes[3].plot(self.data[self.date_col], self.data['SESR'], 
                    color='lightgray', linewidth=1)
        
        # Highlight flash drought periods
        for event_id in self.data['event_id'].unique():
            if event_id > 0:
                event_data = self.data[self.data['event_id'] == event_id]
                axes[3].fill_between(event_data[self.date_col], 
                                    event_data['SESR'].min(), 
                                    event_data['SESR'],
                                    color='red', alpha=0.3, 
                                    label=f'Event {event_id}' if event_id == 1 else '')
        
        axes[3].set_ylabel('SESR with Events')
        axes[3].set_xlabel('Date')
        if self.data['event_id'].max() > 0:
            axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def to_csv(self, 
              results_path: str = 'sesr_results.csv',
              events_path: str = 'flash_drought_events.csv'):
        """
        Save results to CSV files.
        
        Parameters
        ----------
        results_path : str
            Path for saving full results
        events_path : str
            Path for saving event summary
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first. Call .fit() method.")
        
        self.results.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        if self.events is not None and len(self.events) > 0:
            self.events.to_csv(events_path, index=False)
            logger.info(f"Events saved to {events_path}")


def example_usage():
    """
    Example usage of SESR class from FLASHCAT package
    """
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    
    # Simulate AET and PET data with drought periods
    pet = np.random.uniform(3, 8, len(dates))
    aet = np.zeros(len(dates))
    
    for i in range(len(dates)):
        # Create synthetic drought periods
        if 100 <= i <= 130 or 200 <= i <= 245:
            aet[i] = pet[i] * np.random.uniform(0.2, 0.5)
        else:
            aet[i] = pet[i] * np.random.uniform(0.7, 0.95)
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'AET': aet,
        'PET': pet
    })
    
    print("="*60)
    print("FLASHCAT - SESR Flash Drought Analysis")
    print("="*60)
    
    # Example 1: Pentad (5-day) analysis
    print("\n1. PENTAD ANALYSIS (5-day periods)")
    print("-"*40)
    sesr_pentad = SESR(timescale=5)
    sesr_pentad.fit(data)
    events_pentad = sesr_pentad.detect_events()
    intensity_pentad = sesr_pentad.get_intensity()
    stats_pentad = sesr_pentad.get_statistics()
    
    print(f"Timescale: {stats_pentad['timescale_name']}")
    print(f"Events detected: {stats_pentad['number_of_events']}")
    print(f"Flash drought days: {stats_pentad['flash_drought_days']}")
    
    # Example 2: Weekly (7-day) analysis
    print("\n2. WEEKLY ANALYSIS (7-day periods)")
    print("-"*40)
    sesr_weekly = SESR(timescale=7)
    sesr_weekly.fit(data)
    events_weekly = sesr_weekly.detect_events()
    stats_weekly = sesr_weekly.get_statistics()
    
    print(f"Timescale: {stats_weekly['timescale_name']}")
    print(f"Events detected: {stats_weekly['number_of_events']}")
    print(f"Flash drought days: {stats_weekly['flash_drought_days']}")
    
    # Example 3: Biweekly (14-day) analysis
    print("\n3. BIWEEKLY ANALYSIS (14-day periods)")
    print("-"*40)
    sesr_biweekly = SESR(timescale=14)
    sesr_biweekly.fit(data)
    events_biweekly = sesr_biweekly.detect_events()
    stats_biweekly = sesr_biweekly.get_statistics()
    
    print(f"Timescale: {stats_biweekly['timescale_name']}")
    print(f"Events detected: {stats_biweekly['number_of_events']}")
    print(f"Flash drought days: {stats_biweekly['flash_drought_days']}")
    
    # Save results
    print("\n" + "="*60)
    print("Saving results...")
    sesr_pentad.to_csv('sesr_pentad_results.csv', 'pentad_events.csv')
    
    # Generate plot for pentad analysis
    try:
        sesr_pentad.plot_results(save_path='sesr_analysis.png')
    except:
        print("Plotting skipped (matplotlib not available)")
    
    return sesr_pentad


if __name__ == "__main__":
    # Run example
    sesr = example_usage()