"""
Frechet Inception Distance (FID) calculation for dataset comparison.
"""

from .fid_calculator import calculate_fid, calculate_fid_from_datasets, extract_features

__all__ = ['calculate_fid', 'calculate_fid_from_datasets', 'extract_features']
