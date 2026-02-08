"""
Frechet Inception Distance (FID) calculation for dataset comparison.
"""

from .fid_calculator import calculate_fid, extract_features

__all__ = ['calculate_fid', 'extract_features']

