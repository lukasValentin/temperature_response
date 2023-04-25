"""
This module extracts S2 derived LAI values for those farms
and parcels with in-situ data available. We focus on the
stem elongation phase, i.e., we take all S2 scenes between
BBCH 31 and 59 rated in-situ.
"""

from pathlib import Path


if __name__ == '__main__':
    