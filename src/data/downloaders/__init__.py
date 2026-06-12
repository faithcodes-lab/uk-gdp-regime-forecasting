"""Per-source data downloaders.

Each downloader reads its block from ``config/pipeline.yaml``, validates
the raw response against the source-specific pandera schema, writes a
lineage record, and saves the raw CSV plus a snapshot of the raw API
response. 
"""

from src.data.downloaders.boe import download_all_boe, download_boe_series
from src.data.downloaders.fred import download_all_fred, download_fred_series
from src.data.downloaders.ons import download_all_ons, download_ons_series

__all__ = [
    "download_all_boe",
    "download_all_fred",
    "download_all_ons",
    "download_boe_series",
    "download_fred_series",
    "download_ons_series",
]
