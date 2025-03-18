import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import math
import requests
from io import StringIO
import logging
from typing import Tuple, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global DataFrame to cache data
JOSAA_DATA = None

def load_data() -> pd.DataFrame:
    """
    Load JoSAA counseling data from remote source
    
    Returns:
        pd.DataFrame: Loaded and preprocessed dataframe
    """
    global JOSAA_DATA
    try:
        # Multiple fallback URLs
        urls = [
            "https://raw.githubusercontent.com/JARAWA/JOSAA180325/refs/heads/main/josaa2024_cutoff.csv",
            "https://raw.githubusercontent.com/username/repo/main/josaa2024_cutoff.csv"  # Backup URL
        ]
        
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                df = pd.read_csv(StringIO(response.text))
                
                # Data preprocessing
                df["Opening Rank"] = pd.to_numeric(df["Opening Rank"], errors="coerce").fillna(9999999)
                df["Closing Rank"] = pd.to_numeric(df["Closing Rank"], errors="coerce").fillna(9999999)
                df["Round"] = df["Round"].astype(str)
                
                # Normalize columns
                df['Category'] = df['Category'].str.lower()
                df['Academic Program Name'] = df['Academic Program Name'].str.lower()
                df['College Type'] = df['College Type'].str.upper()
                
                JOSAA_DATA = df
                logger.info(f"Data loaded successfully. Total rows: {len(df)}")
                return df
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to load from {url}: {e}")
        
        raise ValueError("Could not load data from any source")
    
    except Exception as e:
        logger.error(f"Critical error in data loading: {e}")
        raise

def get_unique_branches() -> list:
    """
    Retrieve unique academic branches
    
    Returns:
        list: Sorted list of unique branches
    """
    global JOSAA_DATA
    if JOSAA_DATA is None:
        load_data()
    
    try:
        unique_branches = sorted(
            JOSAA_DATA['Academic Program Name']
            .dropna()
            .str.strip()
            .str.lower()
            .unique()
            .tolist()
        )
        return ["All"] + unique_branches
    except Exception as e:
        logger.error(f"Error getting branches: {e}")
        return ["All"]

def hybrid_probability_calculation(rank: int, opening_rank: float, closing_rank: float) -> float:
    """
    Advanced probability calculation method
    
    Args:
        rank (int): JEE Rank
        opening_rank (float): Opening rank of college
        closing_rank (float): Closing rank of college
    
    Returns:
        float: Calculated admission probability
    """
    try:
        # Logistic probability calculation
        M = (opening_rank + closing_rank) / 2
        S = max((closing_rank - opening_rank) / 10, 1)
        
        logistic_prob = 1 / (1 + math.exp((rank - M) / S)) * 100

        # Piecewise probability calculation
        if rank < opening_rank:
            improvement = (opening_rank - rank) / opening_rank
            piece_wise_prob = 99.0 if improvement >= 0.5 else 96 + (improvement * 6)
        elif rank == opening_rank:
            piece_wise_prob = 95.0
        elif rank < closing_rank:
            range_width = closing_rank - opening_rank
            position = (rank - opening_rank) / range_width
            
            if position <= 0.2:
                piece_wise_prob = 94 - (position * 70)
            elif position <= 0.5:
                piece_wise_prob = 80 - ((position - 0.2) / 0.3 * 20)
            elif position <= 0.8:
                piece_wise_prob = 60 - ((position - 0.5) / 0.3 * 20)
            else:
                piece_wise_prob = 40 - ((position - 0.8) / 0.2 * 20)
        else:
            piece_wise_prob = 0.0

        # Final probability calculation
        final_prob = (logistic_prob * 0.7 + piece_wise_prob * 0.3)
        return round(max(0, min(final_prob, 100)), 2)
    
    except Exception as e:
        logger.error(f"Probability calculation error: {e}")
        return 0.0

# Add other utility functions like predict_preferences, get_college_details as needed
