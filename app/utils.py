import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import math
import logging
from typing import Tuple, Optional, Dict, Any
import os

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
    Load JoSAA counseling data from local CSV
    
    Returns:
        pd.DataFrame: Loaded and preprocessed dataframe
    """
    global JOSAA_DATA
    try:
        # Determine the path to the CSV file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'josaa2024_cutoff.csv')
        
        logger.info(f"Attempting to load CSV from: {csv_path}")
        
        # Check if file exists
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found at: {csv_path}")
            # Look for alternative CSV files in the directory
            data_dir = os.path.join(project_root, 'data')
            if os.path.exists(data_dir):
                csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                if csv_files:
                    alternative_path = os.path.join(data_dir, csv_files[0])
                    logger.info(f"Using alternative CSV file: {alternative_path}")
                    csv_path = alternative_path
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Log the column names to help diagnose issues
        logger.info(f"CSV columns: {df.columns.tolist()}")
        
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
    
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {e}")
        # Return an empty DataFrame with expected columns to avoid crashing
        return pd.DataFrame(columns=[
            'Institute', 'Academic Program Name', 'Category', 
            'Opening Rank', 'Closing Rank', 'Round', 'College Type'
        ])
    except Exception as e:
        logger.error(f"Critical error in data loading: {e}")
        # Return an empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'Institute', 'Academic Program Name', 'Category', 
            'Opening Rank', 'Closing Rank', 'Round', 'College Type'
        ])

def get_unique_branches() -> list:
    """
    Retrieve unique academic branches
    
    Returns:
        list: Sorted list of unique branches
    """
    global JOSAA_DATA
    try:
        if JOSAA_DATA is None:
            JOSAA_DATA = load_data()
        
        # Log the shape and first few rows to verify data is loaded
        logger.info(f"Data shape: {JOSAA_DATA.shape}")
        logger.info(f"First few rows: {JOSAA_DATA.head(2).to_dict(orient='records')}")
        
        # Check if 'Academic Program Name' column exists
        if 'Academic Program Name' not in JOSAA_DATA.columns:
            logger.error("'Academic Program Name' column not found in data")
            return ["All"]  # Return at least "All" option
        
        # Get unique branches with robust error handling
        try:
            unique_branches = sorted(
                JOSAA_DATA['Academic Program Name']
                .dropna()
                .apply(lambda x: x.strip().lower() if isinstance(x, str) else x)
                .unique()
                .tolist()
            )
            logger.info(f"Found {len(unique_branches)} unique branches")
            return ["All"] + unique_branches
        except Exception as e:
            logger.error(f"Error processing branches: {e}")
            return ["All"]
    except Exception as e:
        logger.error(f"Error getting branches: {e}")
        return ["All"]

def predict_preferences(
    jee_rank: int, 
    category: str, 
    college_type: str, 
    preferred_branch: str, 
    round_no: int, 
    min_probability: float
) -> Tuple[pd.DataFrame, None, Optional[go.Figure]]:
    """
    Predict college preferences based on input parameters
    """
    try:
        global JOSAA_DATA
        if JOSAA_DATA is None:
            JOSAA_DATA = load_data()

        # Filter DataFrame based on input parameters
        df = JOSAA_DATA.copy()
        
        # Apply filters
        df['Category'] = df['Category'].str.lower()
        df['Academic Program Name'] = df['Academic Program Name'].str.lower()
        df['College Type'] = df['College Type'].str.upper()
        
        if category.lower() != 'all':
            df = df[df['Category'] == category.lower()]
        
        if college_type.upper() != 'ALL':
            df = df[df['College Type'] == college_type.upper()]
        
        if preferred_branch.lower() != 'all':
            df = df[df['Academic Program Name'] == preferred_branch.lower()]
        
        df = df[df['Round'] == str(round_no)]

        # Probability calculation
        df['Admission Probability (%)'] = df.apply(
            lambda row: hybrid_probability_calculation(
                jee_rank, 
                row['Opening Rank'], 
                row['Closing Rank']
            ), 
            axis=1
        )

        # Filter by minimum probability
        df = df[df['Admission Probability (%)'] >= min_probability]
        
        # Sort and prepare result
        df = df.sort_values('Admission Probability (%)', ascending=False)
        df['Preference'] = range(1, len(df) + 1)

        # Select and rename columns
        result = df[[
            'Preference', 
            'Institute', 
            'College Type', 
            'Location', 
            'Academic Program Name', 
            'Opening Rank', 
            'Closing Rank', 
            'Admission Probability (%)'
        ]].rename(columns={
            'Academic Program Name': 'Branch'
        })

        # Create probability distribution plot
        fig = px.histogram(
            result, 
            x='Admission Probability (%)', 
            title='Admission Probability Distribution',
            labels={'Admission Probability (%)': 'Probability', 'count': 'Number of Colleges'}
        )

        return result, None, fig

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return pd.DataFrame({"Error": [str(e)]}), None, None

def hybrid_probability_calculation(rank: int, opening_rank: float, closing_rank: float) -> float:
    """
    Advanced probability calculation method
    """
    try:
        # Prevent division by zero
        if opening_rank == closing_rank:
            return 0.0 if rank > opening_rank else 100.0

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

def get_college_details(institute: str, branch: str) -> Dict[str, Any]:
    """
    Retrieve detailed information about a specific college
    
    Args:
        institute (str): Name of the institute
        branch (str): Academic branch
    
    Returns:
        Dict containing college details
    """
    global JOSAA_DATA
    if JOSAA_DATA is None:
        JOSAA_DATA = load_data()
    
    try:
        # Normalize input
        institute = institute.lower()
        branch = branch.lower()
        
        # Filter and aggregate college details
        college_data = JOSAA_DATA[
            (JOSAA_DATA['Institute'].str.lower() == institute) & 
            (JOSAA_DATA['Academic Program Name'].str.lower() == branch)
        ]
        
        if college_data.empty:
            return {"error": "College not found"}
        
        return {
            "institute": institute.title(),
            "branch": branch.title(),
            "total_seats": int(college_data['Total Seats'].sum()) if 'Total Seats' in college_data.columns else 0,
            "min_rank": int(college_data['Opening Rank'].min()),
            "max_rank": int(college_data['Closing Rank'].max()),
            "rounds": college_data['Round'].unique().tolist()
        }
    
    except Exception as e:
        logging.error(f"Error fetching college details: {e}")
        return {"error": str(e)}
