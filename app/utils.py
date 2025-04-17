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
    Load JoSAA counseling data from CSV
    
    Returns:
        pd.DataFrame: Loaded and preprocessed dataframe
    """
    global JOSAA_DATA
    try:
        # Get path to CSV in root directory
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(current_dir, 'josaa2024_cutoff.csv')
        
        logger.info(f"Attempting to load CSV from: {csv_path}")
        
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found at: {csv_path}")
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Data preprocessing
        df["Opening Rank"] = pd.to_numeric(df["Opening Rank"], errors="coerce").fillna(9999999)
        df["Closing Rank"] = pd.to_numeric(df["Closing Rank"], errors="coerce").fillna(9999999)
        df["Round"] = df["Round"].astype(str)
        
        # Normalize columns
        df['Category'] = df['Category'].str.lower()
        df['Academic Program Name'] = df['Academic Program Name'].str.lower()
        df['College Type'] = df['College Type'].str.upper()
        df['Institute'] = df['Institute'].str.strip()
        
        JOSAA_DATA = df
        logger.info(f"Data loaded successfully. Total rows: {len(df)}")
        return df
    
    except Exception as e:
        logger.error(f"Error in load_data: {e}")
        return pd.DataFrame(columns=[
            'Institute', 'Academic Program Name', 'Category', 
            'Opening Rank', 'Closing Rank', 'Round', 'College Type'
        ])

def get_unique_branches() -> list:
    """
    Retrieve unique academic branches with enhanced error checking
    
    Returns:
        list: Sorted list of unique branches
    """
    global JOSAA_DATA
    try:
        if JOSAA_DATA is None:
            logger.warning("JOSAA_DATA is None, attempting to load data...")
            JOSAA_DATA = load_data()
            
        # Debug information about the DataFrame
        logger.info(f"DataFrame shape: {JOSAA_DATA.shape}")
        logger.info(f"DataFrame columns: {JOSAA_DATA.columns.tolist()}")
        
        if 'Academic Program Name' not in JOSAA_DATA.columns:
            logger.error("'Academic Program Name' column not found in DataFrame")
            return {
                "status": "error",
                "message": "Academic Program Name column missing",
                "branches": ["All"]
            }
        
        # Get unique branches and clean them
        branches = JOSAA_DATA['Academic Program Name'].dropna()
        logger.info(f"Number of non-null branches before cleaning: {len(branches)}")
        
        branches = branches.apply(lambda x: str(x).strip().lower())
        unique_branches = sorted(list(set(branches)))
        
        logger.info(f"Number of unique branches found: {len(unique_branches)}")
        logger.debug(f"Unique branches: {unique_branches[:5]}...")  # Show first 5 branches
        
        # Add "All" option at the beginning
        unique_branches = ["All"] + unique_branches
        
        return {
            "status": "success",
            "message": f"Successfully retrieved {len(unique_branches)} branches",
            "branches": unique_branches
        }
        
    except Exception as e:
        logger.error(f"Error in get_unique_branches: {str(e)}")
        return {
            "status": "error",
            "message": f"Error retrieving branches: {str(e)}",
            "branches": ["All"]
        }

def hybrid_probability_calculation(rank: int, opening_rank: float, closing_rank: float) -> float:
    """
    Calculate admission probability using hybrid method
    
    Args:
        rank (int): Student's rank
        opening_rank (float): Opening rank for the program
        closing_rank (float): Closing rank for the program
    
    Returns:
        float: Calculated probability percentage
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
        if rank <= opening_rank:
            improvement = (opening_rank - rank) / opening_rank
            piece_wise_prob = 99.0 if improvement >= 0.5 else 96 + (improvement * 6)
        elif rank > closing_rank:
            piece_wise_prob = 0.0
        else:
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

        # Final probability calculation
        final_prob = (logistic_prob * 0.7 + piece_wise_prob * 0.3)
        return round(max(0, min(final_prob, 100)), 2)
    
    except Exception as e:
        logger.error(f"Error in probability calculation: {e}")
        return 0.0

def predict_preferences(
    jee_rank: int, 
    category: str, 
    college_type: str, 
    preferred_branch: str, 
    round_no: int, 
    min_probability: float
) -> Tuple[pd.DataFrame, Optional[go.Figure]]:
    """
    Predict college preferences based on input parameters
    
    Args:
        jee_rank (int): Student's JEE rank
        category (str): Category (e.g., OPEN, SC, ST)
        college_type (str): College type (e.g., IIT, NIT)
        preferred_branch (str): Preferred branch
        round_no (int): Counseling round number
        min_probability (float): Minimum probability threshold
    
    Returns:
        Tuple containing DataFrame of predictions and optional plot
    """
    try:
        global JOSAA_DATA
        if JOSAA_DATA is None:
            JOSAA_DATA = load_data()

        # Filter DataFrame based on input parameters
        df = JOSAA_DATA.copy()
        
        if category.lower() != 'all':
            df = df[df['Category'] == category.lower()]
        
        if college_type.upper() != 'ALL':
            df = df[df['College Type'] == college_type.upper()]
        
        if preferred_branch.lower() != 'all':
            df = df[df['Academic Program Name'] == preferred_branch.lower()]
        
        df = df[df['Round'] == str(round_no)]

        # Calculate probabilities
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
            'Academic Program Name', 
            'Category',
            'College Type',
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
            labels={'Admission Probability (%)': 'Probability', 'count': 'Number of Colleges'},
            nbins=20
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Probability (%)",
            yaxis_title="Number of Colleges"
        )

        return result, fig

    except Exception as e:
        logger.error(f"Error in predict_preferences: {e}")
        return pd.DataFrame(), None

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
    try:
        if JOSAA_DATA is None:
            JOSAA_DATA = load_data()
        
        # Normalize input
        institute = institute.lower()
        branch = branch.lower()
        
        # Filter college data
        college_data = JOSAA_DATA[
            (JOSAA_DATA['Institute'].str.lower() == institute) & 
            (JOSAA_DATA['Academic Program Name'].str.lower() == branch)
        ]
        
        if college_data.empty:
            return {"error": "College not found"}
        
        # Aggregate details
        details = {
            "institute": institute.title(),
            "branch": branch.title(),
            "total_seats": int(college_data['Total Seats'].iloc[0]) if 'Total Seats' in college_data.columns else 0,
            "min_rank": int(college_data['Opening Rank'].min()),
            "max_rank": int(college_data['Closing Rank'].max()),
            "rounds": sorted(college_data['Round'].unique().tolist()),
            "categories": sorted(college_data['Category'].unique().tolist()),
            "trend": {
                "opening_ranks": college_data['Opening Rank'].tolist(),
                "closing_ranks": college_data['Closing Rank'].tolist(),
                "rounds": college_data['Round'].tolist()
            }
        }
        
        return details
    
    except Exception as e:
        logger.error(f"Error in get_college_details: {e}")
        return {"error": str(e)}
