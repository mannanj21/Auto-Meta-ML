import pandas as pd
from fastapi import HTTPException

def validate_csv(file_content):
    """Validate uploaded CSV file"""
    try:
        df = pd.read_csv(file_content)
        
        if df.shape[0] < 10:
            raise HTTPException(400, "Dataset must have at least 10 rows")
        
        if df.shape[1] < 2:
            raise HTTPException(400, "Dataset must have at least 2 columns")
        
        return True
    except pd.errors.EmptyDataError:
        raise HTTPException(400, "CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(400, "Invalid CSV format")
