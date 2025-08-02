import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import re

def normalize_financial_numbers(text: str) -> str:
    """Convert financial numbers to a standard format."""
    # Convert B/M/K to actual numbers
    def convert_to_number(match):
        num = float(match.group(1))
        multiplier = match.group(2).upper()
        if multiplier == 'B':
            num *= 1e9
        elif multiplier == 'M':
            num *= 1e6
        elif multiplier == 'K':
            num *= 1e3
        return f"{num:,.2f}"

    # Find numbers with B/M/K and convert them
    pattern = r'(\d+\.?\d*)\s*(B|M|K|b|m|k)'
    text = re.sub(pattern, convert_to_number, text)
    return text

def extract_temporal_context(text: str) -> Dict[str, Any]:
    """Extract dates and time-related information from text."""
    # Extract dates using regex
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
    dates = re.findall(date_pattern, text)
    
    # Extract quarters
    quarter_pattern = r'Q[1-4]\s*\d{4}'
    quarters = re.findall(quarter_pattern, text)
    
    # Extract year-over-year references
    yoy_pattern = r'(?i)(?:year[- ]over[- ]year|YoY|y-o-y)'
    yoy_mentions = re.findall(yoy_pattern, text)
    
    return {
        "dates": dates,
        "quarters": quarters,
        "yoy_mentions": yoy_mentions
    }

def chunk_document(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split document into chunks with overlap."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def extract_financial_metrics(text: str) -> Dict[str, float]:
    """Extract key financial metrics from text."""
    metrics = {}
    
    # Revenue
    revenue_pattern = r'Revenue:\s*\$?([\d,.]+)(?:B|M|K)?'
    if match := re.search(revenue_pattern, text):
        metrics['revenue'] = float(match.group(1).replace(',', ''))
    
    # Net Income
    income_pattern = r'Net Income:\s*\$?([\d,.]+)(?:B|M|K)?'
    if match := re.search(income_pattern, text):
        metrics['net_income'] = float(match.group(1).replace(',', ''))
    
    # EPS
    eps_pattern = r'EPS:\s*\$?([\d,.]+)'
    if match := re.search(eps_pattern, text):
        metrics['eps'] = float(match.group(1).replace(',', ''))
    
    return metrics
