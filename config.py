from pydantic import BaseModel
from typing import List, Optional

class RAGConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    temperature: float = 0.1
    top_k: int = 5
    vector_db_path: str = "vector_store"
    
class PromptTemplates:
    FINANCIAL_ANALYSIS = """You are an expert financial analyst assistant. Using only the provided context, 
    analyze the information and provide insights. Focus on:
    
    1. Key financial metrics and their trends
    2. Temporal context and historical comparisons
    3. Risk factors and their potential impact
    4. Market position and competitive analysis
    
    Context: {context}
    
    Question: {question}
    
    Remember to:
    - Cite specific numbers and data points
    - Highlight temporal trends and changes
    - Discuss potential risks
    - Only use information from the provided context
    """
    
    MARKET_ANALYSIS = """Analyze the following market data and provide insights on:
    
    1. Price trends and momentum
    2. Volume analysis
    3. Comparative performance
    4. Key technical indicators
    
    Data: {context}
    Timeframe: {timeframe}
    
    Question: {question}
    """
