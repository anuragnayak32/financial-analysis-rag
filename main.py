import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
import yfinance as yf
from datetime import datetime, timedelta

from config import RAGConfig, PromptTemplates
from utils import (
    normalize_financial_numbers,
    extract_temporal_context,
    chunk_document,
    extract_financial_metrics
)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
config = RAGConfig()

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

class FinancialRAGSystem:
    def __init__(self):
        try:
            # Initialize Gemini model with the correct model name
            models = genai.list_models()
            model_name = next((m.name for m in models if 'generateContent' in m.supported_generation_methods), None)
            if not model_name:
                raise Exception("No suitable Gemini model found")
            self.model = genai.GenerativeModel(model_name)
            
            # Initialize sentence transformer for embeddings
            self.embedding_model = SentenceTransformer(config.embed_model_name)
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(path=config.vector_db_path)
            try:
                self.collection = self.chroma_client.get_collection(name="financial_docs")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="financial_docs",
                    metadata={"hnsw:space": "cosine"}
                )
            
            # Test the model configuration
            test_response = self.model.generate_content("Test connection")
            if not test_response:
                raise Exception("Failed to initialize Gemini model")
                
        except Exception as e:
            print(f"Error initializing system: {str(e)}")
            print("Please check your API key and internet connection")

    def load_financial_data(self, data_directory: str):
        """Load and process financial documents into the vector store."""
        try:
            documents = []
            for file in os.listdir(data_directory):
                if file.endswith('.txt'):
                    file_path = os.path.join(data_directory, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            # Normalize financial numbers
                            text = normalize_financial_numbers(text)
                            
                            # Extract temporal context
                            temporal_info = extract_temporal_context(text)
                            
                            # Chunk the document
                            chunks = chunk_document(text, config.chunk_size, config.chunk_overlap)
                            
                            # Process each chunk
                            for i, chunk in enumerate(chunks):
                                doc_id = f"{os.path.basename(file_path)}_{i}"
                                
                                # Extract financial metrics
                                metrics = extract_financial_metrics(chunk)
                                
                                # Create embeddings
                                embedding = self.embedding_model.encode(chunk)
                                
                                # Format metadata for ChromaDB (flatten the structure)
                                metadata = {
                                    "source": file_path,
                                    "chunk_id": str(i),
                                    "dates": ",".join(temporal_info.get('dates', [])),
                                    "quarters": ",".join(temporal_info.get('quarters', [])),
                                    "yoy_mentions": str(len(temporal_info.get('yoy_mentions', []))),
                                }
                                
                                # Add numeric metrics to metadata
                                for metric_name, metric_value in metrics.items():
                                    metadata[f"metric_{metric_name}"] = metric_value
                                
                                # Add to ChromaDB
                                self.collection.add(
                                    documents=[chunk],
                                    embeddings=[embedding.tolist()],
                                    ids=[doc_id],
                                    metadatas=[metadata]
                                )
                                
                                documents.append({
                                    "id": doc_id,
                                    "content": chunk,
                                    "source": file_path,
                                    "metrics": metrics
                                })
                                
                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")
            
            if not documents:
                print("No documents processed")
                return False
                
            print(f"Successfully processed {len(documents)} document chunks")
            return True
        except Exception as e:
            print(f"Error loading financial data: {str(e)}")
            return False

    def fetch_financial_news(self, symbol: str) -> List[Dict]:
        """Fetch recent financial news for a given stock symbol."""
        # This is a placeholder. In a real implementation, you would use 
        # a proper financial API service
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            # Process and extract news (simplified version)
            news = []
            # Add proper news extraction logic here
            return news
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return []

    def analyze_query(self, query: str):
        """
        Process a user query and generate a response using RAG with vector similarity search.
        """
        try:
            # Check if model was initialized successfully
            if not hasattr(self, 'model'):
                return "The AI model is not initialized. Please check your API key and internet connection."

            # For simple questions about financial metrics, try direct lookup first
            if "revenue" in query.lower():
                try:
                    # Search specifically for chunks with revenue information
                    results = self.collection.query(
                        query_texts=["revenue financial metrics"],
                        n_results=1
                    )
                    if results and results['documents']:
                        # Extract revenue directly from the document
                        doc = results['documents'][0][0]
                        if "Revenue:" in doc:
                            revenue_line = [line for line in doc.split('\n') if "Revenue:" in line][0]
                            return f"Based on the financial report: {revenue_line}"
                except Exception:
                    pass  # Fall back to full RAG if direct lookup fails
            
            # Generate query embedding for full RAG search
            query_embedding = self.embedding_model.encode(query)
            
            # Search for relevant chunks
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=config.top_k
            )
            
            if not results or not results['documents']:
                return "No relevant information found in the database."
            
            # Combine relevant chunks with their temporal context
            contexts = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                quarters = metadata.get('quarters', '').split(',')
                dates = metadata.get('dates', '').split(',')
                temporal_info = quarters[0] if quarters and quarters[0] else (dates[0] if dates and dates[0] else 'N/A')
                contexts.append(f"Document (Date: {temporal_info}):\n{doc}")
            
            context = "\n\n".join(contexts)
            
            # Construct the prompt using template
            prompt = PromptTemplates.FINANCIAL_ANALYSIS.format(
                context=context,
                question=query
            )

            try:
                # For rate-limited API, first try a simple direct response
                simple_response = self.generate_simple_response(query, context)
                if simple_response:
                    return simple_response
                    
                # If no simple response, try using Gemini API
                response = self.model.generate_content(prompt)
                if response and hasattr(response, 'text'):
                    return response.text
                else:
                    return "Sorry, I couldn't generate a response. Please try again."
            except Exception as api_error:
                print(f"API Error details: {str(api_error)}")
                # If we hit rate limits, fall back to simple response
                if "429" in str(api_error):
                    simple_response = self.generate_simple_response(query, context)
                    if simple_response:
                        return simple_response
                return "Rate limit reached. Here's what I found in the documents: " + context

        except Exception as e:
            print(f"Error details: {str(e)}")
            return "An error occurred while processing your query. Please try again."

    def generate_simple_response(self, query: str, context: str) -> str:
        """Generate a simple response without using the AI model."""
        query = query.lower()
        
        # Look for specific metrics in the query
        if "revenue" in query:
            # First try to find quarter-specific revenue
            quarter_pattern = r'Q[1-4]\s*\d{4}.*?revenue.*?\$?([\d,.]+[BM]?)'
            if match := re.search(quarter_pattern, context.lower()):
                return f"In {match.group(0).split()[0]}, Apple reported revenue of ${match.group(1)}"
            
            # Fallback to general revenue
            revenue_pattern = r'Revenue:\s*\$?([\d,.]+[BM]?)'
            if match := re.search(revenue_pattern, context):
                return f"Apple reported revenue of ${match.group(1)}"
                
        elif "eps" in query or "earnings per share" in query:
            eps_pattern = r'EPS:\s*\$?([\d,.]+)'
            if match := re.search(eps_pattern, context):
                return f"The EPS reported was ${match.group(1)}"
                
        elif "margin" in query:
            margin_pattern = r'Gross Margin:\s*([\d,.]+)%'
            if match := re.search(margin_pattern, context):
                return f"The gross margin was {match.group(1)}%"
                
        # If no specific metric found, return the most relevant sentence
        sentences = context.split('.')
        relevant_sentences = [s for s in sentences if any(word in s.lower() for word in query.split())]
        if relevant_sentences:
            return relevant_sentences[0].strip()
            
        return None
        
    def analyze_market_trends(self, symbol: str, timeframe: str = '1y'):
        """
        Analyze market trends for a specific symbol using yfinance.
        """
        try:
            # Convert timeframe to number of days
            timeframe_map = {
                '1d': '1d',
                '1w': '7d',
                '1m': '30d',
                '3m': '90d',
                '6m': '180d',
                '1y': '365d'
            }
            period = timeframe_map.get(timeframe, '365d')
            
            # Fetch stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            
            if hist.empty:
                return f"No data found for {symbol}"
            
            # Calculate key metrics
            current_price = hist['Close'][-1]
            start_price = hist['Close'][0]
            price_change = ((current_price - start_price) / start_price) * 100
            high = hist['High'].max()
            low = hist['Low'].min()
            avg_volume = hist['Volume'].mean()
            
            analysis = f"""Market Analysis for {symbol} over {timeframe}:
            
Current Price: ${current_price:.2f}
Price Change: {price_change:.2f}%
Period High: ${high:.2f}
Period Low: ${low:.2f}
Average Daily Volume: {avg_volume:,.0f} shares

Key Insights:
- {'Upward' if price_change > 0 else 'Downward'} trend with {abs(price_change):.2f}% {'gain' if price_change > 0 else 'loss'}
- Trading range: ${low:.2f} - ${high:.2f}
- {'High' if avg_volume > 1000000 else 'Moderate' if avg_volume > 100000 else 'Low'} trading volume
"""
            # Prepare graph data
            graph_data = {
                'dates': hist.index,
                'prices': hist['Close'],
                'volume': hist['Volume']
            }
            
            return analysis, graph_data
            
        except Exception as e:
            return f"Error analyzing market trends: {str(e)}"

def main():
    # Initialize the RAG system
    rag_system = FinancialRAGSystem()
    
    # Example usage
    print("Financial Analysis RAG System")
    print("============================")
    
    # Load data (specify your data directory)
    data_dir = "financial_data"  # Create this directory and add your financial documents
    if os.path.exists(data_dir):
        success = rag_system.load_financial_data(data_dir)
        if success:
            print("Financial data loaded successfully!")
        else:
            print("Failed to load financial data.")

    while True:
        print("\nOptions:")
        print("1. Ask a financial question")
        print("2. Analyze market trends")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            query = input("\nEnter your financial question: ")
            response = rag_system.analyze_query(query)
            print("\nAnalysis:")
            print(response)
        
        elif choice == '2':
            symbol = input("\nEnter stock symbol: ")
            timeframe = input("Enter timeframe (e.g., 1d, 1w, 1m, 1y): ")
            response = rag_system.analyze_market_trends(symbol, timeframe)
            print("\nMarket Trends:")
            print(response)
        
        elif choice == '3':
            print("\nThank you for using the Financial Analysis RAG System!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
