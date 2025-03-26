import sys
import io
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pmdarima import auto_arima
from langgraph.graph import StateGraph, END
from sklearn.ensemble import IsolationForest
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timedelta
import os
import re
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Define enhanced state model
class FinancialState(BaseModel):
    user_query: str = Field(..., description="Original user input")
    symbol: Optional[str] = Field(None, description="Detected stock symbol")
    raw_data: Optional[Dict[str, Any]] = None
    processed_data: Optional[Dict[str, Any]] = None
    model: Optional[Any] = None
    predictions: Optional[List[float]] = None
    anomalies: Optional[List[int]] = None
    insights: Optional[str] = None
    visualizations: List[str] = []
    error: Optional[str] = None


def data_ingestion_node(state: FinancialState) -> FinancialState:
    """Fetch data from APIs"""
    if state.error or not state.symbol:
        return state
    
    state_dict = state.model_dump()
    
    try:
        # Stock Data
        stock_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={state.symbol}&apikey={os.environ['ALPHA_VANTAGE']}"
        stock_response = requests.get(stock_url)
        stock_response.raise_for_status()
        
        # News Data
        news_url = "https://newsapi.org/v2/everything"
        news_params = {
            'q': f"{state.symbol} stock",
            'sortBy': 'relevancy',
            'apiKey': os.environ['NEWS_API_KEY'],
            'language': 'en',
            'pageSize': 10
        }
        news_response = requests.get(news_url, params=news_params)
        news_response.raise_for_status()
        
        state_dict["raw_data"] = {
            "stock": stock_response.json(),
            "news": news_response.json()
        }
        state_dict["error"] = None
    
    except Exception as e:
        state_dict["error"] = f"Data fetch failed: {str(e)}"
    
    return FinancialState(**state_dict)