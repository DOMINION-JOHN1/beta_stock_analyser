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

def preprocessing_node(state: FinancialState) -> FinancialState:
    if state.error or not state.raw_data:
        return state
    
    state_dict = state.model_dump()
    
    try:
        # Process stock data with proper date sorting
        stock_df = pd.DataFrame(
            state.raw_data["stock"]["Time Series (Daily)"]
        ).T.rename(columns={
            "1. open": "Open",
            "4. close": "Close"
        })
        stock_df.index = pd.to_datetime(stock_df.index)
        stock_df = stock_df.sort_index(ascending=True)  # Ensure ascending order
        stock_df = stock_df[["Open", "Close"]].astype(float)
        
        # Process news with Unicode handling
        news_content = "\n".join([
            f"{article['title']}: {article['description']}"
            for article in state.raw_data["news"]["articles"]
            if article.get('description')
        ]).encode('utf-8', 'ignore').decode('utf-8')
        
        state_dict["processed_data"] = {
            "stock": stock_df,
            "news": news_content[:5000]  # Limit context length
        }
        state_dict["error"] = None
    
    except Exception as e:
        state_dict["error"] = f"Preprocessing failed: {str(e)}"
    
    return FinancialState(**state_dict)