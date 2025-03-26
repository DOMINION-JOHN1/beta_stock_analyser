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

load_dotenv()

# Initialize Groq client
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.1,
    max_tokens=1024,
    api_key=os.environ["GROQ_API_KEY"]
)

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

# Define all nodes with Pydantic v2 compatibility
def process_query_node(state: FinancialState) -> FinancialState:
    """Extract stock symbol from natural language query"""
    if state.error:
        return state
    
    state_dict = state.model_dump()
    
    try:
        prompt = f"""Analyze this financial query and extract the company stock ticker symbol:
        Query: {state.user_query}
        Respond ONLY with the 1-5 character uppercase ticker symbol."""
        
        response = llm.invoke(prompt)
        symbol = response.content.strip().upper()
        
        if re.match(r"^[A-Z]{1,5}$", symbol):
            state_dict["symbol"] = symbol
            state_dict["error"] = None
        else:
            state_dict["error"] = "Invalid symbol detected"
        
    except Exception as e:
        state_dict["error"] = f"Query processing failed: {str(e)}"
    
    return FinancialState(**state_dict)