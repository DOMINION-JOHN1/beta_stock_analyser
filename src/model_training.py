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

def model_training_node(state: FinancialState) -> FinancialState:
    """Train forecasting model"""
    if state.error or not state.processed_data:
        return state
    
    state_dict = state.model_dump()
    
    try:
        model = auto_arima(
            state.processed_data["stock"]["Close"],
            seasonal=False,
            trace=True
        )
        state_dict["model"] = model
        state_dict["error"] = None
    
    except Exception as e:
        state_dict["error"] = f"Model training failed: {str(e)}"
    
    return FinancialState(**state_dict)