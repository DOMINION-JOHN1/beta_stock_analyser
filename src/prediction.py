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

def prediction_node(state: FinancialState) -> FinancialState:
    """Generate predictions"""
    if state.error or not state.model:
        return state
    
    state_dict = state.model_dump()
    
    try:
        forecast = state.model.predict(n_periods=7)
        state_dict["predictions"] = forecast.tolist()
        state_dict["error"] = None
    
    except Exception as e:
        state_dict["error"] = f"Prediction failed: {str(e)}"
    
    return FinancialState(**state_dict)