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

def insight_generation_node(state: FinancialState) -> FinancialState:
    """Generate financial insights"""
    if state.error or not all([state.processed_data, state.predictions]):
        return state
    
    state_dict = state.model_dump()
    
    try:
        prompt_template = ChatPromptTemplate.from_template("""
        As a senior financial analyst, analyze this data for {symbol}:
        Latest Prices: {prices}
        7-Day Forecast: {forecast}
        Relevant News: {news}
        Provide professional investment analysis with recommendations.""")
        
        response = llm.invoke(prompt_template.format(
            symbol=state.symbol,
            prices=state.processed_data["stock"]["Close"].tail(5).to_string(),
            forecast="\n".join([f"Day {i+1}: {price:.2f}" for i, price in enumerate(state.predictions)]),
            news=state.processed_data["news"]
        ))
        
        state_dict["insights"] = response.content
        state_dict["error"] = None
    
    except Exception as e:
        state_dict["error"] = f"Insight generation failed: {str(e)}"
    
    return FinancialState(**state_dict)
