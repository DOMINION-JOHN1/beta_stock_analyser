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

def visualization_node(state: FinancialState) -> FinancialState:
    if state.error or not state.processed_data:
        return state
    
    state_dict = state.model_dump()
    
    try:
        # Get historical data
        stock_data = state.processed_data["stock"]
        historical_dates = stock_data.index
        
        # 1. Main Price Chart with Forecast
        # --------------------------------
        # Infer frequency from historical data
        freq = pd.infer_freq(historical_dates) or 'D'  # Default to daily
        
        # Generate forecast dates using frequency-aware offset
        forecast_dates = [
            historical_dates[-1] + (i * pd.tseries.frequencies.to_offset(freq)) 
            for i in range(1, 8)
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=stock_data["Close"],
            name="Historical",
            line=dict(color='blue', width=2))
        )
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=state.predictions,
            name="Forecast",
            line=dict(color='green', dash='dot', width=2))
        )
        fig.update_layout(
            title=f"{state.symbol} Price Analysis",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis=dict(
                range=[historical_dates[0], forecast_dates[-1]],
                tickformat="%b %Y",
                dtick="M1",
                tickangle=45
            )
        )

        # 2. Anomaly Detection Visualization
        # ----------------------------------
        anomaly_fig = go.Figure()
        anomaly_fig.add_trace(go.Scatter(
            x=historical_dates,
            y=stock_data["Close"],
            name="Prices",
            line=dict(color='blue', width=1))
        )
        
        if state.anomalies:
            # Get indices of detected anomalies
            anomaly_indices = np.where(np.array(state.anomalies) == -1)[0]
            anomaly_dates = historical_dates[anomaly_indices]
            anomaly_prices = stock_data["Close"].iloc[anomaly_indices]
            
            anomaly_fig.add_trace(go.Scatter(
                x=anomaly_dates,
                y=anomaly_prices,
                mode='markers',
                name='Anomalies',
                marker=dict(
                    color='red',
                    size=8,
                    line=dict(width=1, color='DarkSlateGrey'))
            ))
        
        anomaly_fig.update_layout(
            title="Anomaly Detection",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis=dict(
                tickformat="%b %Y",
                dtick="M1",
                tickangle=45
            )
        )

        # Save both visualizations
        state_dict["visualizations"] = [
            fig.to_html(full_html=False),
            anomaly_fig.to_html(full_html=False)
        ]
        
    except Exception as e:
        state_dict["error"] = f"Visualization failed: {str(e)}"
    
    return FinancialState(**state_dict)