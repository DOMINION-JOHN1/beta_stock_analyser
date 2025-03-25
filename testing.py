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

# Fix Unicode handling system-wide
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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

def preprocessing_node(state: FinancialState) -> FinancialState:
    """Clean and prepare data"""
    if state.error or not state.raw_data:
        return state
    
    state_dict = state.model_dump()
    
    try:
        # Process stock data
        stock_series = pd.DataFrame(
            state.raw_data["stock"]["Time Series (Daily)"]
        ).T.rename(columns={
            "1. open": "Open",
            "4. close": "Close"
        })
        stock_series.index = pd.to_datetime(stock_series.index)
        stock_series = stock_series[["Open", "Close"]].astype(float)
        
        # Process news with Unicode handling
        news_content = "\n".join([
            f"{article['title']}: {article['description']}"
            for article in state.raw_data["news"]["articles"]
            if article.get('description')
        ]).encode('utf-8', 'ignore').decode('utf-8')
        
        state_dict["processed_data"] = {
            "stock": stock_series,
            "news": news_content[:5000]  # Limit context length
        }
        state_dict["error"] = None
    
    except Exception as e:
        state_dict["error"] = f"Preprocessing failed: {str(e)}"
    
    return FinancialState(**state_dict)

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

def anomaly_detection_node(state: FinancialState) -> FinancialState:
    """Detect market anomalies"""
    if state.error or not state.processed_data:
        return state
    
    state_dict = state.model_dump()
    
    try:
        clf = IsolationForest(contamination=0.1)
        prices = state.processed_data["stock"]["Close"].values.reshape(-1,1)
        state_dict["anomalies"] = clf.fit_predict(prices).tolist()
        state_dict["error"] = None
    
    except Exception as e:
        state_dict["error"] = f"Anomaly detection failed: {str(e)}"
    
    return FinancialState(**state_dict)

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

def visualization_node(state: FinancialState) -> FinancialState:
    """Generate visualizations"""
    if state.error or not state.processed_data:
        return state
    
    state_dict = state.model_dump()
    
    try:
        # Price chart with Unicode font
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=state.processed_data["stock"].index,
            y=state.processed_data["stock"]["Close"],
            name="Historical"
        ))
        fig.add_trace(go.Scatter(
            x=pd.date_range(start=state.processed_data["stock"].index[-1], periods=7),
            y=state.predictions,
            name="Forecast"
        ))
        fig.update_layout(
            font=dict(family="Arial Unicode MS", size=12),
            title=f"{state.symbol} Price Analysis"
        )
        
        # Anomaly chart
        anomaly_fig = go.Figure()
        anomaly_fig.add_trace(go.Scatter(
            x=state.processed_data["stock"].index,
            y=state.processed_data["stock"]["Close"],
            mode='lines',
            name='Prices'
        ))
        if state.anomalies:
            anomaly_fig.add_trace(go.Scatter(
                x=state.processed_data["stock"].index[np.array(state.anomalies) == -1],
                y=state.processed_data["stock"]["Close"][np.array(state.anomalies) == -1],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8)
            ))
        anomaly_fig.update_layout(
            font=dict(family="Arial Unicode MS", size=12),
            title="Anomaly Detection"
        )
        
        state_dict["visualizations"] = [
            fig.to_html(full_html=False),
            anomaly_fig.to_html(full_html=False)
        ]
        state_dict["error"] = None
    
    except Exception as e:
        state_dict["error"] = f"Visualization failed: {str(e)}"
    
    return FinancialState(**state_dict)

def report_node(state: FinancialState) -> FinancialState:
    """Generate final report"""
    if state.error:
        return state
    
    state_dict = state.model_dump()
    
    try:
        report = f"""
        <html>
            <head>
                <meta charset="UTF-8">
                <title>{state.symbol} Analysis Report</title>
                <style>
                    body {{ font-family: Arial Unicode MS, sans-serif; margin: 2rem; }}
                    .insights {{ white-space: pre-wrap; line-height: 1.6; }}
                    .visualization {{ margin: 2rem 0; border: 1px solid #ddd; padding: 1rem; }}
                </style>
            </head>
            <body>
                <h1>{state.symbol} Financial Analysis</h1>
                <div class="insights">{state.insights}</div>
                {"".join(f'<div class="visualization">{viz}</div>' for viz in state.visualizations)}
            </body>
        </html>
        """
        
        with open(f"{state.symbol}_report.html", "w", encoding="utf-8") as f:
            f.write(report)
        
        state_dict["error"] = None
    
    except Exception as e:
        state_dict["error"] = f"Report generation failed: {str(e)}"
    
    return FinancialState(**state_dict)

# Build and compile workflow
workflow = StateGraph(FinancialState)

nodes = [
    process_query_node,
    data_ingestion_node,
    preprocessing_node,
    model_training_node,
    prediction_node,
    anomaly_detection_node,
    insight_generation_node,
    visualization_node,
    report_node
]

for node in nodes:
    workflow.add_node(node.__name__, node)

workflow.set_entry_point("process_query_node")
workflow.add_edge("process_query_node", "data_ingestion_node")
workflow.add_edge("data_ingestion_node", "preprocessing_node")
workflow.add_edge("preprocessing_node", "model_training_node")
workflow.add_edge("model_training_node", "prediction_node")
workflow.add_edge("prediction_node", "anomaly_detection_node")
workflow.add_edge("anomaly_detection_node", "insight_generation_node")
workflow.add_edge("insight_generation_node", "visualization_node")
workflow.add_edge("visualization_node", "report_node")
workflow.add_edge("report_node", END)

chain = workflow.compile()

# Main execution with proper error handling
if __name__ == "__main__":
    queries = [
        "What's the outlook for Microsoft stock?",
        "Analyze NVDA technicals",
        "Should I invest in Amazon?",
        "Update me on TSLA"
    ]
    
    for query in queries:
        print(f"\nProcessing: {query}")
        try:
            result = chain.invoke(FinancialState(user_query=query))
            result = FinancialState(**result)
            
            if result.error:
                print(f"Error: {result.error}")
            else:
                print(f"Analysis for {result.symbol} complete")
                print(f"Report saved to {result.symbol}_report.html")
                
        except Exception as e:
            print(f"System error: {str(e)}")