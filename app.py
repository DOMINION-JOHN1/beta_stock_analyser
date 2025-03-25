import requests
import json
import os

#stock_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey={os.environ['ALPHA_VANTAGE']}"
#stock_data = requests.get(stock_url).json()
#print(stock_data)
#news_url = f"https://newsapi.org/v2/everything?q=finance&apiKey={os.environ['NEWS_API_KEY']}"
#news_data = requests.get(news_url).json()
#print(news_data)

import os
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
from dotenv import load_dotenv
load_dotenv()


# Initialize clients
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.1,
    max_tokens=1024,
    api_key=os.environ["GROQ_API_KEY"]
)

# ======================
# LangGraph State Setup
# ======================
initial_state = {
    "symbol": "IBM",  # Default symbol
    "raw_data": None,
    "processed_data": None,
    "model": None,
    "predictions": None,
    "anomalies": None,
    "insights": None,
    "visualizations": [],
    "feedback": None,
    "user_query": None
}

# ======================
# Core Components
# ======================

def data_ingestion_node(state):
    """Fetch data from Alpha Vantage and NewsAPI"""
    print(f"Fetching data for {state['symbol']}...")
    
    try:
        # Stock Data from Alpha Vantage
        stock_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={state['symbol']}&apikey={os.environ['ALPHA_VANTAGE_API']}"
        stock_response = requests.get(stock_url)
        stock_response.raise_for_status()
        
        # News from NewsAPI
        news_url = "https://newsapi.org/v2/everything"
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        news_params = {
            'q': f"{state['symbol']} stock",
            'from': week_ago,
            'sortBy': 'publishedAt',
            'apiKey': os.environ['NEWSAPI_KEY'],
            'language': 'en',
            'pageSize': 20
        }
        news_response = requests.get(news_url, params=news_params)
        news_response.raise_for_status()
        
        state["raw_data"] = {
            "stock": stock_response.json(),
            "news": news_response.json()
        }
        
    except requests.exceptions.RequestException as e:
        print(f"API Error: {str(e)}")
        state["raw_data"] = None
        
    return state

def preprocessing_node(state):
    """Clean and prepare data"""
    print("Preprocessing data...")
    
    if not state["raw_data"]:
        return state
    
    try:
        # Process stock data
        stock_series = pd.DataFrame(
            state["raw_data"]["stock"]["Time Series (Daily)"]
        ).T.rename(columns={
            "1. open": "Open",
            "4. close": "Close"
        })
        stock_series.index = pd.to_datetime(stock_series.index)
        stock_series = stock_series[["Open", "Close"]].astype(float)
        
        # Process news
        news_content = "\n".join([
            f"{article['title']}: {article['description']}"
            for article in state["raw_data"]["news"]["articles"]
            if article['description'] and article['title']
        ])
        
        state["processed_data"] = {
            "stock": stock_series,
            "news": news_content[:5000]  # Limit context length
        }
        
    except KeyError as e:
        print(f"Data processing error: {str(e)}")
        state["processed_data"] = None
        
    return state

def model_training_node(state):
    """Auto ARIMA training"""
    if not state["processed_data"]:
        return state
    
    print("Training model...")
    try:
        model = auto_arima(
            state["processed_data"]["stock"]["Close"],
            seasonal=False,
            trace=True
        )
        state["model"] = model
    except ValueError as e:
        print(f"Model training failed: {str(e)}")
        state["model"] = None
        
    return state

def prediction_node(state):
    """Generate forecasts"""
    if not state["model"]:
        return state
    
    print("Making predictions...")
    try:
        forecast = state["model"].predict(n_periods=7)
        state["predictions"] = forecast
    except ValueError as e:
        print(f"Prediction failed: {str(e)}")
        state["predictions"] = None
        
    return state

def anomaly_detection_node(state):
    """Identify market anomalies"""
    if not state["processed_data"]:
        return state
    
    print("Detecting anomalies...")
    try:
        clf = IsolationForest(contamination=0.1)
        prices = state["processed_data"]["stock"]["Close"].values.reshape(-1,1)
        state["anomalies"] = clf.fit_predict(prices)
    except ValueError as e:
        print(f"Anomaly detection failed: {str(e)}")
        state["anomalies"] = None
        
    return state

def insight_generation_node(state):
    """Generate financial insights using Groq/Llama3"""
    if not all([state["processed_data"], state["predictions"]]):
        return state
    
    print("Generating insights...")
    
    prompt_template = ChatPromptTemplate.from_template("""
    As a senior financial analyst, analyze this data for {symbol}:
    
    **Latest Stock Prices**:
    {latest_prices}
    
    **7-Day Forecast**:
    {predictions}
    
    **Recent News Highlights**:
    {news}
    
    Provide professional analysis with:
    1. Investment recommendation (Buy/Hold/Sell)
    2. Technical analysis summary
    3. News impact assessment
    4. Risk level (Low/Medium/High) with justification
    5. Price targets for next week
    """)
    
    try:
        formatted_prompt = prompt_template.format(
            symbol=state["symbol"],
            latest_prices=state["processed_data"]["stock"]["Close"].tail(5).to_string(),
            predictions="\n".join([f"Day {i+1}: {price:.2f}" for i, price in enumerate(state["predictions"])]),
            news=state["processed_data"]["news"]
        )

        response = llm.invoke(formatted_prompt)
        state["insights"] = response.content
    except Exception as e:
        state["insights"] = f"Analysis failed: {str(e)}"
        print(f"Insight generation error: {str(e)}")
    
    return state

def visualization_node(state):
    """Create interactive charts"""
    if not state["processed_data"]:
        return state
    
    print("Generating visualizations...")
    
    try:
        # Main price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=state["processed_data"]["stock"].index,
            y=state["processed_data"]["stock"]["Close"],
            name="Historical"
        ))
        fig.add_trace(go.Scatter(
            x=pd.date_range(start=state["processed_data"]["stock"].index[-1], periods=7),
            y=state["predictions"],
            name="Forecast"
        ))
        state["visualizations"].append(fig.to_html(full_html=False))
        
        # Anomaly detection chart
        if state["anomalies"] is not None:
            anomaly_fig = go.Figure()
            anomaly_fig.add_trace(go.Scatter(
                x=state["processed_data"]["stock"].index,
                y=state["processed_data"]["stock"]["Close"],
                mode='lines',
                name='Prices'
            ))
            anomaly_fig.add_trace(go.Scatter(
                x=state["processed_data"]["stock"].index[state["anomalies"] == -1],
                y=state["processed_data"]["stock"]["Close"][state["anomalies"] == -1],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8)
            ))
            state["visualizations"].append(anomaly_fig.to_html(full_html=False))
            
    except Exception as e:
        print(f"Visualization error: {str(e)}")
    
    return state

def report_node(state):
    """Compile final report"""
    print("Compiling report...")
    
    try:
        report = f"""
        <html>
            <head>
                <title>{state['symbol']} Analysis Report</title>
                <style>
                    body {{ font-family: Arial; margin: 2rem; }}
                    .insights {{ white-space: pre-wrap; line-height: 1.6; }}
                    .visualization {{ margin: 2rem 0; border: 1px solid #ddd; padding: 1rem; }}
                </style>
            </head>
            <body>
                <h1>{state['symbol']} Financial Analysis Report</h1>
                <div class="insights">{state['insights']}</div>
                {"".join([f'<div class="visualization">{viz}</div>' for viz in state['visualizations']])}
                <form action="/feedback">
                    <h3>Analyst Feedback</h3>
                    <textarea name="feedback" rows="4" cols="50"></textarea><br>
                    <button type="submit">Submit Feedback</button>
                </form>
            </body>
        </html>
        """
        
        with open(f"{state['symbol']}_report.html", "w") as f:
            f.write(report)
            
    except Exception as e:
        print(f"Report generation failed: {str(e)}")
    
    return state

# ======================
# Workflow Construction
# ======================
workflow = StateGraph(initial_state)

nodes = [
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

# Build linear workflow
workflow.set_entry_point("data_ingestion_node")
workflow.add_edge("data_ingestion_node", "preprocessing_node")
workflow.add_edge("preprocessing_node", "model_training_node")
workflow.add_edge("model_training_node", "prediction_node")
workflow.add_edge("prediction_node", "anomaly_detection_node")
workflow.add_edge("anomaly_detection_node", "insight_generation_node")
workflow.add_edge("insight_generation_node", "visualization_node")
workflow.add_edge("visualization_node", "report_node")
workflow.add_edge("report_node", END)

# Compile workflow
chain = workflow.compile()

# ======================
# Execution
# ======================
if __name__ == "__main__":
    # Example usage
    final_state = chain.invoke({
        "symbol": "AAPL"  # Change symbol here
    })
    print(f"Report generated at {final_state['symbol']}_report.html")