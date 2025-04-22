import os
import json
import pickle
import requests
import streamlit as st
from datetime import datetime, timedelta
from urllib.parse import urlencode
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from transformers import pipeline
from langgraph import Graph, Node  # Import Graph and Node from langgraph
from typing import List

# Set up environment
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Load credentials
with open('credentials.json') as f:
    credentials_data = json.load(f)

redirect_uri = credentials_data['web']['redirect_uris'][0]
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
NEWS_API_KEY = credentials_data.get('NEWS_API_KEY', '')
WEATHER_API_KEY = credentials_data.get('WEATHER_API_KEY', '')

# Initialize Streamlit
st.title("🧠 Agentic AI Daily Assistant")

# Authenticate user
def authenticate_web():
    flow = Flow.from_client_secrets_file(
        'credentials.json',
        scopes=SCOPES,
        redirect_uri=redirect_uri
    )
    authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true')
    return authorization_url, flow, state

# Handle OAuth callback
if 'code' in st.query_params and 'credentials' not in st.session_state:
    received_state = st.query_params.get('state')
    if os.path.exists('state_temp.json'):
        with open('state_temp.json', 'r') as f:
            stored_state = json.load(f).get('state')
        os.remove('state_temp.json')
    else:
        stored_state = None

    if stored_state == received_state:
        flow = Flow.from_client_secrets_file(
            'credentials.json',
            scopes=SCOPES,
            redirect_uri=redirect_uri
        )
        query_dict = st.query_params
        query_string = urlencode(query_dict, doseq=True)
        authorization_response = f"{redirect_uri}?{query_string}"

        try:
            flow.fetch_token(authorization_response=authorization_response)
            creds = flow.credentials
            st.session_state.credentials = creds

            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        except Exception as e:
            st.error(f"Authentication failed: {e}")
    else:
        st.error("State mismatch! Possible CSRF attack.")

# Load saved credentials
if 'credentials' not in st.session_state:
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

        if creds and creds.valid:
            st.session_state.credentials = creds
        elif creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            st.session_state.credentials = creds
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

# Prompt login
if 'credentials' not in st.session_state:
    authorization_url, flow, state = authenticate_web()
    with open('state_temp.json', 'w') as f:
        json.dump({'state': state}, f)

    st.markdown(
        f"""
        <div style="text-align:center;">
            <a href="{authorization_url}" target="_self">
                <button style="padding: 0.75em 1.5em; font-size: 1rem; background-color: #0b8043; color: white; border: none; border-radius: 5px; cursor: pointer;">
                    Click here to authorize Google Calendar
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.info("Please authorize access to your Google Calendar.")
    st.stop()

# Fetch calendar events
creds = st.session_state.credentials
service = build('calendar', 'v3', credentials=creds)

now = datetime.utcnow()
today_start = datetime(now.year, now.month, now.day, tzinfo=datetime.now().astimezone().tzinfo)
tomorrow_end = today_start + timedelta(days=2)

events_result = service.events().list(
    calendarId='primary',
    timeMin=today_start.isoformat(),
    timeMax=tomorrow_end.isoformat(),
    singleEvents=True,
    orderBy='startTime'
).execute()

events = events_result.get('items', [])

st.subheader("Today's & Tomorrow's Events")
event_texts = []
for event in events:
    try:
        start = event['start'].get('dateTime', event['start'].get('date'))
        summary = event.get('summary', 'No Title')
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if today_start <= start_dt <= tomorrow_end:
            st.write(f"• {start_dt.strftime('%Y-%m-%d %H:%M')} — {summary}")
            event_texts.append(f"{start_dt.strftime('%Y-%m-%d %H:%M')} — {summary}")
    except Exception as e:
        st.warning(f"Unexpected error with event: {e}. Event: {event}")

# Summarization function using LangGraph
def summarize_with_langgraph(texts: List[str]) -> str:
    if not texts:
        return "No data to summarize."

    # Create a LangGraph
    graph = Graph()

    # Define a summarization node
    @node
    def summarizer(text: str) -> str:
        summarizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        summary = summarizer_pipeline(text, max_length=80, min_length=40, do_sample=False)
        return summary[0]['summary_text']

    # Add the summarizer node to the graph
    graph.add_node(summarizer)

    # Combine texts for summarization
    combined_text = " ".join(texts)
    summary = graph.run(summarizer, combined_text)  # Run the graph with the combined text
    return summary

# Show LangGraph summary of events
if event_texts:
    st.subheader("Calendar Summary")
    event_summary = summarize_with_langgraph(event_texts)
    st.write(event_summary)

# Show top US news
st.subheader("Today's Top News")
news_url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
response = requests.get(news_url)
news_data = response.json()

articles = news_data.get('articles', [])[:5]
news_texts = []

if not articles:
    st.warning("No news articles found.")
else:
    for article in articles:
        st.markdown(f"**• {article['title']}**")
        news_texts.append(article['title'] + ". " + (article.get('description') or ''))

# LangGraph workflow for news summary
if news_texts:
    st.subheader("News Summary")
    news_summary = summarize_with_langgraph(news_texts)
    st.write(news_summary)

# Weather input after calendar and news
st.subheader("🌤️ Weather Forecast")
cities = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "Indianapolis", "Charlotte", "San Francisco", "Seattle", "Denver", "Washington D.C.",
    "Boston", "El Paso", "Detroit", "Nashville", "Portland", "Memphis", "Oklahoma City", "Las Vegas", "Louisville", "Baltimore",
    "Milwaukee", "Albuquerque", "Tucson", "Fresno", "Mesa", "Sacramento", "Kansas City", "Long Beach", "Atlanta", "Raleigh",
    "Miami", "Omaha", "Cleveland", "Tulsa", "Minneapolis", "Arlington", "New Orleans", "Wichita", "Bakersfield", "Cincinnati"
]
selected_city = st.selectbox("Select a city to see the current weather:", cities)

if selected_city:
    weather_url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={selected_city}"
    weather_response = requests.get(weather_url)

    if weather_response.status_code == 200:
        weather_data = weather_response.json()
        condition = weather_data['current']['condition']['text']
        temp_c = weather_data['current']['temp_c']
        feels_like = weather_data['current']['feelslike_c']
        humidity = weather_data['current']['humidity']

        st.markdown(f"**City:** {selected_city}")
        st.markdown(f"**Condition:** {condition}")
        st.markdown(f"**Temperature:** {temp_c}°C (Feels like {feels_like}°C)")
        st.markdown(f"**Humidity:** {humidity}%")
    else:
        st.error(f"Could not retrieve weather data for {selected_city}")
