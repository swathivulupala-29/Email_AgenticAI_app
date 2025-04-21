import os
import streamlit as st
import requests
import time
import datetime
import pickle
from langgraph.graph import StateGraph, END
from typing import TypedDict
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials

# Load secrets
API_TOKEN = st.secrets["general"]["HUGGINGFACE_API_KEY"]
NEWS_API_KEY = st.secrets["general"]["NEWS_API_KEY"]
APP_URL = st.secrets["general"]["STREAMLIT_APP_URL"]
ENV = st.secrets["general"].get("STREAMLIT_ENV", "development")
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

# Debug info
st.write("🔧 Environment:", ENV)
st.write("🔧 App URL:", APP_URL)

class SummaryState(TypedDict):
    text: str
    summary: str

def summarize_text(state: SummaryState) -> SummaryState:
    text = state["text"]
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {"inputs": text}
    for attempt in range(3):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            summary = response.json()[0]["summary_text"]
            return {"text": text, "summary": summary}
        elif response.status_code == 503:
            st.warning(f"Service temporarily unavailable (503). Retry {attempt+1}/3")
            time.sleep(2)
        else:
            st.error(f"Error summarizing: {response.status_code}, {response.text}")
            return {"text": text, "summary": response.text}
    st.error("Summarization service unavailable after multiple attempts.")
    return {"text": text, "summary": "Summarization failed."}

def create_langgraph_pipeline():
    builder = StateGraph(SummaryState)
    builder.add_node("summarize", summarize_text)
    builder.set_entry_point("summarize")
    builder.set_finish_point("summarize")
    return builder.compile()

def get_google_auth_url():
    redirect_uri = APP_URL if ENV == "production" else "http://localhost:8501/"
    st.write("🔗 Redirect URI:", redirect_uri)
    if not os.path.exists("credentials.json"):
        st.error("Missing credentials.json")
        return None
    try:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        flow.redirect_uri = redirect_uri
        auth_url, _ = flow.authorization_url(access_type="offline", prompt='consent', include_granted_scopes="true")
        return auth_url
    except Exception as e:
        st.error(f"Error creating OAuth flow: {e}")
        return None

def get_calendar_service(credentials):
    try:
        return build("calendar", "v3", credentials=credentials)
    except Exception as e:
        st.error(f"Error building service: {e}")
        raise

def get_google_calendar_events(credentials):
    try:
        service = get_calendar_service(credentials)
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        st.write(f"📅 Fetching events from: {now}")
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now,
            maxResults=10,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])
        st.write(f"Found {len(events)} events.")
        if not events:
            return "No upcoming events found."
        events_str = "Upcoming events:\n"
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            events_str += f"- {event.get('summary', 'No Title')} ({start})\n"
        return events_str
    except RefreshError:
        st.error("🔒 Token expired. Please re-authorize.")
        if os.path.exists("token.pickle"):
            os.remove("token.pickle")
            st.info("Removed expired token.")
        if 'credentials' in st.session_state:
            del st.session_state['credentials']
        raise
    except Exception as e:
        st.error(f"❌ Failed to fetch events: {e}")
        raise

# UI Start
st.title("📅 Calendar + 🗞️ News Summarizer with LangGraph")

# Clear credentials if a new user accesses the app
if 'credentials' in st.session_state and st.session_state.credentials:
    st.session_state.credentials = None  # Reset credentials for new users

if 'credentials' not in st.session_state:
    st.session_state.credentials = None

# Load token if available
if not st.session_state.credentials and os.path.exists("token.pickle"):
    try:
        with open("token.pickle", "rb") as f:
            st.session_state.credentials = pickle.load(f)
        st.success("🔑 Credentials loaded.")
    except Exception as e:
        st.warning(f"Token load failed: {e}")
        os.remove("token.pickle")

# Step 1: Authorization
if not st.session_state.credentials:
    st.subheader("1. Authorize Google Calendar")
    if st.button("🔐 Authorize"):
        auth_url = get_google_auth_url()
        if auth_url:
            st.markdown(f"👉 [Click here to authorize]({auth_url})")
            st.info("Return here after authorization.")

# Step 2: Handle OAuth Redirect
query_params = st.query_params
code = query_params.get("code")

if code and not st.session_state.credentials:
    st.write("🔑 Received authorization code. Fetching token...")
    try:
        redirect_uri = APP_URL if ENV == "production" else "http://localhost:8501/"
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        flow.redirect_uri = redirect_uri
        flow.fetch_token(code=code)
        credentials = flow.credentials
        st.session_state.credentials = credentials
        with open("token.pickle", "wb") as f:
            pickle.dump(credentials, f)
        st.success("✅ Authorization successful!")
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Token fetch error: {e}")
        if 'credentials' in st.session_state:
            del st.session_state['credentials']

# Step 3: Calendar + Summary
if st.session_state.credentials:
    st.subheader("📅 Google Calendar Events")
    try:
        events_str = get_google_calendar_events(st.session_state.credentials)
        st.text_area("Events:", events_str, height=200, key="calendar_display")

        if events_str != "No upcoming events found.":
            st.info("Automatically summarizing your events...")
            with st.spinner("Summarizing..."):
                langgraph = create_langgraph_pipeline()
                result = langgraph.invoke({"text": events_str, "summary": ""})
                summary = result.get("summary", "No summary returned.")
                st.subheader("📋 Summary:")
                st.write(summary)
        else:
            st.info("No events to summarize.")
    except Exception as e:
        st.error(f"Could not show events. {e}")
        if st.button("🔄 Clear Credentials"):
            if os.path.exists("token.pickle"):
                os.remove("token.pickle")
            if 'credentials' in st.session_state:
                del st.session_state['credentials']
            st.query_params.clear()
            st.rerun()

# Step 4: Auto Load & Summarize Top USA News
st.subheader("🗞️ Today's Top USA News")

def get_top_usa_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=10&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        if not articles:
            return "No top news found today."
        combined_text = "\n".join(f"- {a['title']} ({a['source']['name']})" for a in articles if 'title' in a)
        return combined_text
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        return ""

# Run once per session
if 'news_summary' not in st.session_state:
    with st.spinner("Fetching and summarizing top news..."):
        news_text = get_top_usa_news()
        if news_text and news_text != "No top news found today.":
            langgraph = create_langgraph_pipeline()
            result = langgraph.invoke({"text": news_text, "summary": ""})
            st.session_state["news_text"] = news_text
            st.session_state["news_summary"] = result.get("summary", "No summary returned.")
        else:
            st.session_state["news_text"] = ""
            st.session_state["news_summary"] = "No news found to summarize."

# Display news + summary
st.text_area("Top Headlines:", st.session_state.get("news_text", ""), height=200)
st.subheader("📝 Summary:")
st.write(st.session_state.get("news_summary", "No summary returned."))