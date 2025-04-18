
import os  # Import the os module for operating system functionalities
import json  # Import the json module for handling JSON data
import streamlit as st  # Import Streamlit for building the web app
from google_auth_oauthlib.flow import Flow  # Import Flow for OAuth 2.0 authorization
from googleapiclient.discovery import build  # Import build to create a service for Google APIs
from transformers import pipeline  # Import pipeline for using Hugging Face models
from google.auth.transport.requests import Request  # Import Request for making HTTP requests
import pickle  # Import pickle for serializing and deserializing Python objects

# Allow insecure transport for testing purposes (not recommended for production)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Load credentials from the JSON file containing OAuth 2.0 client secrets
with open('credentials.json') as f:
    credentials_data = json.load(f)  # Parse the JSON file into a Python dictionary

# Extract the redirect URI from the credentials data
redirect_uri = credentials_data['web']['redirect_uris'][0]  # Ensure this is HTTPS

# Define the scopes for Google Calendar API access
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']  # Read-only access to calendar

# Function to authenticate with Google Calendar
def authenticate_web():
    # Create an OAuth 2.0 flow object using the client secrets file and defined scopes
    flow = Flow.from_client_secrets_file(
        'credentials.json',
        scopes=SCOPES,
        redirect_uri=redirect_uri  # Set the redirect URI
    )
    
    # Generate the authorization URL and state parameter
    authorization_url, state = flow.authorization_url(access_type='offline')
    return authorization_url, flow, state  # Return the authorization URL, flow object, and state

# Streamlit app title
st.title("Google Calendar Reader with Hugging Face Transformers")  # Set the title of the app

# Print the redirect URI for reference
st.write(f"Redirect URI: {redirect_uri}")

# Step 1: Authenticate with Google
if 'credentials' not in st.session_state:  # Check if credentials are not already stored in session state
    if 'code' in st.query_params:  # Check if the authorization code is present in the query parameters
        flow = authenticate_web()[1]  # Get the flow object from the authentication function
        st.write(f"Query Params: {st.query_params}")  # Log the query parameters for debugging

        # Log the received state from the query parameters
        received_state = st.query_params.get('state')
        st.write(f"Received State: {received_state}")

        # Log the stored state for comparison
        stored_state = st.session_state.get('state')
        st.write(f"Stored State: {stored_state}")

        # Fetch the token using the authorization code from the query parameters
        try:
            # Ensure the stored state matches the received state for validation
            if stored_state and stored_state == received_state:
                flow.fetch_token(authorization_response=st.query_params['code'])  # Exchange code for token
                creds = flow.credentials  # Get the credentials from the flow
                st.session_state.credentials = creds  # Store the credentials in session state
                
                # Save the credentials to a file for future use
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)  # Serialize and save the credentials
                st.success("Authenticated successfully!")  # Display success message
            else:
                st.error("State mismatch! Possible CSRF attack.")  # Display error for state mismatch
        except Exception as e:
            st.error(f"Error during authentication: {e}")  # Display any errors that occur during authentication
    else:
        # Clear previous state if it exists
        if 'state' in st.session_state:
            del st.session_state['state']  # Remove the state from session state

        # Check if the token file exists
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)  # Load the credentials from the file
            if creds and creds.valid:  # Check if the credentials are valid
                st.session_state.credentials = creds  # Store the valid credentials in session state
            elif creds and creds.expired and creds.refresh_token:  # Check if the credentials are expired but can be refreshed
                creds.refresh(Request())  # Refresh the credentials
                st.session_state.credentials = creds  # Store the refreshed credentials
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)  # Save the refreshed credentials to the file
            else:
                # Generate the authorization URL and store the state
                authorization_url, flow, state = authenticate_web()  # Get the authorization URL and flow
                st.session_state.state = state  # Store the state in session state
                st.write(f"[Click here to authorize]({authorization_url})")  # Provide a link to authorize
        else:
            # Generate the authorization URL and store the state
            authorization_url, flow, state = authenticate_web()  # Get the authorization URL and flow
            st.session_state.state = state  # Store the state in session state
            st.write(f"[Click here to authorize]({authorization_url})")  # Provide a link to authorize
else:
    creds = st.session_state.credentials  # Retrieve the stored credentials from session state
    service = build('calendar', 'v3', credentials=creds)  # Build the Google Calendar service

    # Step 2: Read events from Google Calendar
    events_result = service.events().list(calendarId='primary', maxResults=10, singleEvents=True,
                                          orderBy='startTime').execute()  # Fetch upcoming events
    events = events_result.get('items', [])  # Get the list of events

    if not events:  # Check if there are no upcoming events
        st.write('No upcoming events found.')  # Display message if no events are found
    else:
        st.write("Upcoming events:")  # Display header for upcoming events
        for event in events:  # Iterate through the list of events
            start = event['start'].get('dateTime', event['start'].get('date'))  # Get the start time of the event
            st.write(f"{start}: {event['summary']}")  # Display the event summary and start time

        # Step 3: Process events with Hugging Face Transformers
        st.write("Processing events with Hugging Face Transformers...")  # Display processing message
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Load the summarization model
        event_summaries = "\n".join([f"{start}: {event['summary']}" for event in events])  # Create a summary of events
        prompt = f"Here are my upcoming events:\n{event_summaries}\nCan you summarize these events for me?"  # Create a prompt for summarization
        response = summarizer(prompt, max_length=50, min_length=25, do_sample=False)  # Get the summary from the model
        st.write("Model Response:")  # Display header for model response
        st.write(response[0]['summary_text'])  # Display the summarized text from the model