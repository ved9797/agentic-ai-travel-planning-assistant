# Agentic AI-Based Travel Planning Assistant

An intelligent travel planning assistant built using Python, LangChain, Gemini, Open-Meteo, and Streamlit.

## Features

- Flight search and cheapest-flight recommendation
- Hotel recommendation based on star rating
- Tourist place discovery
- Weather forecast using Open-Meteo
- Trip budget estimation
- Agentic AI workflow using LangChain
- Day-wise itinerary generation
- Interactive Streamlit interface

## Technologies

- Python
- Pandas
- LangChain
- Google Gemini
- Open-Meteo API
- Streamlit

## Project Files

- app.py - Streamlit application
- flights.json - Flight dataset
- hotels.json - Hotel dataset
- places.json - Tourist places dataset
- requirements.txt - Required Python packages
- .gitignore - Files excluded from GitHub

## API Key

The Google Gemini API key is not stored in the repository.

For Streamlit Cloud, add the API key as a Streamlit secret using:

Google-API-Key

## Data Note

Flight, hotel, and tourist-place information comes from the provided static project datasets.

Weather information is retrieved dynamically from Open-Meteo.

## Run Locally

Install the required packages:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Project Objective

The objective is to demonstrate how an Agentic AI system can use multiple tools and data sources to automatically create practical and personalized travel plans.
