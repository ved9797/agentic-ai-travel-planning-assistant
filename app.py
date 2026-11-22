
import json
import pandas as pd
import requests
import streamlit as st

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

with open("flights.json", "r") as file:
    flights_data = json.load(file)

with open("hotels.json", "r") as file:
    hotels_data = json.load(file)

with open("places.json", "r") as file:
    places_data = json.load(file)


flights_df = pd.DataFrame(flights_data)
hotels_df = pd.DataFrame(hotels_data)
places_df = pd.DataFrame(places_data)


# ---------------------------------------------------------
# City coordinates
# ---------------------------------------------------------

city_coordinates = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873),
    "Kolkata": (22.5726, 88.3639),
    "Goa": (15.2993, 74.1240)
}


# ---------------------------------------------------------
# Flight Search Tool
# ---------------------------------------------------------

@tool
def flight_search_tool(origin, destination):
    """Search for flights between two cities."""

    results = flights_df[
        (flights_df["from"].str.lower() == origin.lower()) &
        (flights_df["to"].str.lower() == destination.lower())
    ]

    results = results.sort_values("price")

    return results.to_dict(orient="records")


# ---------------------------------------------------------
# Hotel Recommendation Tool
# ---------------------------------------------------------

@tool
def hotel_search_tool(city, min_stars=0):
    """Search for hotels in a city."""

    min_stars = float(min_stars)

    results = hotels_df[
        hotels_df["city"].str.lower() == city.lower()
    ]

    results = results[
        results["stars"] >= min_stars
    ]

    results = results.sort_values(
        "price_per_night"
    )

    return results.to_dict(orient="records")


# ---------------------------------------------------------
# Places Discovery Tool
# ---------------------------------------------------------

@tool
def places_search_tool(
    city,
    place_type="",
    min_rating=0
):
    """Search for tourist places in a city."""

    min_rating = float(min_rating)

    if place_type == "":
        place_type = None

    results = places_df[
        places_df["city"].str.lower() == city.lower()
    ]

    if place_type:
        results = results[
            results["type"].str.lower() ==
            place_type.lower()
        ]

    results = results[
        results["rating"] >= min_rating
    ]

    results = results.sort_values(
        "rating",
        ascending=False
    )

    return results.to_dict(orient="records")


# ---------------------------------------------------------
# Weather Tool
# ---------------------------------------------------------

@tool
def city_weather_tool(
    city,
    forecast_days=7
):
    """Get the weather forecast for a city."""

    coordinates = city_coordinates.get(
        city.title()
    )

    if coordinates is None:
        return {
            "error": "Weather data is unavailable for this city."
        }

    latitude, longitude = coordinates

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "weather_code"
        ],
        "forecast_days": forecast_days,
        "timezone": "auto"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as error:
        return {
            "error": str(error)
        }


# ---------------------------------------------------------
# Budget Estimation Tool
# ---------------------------------------------------------

@tool
def budget_estimation_tool(
    flight_price,
    hotel_price_per_night,
    number_of_nights,
    daily_activity_budget
):
    """Estimate the total trip budget."""

    flight_price = float(flight_price)
    hotel_price_per_night = float(
        hotel_price_per_night
    )
    number_of_nights = int(number_of_nights)
    daily_activity_budget = float(
        daily_activity_budget
    )

    hotel_cost = (
        hotel_price_per_night *
        number_of_nights
    )

    activity_cost = (
        daily_activity_budget *
        number_of_nights
    )

    total_cost = (
        flight_price +
        hotel_cost +
        activity_cost
    )

    return {
        "flight_cost": flight_price,
        "hotel_cost": hotel_cost,
        "activity_cost": activity_cost,
        "total_cost": total_cost
    }


# ---------------------------------------------------------
# Gemini LLM
# ---------------------------------------------------------

google_api_key = st.secrets["Google-API-Key"]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=google_api_key
)


# ---------------------------------------------------------
# LangChain Agent
# ---------------------------------------------------------

tools = [
    flight_search_tool,
    hotel_search_tool,
    places_search_tool,
    city_weather_tool,
    budget_estimation_tool
]

travel_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are an intelligent travel planning assistant.

Use the available travel tools to create personalized
travel plans.

Always use the provided datasets and weather API.

Never invent flights, hotels, tourist places, prices,
ratings, or weather information.

For an N-day trip, use N-1 hotel nights unless the
user specifies otherwise.

Use the appropriate tool for:
- Flights
- Hotels
- Tourist places
- Weather
- Budget estimation

If information is unavailable, clearly say so.

Return the travel plan using:
1. Trip Summary
2. Flight Recommendation
3. Hotel Recommendation
4. Day-wise Itinerary
5. Weather Forecast
6. Budget Breakdown

Keep the response clear, practical, and concise.
"""
)


# ---------------------------------------------------------
# Streamlit Interface
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Travel Planning Assistant",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Agentic AI Travel Planning Assistant")

st.write(
    "Create personalized travel plans using flights, "
    "hotels, tourist places, weather, and budget information."
)

st.divider()


# User inputs
origin = st.selectbox(
    "Departure City",
    sorted(flights_df["from"].unique())
)

destination = st.selectbox(
    "Destination City",
    sorted(flights_df["to"].unique())
)

number_of_days = st.number_input(
    "Number of Days",
    min_value=2,
    max_value=7,
    value=3
)

min_hotel_stars = st.slider(
    "Minimum Hotel Star Rating",
    min_value=0,
    max_value=5,
    value=3
)

daily_activity_budget = st.number_input(
    "Daily Activity Budget (₹)",
    min_value=0,
    value=1500,
    step=500
)


# Generate travel plan
if st.button("Generate Travel Plan"):

    query = f"""
    Plan a {number_of_days}-day trip from
    {origin} to {destination}.

    Find the cheapest flight.

    Recommend a hotel with at least
    {min_hotel_stars} stars.

    Suggest tourist places from the available dataset.

    Check the weather.

    Estimate the total budget using a daily
    activity budget of ₹{daily_activity_budget}.
    """

    with st.spinner(
        "Creating your personalized travel plan..."
    ):

        response = travel_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        })

    final_response = response["messages"][-1].content

    if isinstance(final_response, list):
       final_response = "".join(
          item["text"]
          for item in final_response
          if isinstance(item, dict) and "text" in item
       )

    st.markdown(final_response)
