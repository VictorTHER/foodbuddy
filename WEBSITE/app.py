import streamlit as st
import requests
from streamlit_folium import st_folium
import folium
from PIL import Image
import toml

# Load the .toml file
config = toml.load(".secrets.toml")

# Extract Mapbox details
MAPBOX_ACCESS_TOKEN = config["mapbox"]["MAPBOX_ACCESS_TOKEN"]
GEOCODING_API_URL = config["mapbox"]["GEOCODING_API_URL"]
DIRECTIONS_API_URL = config["mapbox"]["DIRECTIONS_API_URL"]
MODEL_API_URL = config["model"]["MODEL_API_URL"]


st.title("NYC taxi Fare Calculator")

image = Image.open("/Users/pato/code/Paulltho/taxifare-website/Taxi_picture")
st.image(image, use_container_width=True)

### HEADER 1: FILL IN FORM###
st.header("Your trip details")

### SUBHEADER 1.1: DETAIL FILL IN ###
st.markdown("_Please fill in your trip details._")

timestamp = st.text_input("Date and Time")
pickup_location = st.text_input("Pickup Location")
dropoff_location = st.text_input("Dropoff Location")
passenger_count = st.selectbox("Passenger Count", [1,2,3,4,5,6,7,8])

timestamp =
pickup_location =
dropoff_location =
passenger_count = 

2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2

### SUBHEADER 1.1 TECHNICAL: LAT/LON COMPUTATION ###

def get_coordinates(address):
    params = {
        "access_token": MAPBOX_ACCESS_TOKEN,
        "limit": 1,
    }
    response = requests.get(f"{GEOCODING_API_URL}{address}.json", params=params)
    data = response.json()
    if "features" in data and len(data["features"]) > 0:
        return data["features"][0]["center"]  # [longitude, latitude]
    return None

if st.button("Calculate fare"):
    pickup_coords = get_coordinates(pickup_location)
    dropoff_coords = get_coordinates(dropoff_location)

    if pickup_coords and dropoff_coords:
        # Fetch route using the Directions API
        route_url = f"{DIRECTIONS_API_URL}{pickup_coords[0]},{pickup_coords[1]};{dropoff_coords[0]},{dropoff_coords[1]}"
        params = {
            "geometries": "geojson",
            "access_token": MAPBOX_ACCESS_TOKEN,
        }
        response = requests.get(route_url, params=params)
        route_data = response.json()

        # Extract the route geometry
        if "routes" in route_data and len(route_data["routes"]) > 0:
            route_geometry = route_data["routes"][0]["geometry"]["coordinates"]

            # Create a folium map
            m = folium.Map(location=[pickup_coords[1], pickup_coords[0]], zoom_start=12)

            # Add the route line to the map
            folium.PolyLine(
                locations=[(coord[1], coord[0]) for coord in route_geometry],
                color="blue",
                weight=5,
                opacity=0.8,
            ).add_to(m)

            # Add markers for pickup and dropoff
            folium.Marker(
                location=[pickup_coords[1], pickup_coords[0]], tooltip="Pickup Location"
            ).add_to(m)
            folium.Marker(
                location=[dropoff_coords[1], dropoff_coords[0]], tooltip="Dropoff Location"
            ).add_to(m)

            # Display the map in Streamlit
            st_folium(m, width=700, height=500)

            # Run the model
            params = {
            "pickup_datetime": timestamp,
            "pickup_longitude": pickup_coords[0],
            "pickup_latitude": pickup_coords[1],
            "dropoff_longitude": dropoff_coords[0],
            "dropoff_latitude": dropoff_coords[1],
            "passenger_count": passenger_count,
            }

            try:
                # Make a GET request to the model API
                response = requests.get(MODEL_API_URL, params=params)
                response_data = response.json()

                # Display the fare prediction
                if response.status_code == 200 and "fare" in response_data:
                    fare = response_data["fare"]
                    st.success(f"Predicted Fare: ${fare:.2f}")
                else:
                    st.error("Failed to retrieve a valid fare prediction. Check your inputs or API.")
            except Exception as e:
                st.error(f"Error communicating with the API: {e}")

        else:
            st.error("Could not retrieve the route. Please try again.")
    else:
        st.error("Could not retrieve coordinates for the locations. Please check the input.")
