from flask import Flask, request, jsonify,render_template
import logging
import asyncio
import aiohttp
from queue import PriorityQueue
import math
import geopy.distance
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for weather data to avoid redundant API calls
weather_data_cache = {}

# Load land polygons using geopandas
logger.info("Loading land data...")
land = gpd.read_file('ne_10m_land.shp')
if land.crs is None:
    land = land.set_crs("EPSG:4326")  # Set CRS if missing

# Transform to the desired CRS
land = land.to_crs("EPSG:4326")  # Ensure coordinates are in lat/lon
logger.info("Land data loaded.")

# Build a unified land geometry for efficient intersection checks
unified_land = land.geometry.unary_union  # This is the correct way to access the geometry

# Load grid data from CSV files
def load_grid_data(nodes_file, edges_file):
    """
    Load the complete grid of nodes and edges from the specified CSV files.
    """
    logger.info("Loading grid data...")
    
    # Load nodes
    nodes_df = pd.read_csv(nodes_file)
    nodes = {row['id']: {'id': row['id'], 'lat': row['lat'], 'lon': row['lon']} for _, row in nodes_df.iterrows()}
    
    # Load edges
    edges_df = pd.read_csv(edges_file)
    edges = {}
    for _, row in edges_df.iterrows():
        if row['node_id'] not in edges:
            edges[row['node_id']] = []
        edges[row['node_id']].append(row['connected_node_id'])
    
    logger.info("Complete grid data loaded.")
    return nodes, edges


# Utility function to calculate the Haversine distance between two coordinates
def haversine_distance(coord1, coord2):
    return geopy.distance.distance(coord1, coord2).meters

# Function to check if a coordinate is on land
def is_land(lat, lon):
    point = Point(lon, lat)  # Note: Point takes (lon, lat)
    return unified_land.contains(point)

# Function to check if the path between two coordinates crosses land
def crosses_land(lat1, lon1, lat2, lon2):
    line = LineString([(lon1, lat1), (lon2, lat2)])
    return unified_land.intersects(line)

# Function to find the nearest node from a given coordinate
def find_nearest_node(lat, lon, nodes):
    logger.info(f"Finding nearest node for coordinates ({lat}, {lon})...")
    nearest_node = None
    min_distance = float('inf')
    for node in nodes.values():
        distance = haversine_distance((lat, lon), (node['lat'], node['lon']))
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
    logger.info(f"Nearest node found: {nearest_node['id']} at distance {min_distance} meters.")
    return nearest_node

# Function to fetch weather data with caching
async def get_weather_data(lat, lon, session):
    key = (round(lat, 4), round(lon, 4))
    if key in weather_data_cache:
        logger.debug(f"Using cached weather data for ({lat}, {lon})")
        return weather_data_cache[key]

    logger.info(f"Fetching weather data for coordinates ({lat}, {lon})...")
    url = 'https://api.open-meteo.com/v1/forecast'
    params = {'latitude': lat, 'longitude': lon}
    try:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()
            weather_data_cache[key] = data
            logger.info(f"Weather data received for ({lat}, {lon}).")
            return data
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return None

# Function to simulate ocean currents
def get_current_data(lat, lon):
    logger.debug(f"Simulating ocean current data for ({lat}, {lon})...")
    current_speed = abs(math.sin(math.radians(lat))) * 2  # Simulated speed
    current_direction = lon % 360
    return {
        'current_speed': current_speed,
        'current_direction': current_direction
    }

# Function to calculate fuel efficiency adjustment based on weather and currents
# Function to calculate fuel efficiency adjustment based on weather and currents
def fuel_efficiency_adjustment(weather_data, current_data):
    adjustment = 1.0
    logger.debug('Calculating fuel efficiency adjustment')

    # Apply weather thresholds to adjust cost
    if weather_data:
        if 'temperature_2m' in weather_data:
            temp = weather_data['temperature_2m']
            if temp < 0 or temp > 35:
                logger.warning(f"Temperature out of optimal range: {temp}°C")
                adjustment += 0.1  # Penalty for extreme temperatures

        if 'precipitation_probability' in weather_data:
            precipitation_prob = weather_data['precipitation_probability']
            if precipitation_prob > 30:
                logger.warning(f"High precipitation probability: {precipitation_prob}%")
                adjustment += 0.2  # Penalty for high chance of rain

        if 'windspeed_10m' in weather_data:
            wind_speed = weather_data['windspeed_10m']
            if wind_speed > 15:
                logger.warning(f"High wind speed: {wind_speed} m/s")
                adjustment += 0.15  # Penalty for strong winds

        if 'visibility' in weather_data:
            visibility = weather_data['visibility']
            if visibility < 1000:
                logger.warning(f"Low visibility: {visibility} m")
                adjustment += 0.2  # Penalty for low visibility

        if 'cloudcover_total' in weather_data:
            cloud_cover = weather_data['cloudcover_total']
            if cloud_cover > 70:
                logger.warning(f"High cloud cover: {cloud_cover}%")
                adjustment += 0.1  # Penalty for heavy cloud cover

    # Adjust for ocean current speed
    if current_data:
        current_speed = current_data['current_speed']
        adjustment -= current_speed * 0.01  # Favorable currents reduce cost
        logger.debug(f"Adjusted for current speed: {current_speed} m/s, new adjustment factor: {adjustment}")

    logger.debug(f"Final fuel efficiency adjustment factor: {adjustment}")
    return adjustment

# Function to simulate hazard data (e.g., piracy zones, reefs)
def get_hazard_data(lat, lon):
    logger.debug(f"Checking hazard data for coordinates ({lat}, {lon})")
    
    # Define some example hazard zones with latitude and longitude boundaries
    hazard_zones = [
        {
            'min_lat': 27.0,
            'max_lat': 28.0,
            'min_lon': -83.0,
            'max_lon': -82.0,
            'name': 'Piracy Zone 1'
        },
        {
            'min_lat': 10.0,
            'max_lat': 11.0,
            'min_lon': 70.0,
            'max_lon': 71.0,
            'name': 'Reef Area 1'
        },
        # Add more hazard zones as needed
    ]

    # Check if the current coordinates are within any hazard zone
    is_hazardous = any(
        zone['min_lat'] <= lat <= zone['max_lat'] and
        zone['min_lon'] <= lon <= zone['max_lon']
        for zone in hazard_zones
    )

    if is_hazardous:
        logger.warning(f"Hazard detected at coordinates ({lat}, {lon})")
    
    return {'is_hazardous': is_hazardous}


# Cost function to calculate the cost between two nodes
async def calculate_cost(node_a, node_b, session):
    distance = haversine_distance((node_a['lat'], node_a['lon']), (node_b['lat'], node_b['lon']))

    # Fetch environmental data
    weather_data = await get_weather_data(node_b['lat'], node_b['lon'], session)
    current_data = get_current_data(node_b['lat'], node_b['lon'])

    # Calculate adjustment factor based on weather and current data
    fuel_adjustment = fuel_efficiency_adjustment(weather_data, current_data)
    fuel_cost = distance * fuel_adjustment

    # Include other penalties like hazards
    hazard_data = get_hazard_data(node_b['lat'], node_b['lon'])
    safety_penalty = 1e6 if hazard_data['is_hazardous'] else 0

    total_cost = fuel_cost + safety_penalty
    return total_cost

# Load grid data from CSV files
def load_grid_data(nodes_file, edges_file):
    """
    Loads the grid structure from CSV files. Keeps the full grid height and connectivity intact.
    """
    logger.info("Loading grid data...")
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)
    
    # Create nodes dictionary
    nodes = {row['id']: {'id': row['id'], 'lat': row['lat'], 'lon': row['lon']} for index, row in nodes_df.iterrows()}
    
    # Create edges dictionary
    edges = {}
    for index, row in edges_df.iterrows():
        if row['node_id'] not in edges:
            edges[row['node_id']] = []
        edges[row['node_id']].append(row['connected_node_id'])
    
    logger.info("Grid data loaded.")
    return nodes, edges

# Adjust A* Search for fixed grid height
async def a_star_search(start_node, goal_node, nodes, edges):
    """
    Perform A* search while maintaining the full grid structure.
    """
    logger.info(f"Starting A* search from Node {start_node['id']} to Node {goal_node['id']}...")
    frontier = PriorityQueue()
    frontier.put((0, start_node))
    came_from = {start_node['id']: None}
    cost_so_far = {start_node['id']: 0}
    
    async with aiohttp.ClientSession() as session:
        while not frontier.empty():
            current_priority, current = frontier.get()
            
            # Stop if the goal is reached
            if current['id'] == goal_node['id']:
                logger.info("Goal reached.")
                break

            for neighbor_id in edges[current['id']]:
                neighbor = nodes[neighbor_id]
                
                # Keep grid height intact but restrict traversal to relevant sections
                if not is_closer_to_goal(current, neighbor, goal_node):
                    continue

                # Skip if the neighbor is on land or if the path crosses land
                if is_land(neighbor['lat'], neighbor['lon']):
                    continue
                if crosses_land(current['lat'], current['lon'], neighbor['lat'], neighbor['lon']):
                    continue
                
                # Calculate new cost and prioritize the neighbor
                new_cost = cost_so_far[current['id']] + await calculate_cost(current, neighbor, session)
                if neighbor_id not in cost_so_far or new_cost < cost_so_far[neighbor_id]:
                    cost_so_far[neighbor_id] = new_cost
                    priority = new_cost + haversine_distance(
                        (neighbor['lat'], neighbor['lon']),
                        (goal_node['lat'], goal_node['lon'])
                    )
                    frontier.put((priority, neighbor))
                    came_from[neighbor_id] = current

    # Reconstruct path
    path = []
    current = goal_node
    while current:
        path.insert(0, current)
        current = came_from.get(current['id'])
    
    logger.info("Path reconstructed.")
    return path


def is_closer_to_goal(current_node, neighbor_node, goal_node):
    """
    Determines if the neighbor node is in the general direction of the goal.
    This filters out nodes that deviate significantly from the path to the goal.
    """
    current_to_goal_distance = haversine_distance(
        (current_node['lat'], current_node['lon']),
        (goal_node['lat'], goal_node['lon'])
    )
    neighbor_to_goal_distance = haversine_distance(
        (neighbor_node['lat'], neighbor_node['lon']),
        (goal_node['lat'], goal_node['lon'])
    )

    # Allow only neighbors that bring the path closer to the goal
    return neighbor_to_goal_distance < current_to_goal_distance


@app.route('/route', methods=['POST'])
def route():
    try:
        data = request.get_json()
        start_coords = data.get('start')
        goal_coords = data.get('goal')
        if not start_coords or not goal_coords:
            return jsonify({'error': 'Invalid input. Provide start and goal coordinates.'}), 400
        nodes, edges = load_grid_data('indian_ocean_nodes.csv', 'indian_ocean_edges.csv')
        start_node = find_nearest_node(start_coords['lat'], start_coords['lon'], nodes)
        goal_node = find_nearest_node(goal_coords['lat'], goal_coords['lon'], nodes)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        path = loop.run_until_complete(a_star_search(start_node, goal_node, nodes, edges))
        loop.close()
        return jsonify({'path': [{'id': n['id'], 'lat': n['lat'], 'lon': n['lon']} for n in path]})
    except Exception as e:
        logger.error(f"Error in route calculation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
