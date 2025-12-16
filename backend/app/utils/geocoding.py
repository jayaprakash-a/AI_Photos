import requests
import logging

logger = logging.getLogger(__name__)

def reverse_geocode(lat: float, lon: float) -> str:
    """
    Resolves a location name from latitude and longitude using OpenStreetMap Nominatim.
    Respects OSM's Usage Policy (Max 1 req/sec, specific User-Agent).
    """
    if not lat or not lon:
        return None
        
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 10  # City level
        }
        headers = {
            "User-Agent": "PhotosApp/1.0 (internal-dev-project)",
            "Accept-Language": "en"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            address = data.get("address", {})
            
            # Construct a readable location string
            city = address.get("city") or address.get("town") or address.get("village")
            country = address.get("billing_country") or address.get("country")
            
            if city and country:
                return f"{city}, {country}"
            elif country:
                return country
            elif city:
                return city
            else:
                return data.get("display_name", "").split(",")[0]
                
    except Exception as e:
        logger.error(f"Geocoding failed: {e}")
        
    return None

def search_place(query: str) -> list:
    """
    Search for a place by name.
    Returns: List of dicts with {'name': str, 'lat': float, 'lon': float}
    """
    if not query or len(query) < 3:
        return []
        
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": query,
            "format": "json",
            "limit": 5,
            "addressdetails": 1
        }
        headers = {
            "User-Agent": "PhotosApp/1.0 (internal-dev-project)",
            "Accept-Language": "en"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            results = []
            for item in response.json():
                # Format a nice name - prioritize specific places over generic locations
                addr = item.get('address', {})
                name_parts = []
                
                # Check for specific POIs first (landmarks, buildings, tourist attractions)
                poi_types = ['tourism', 'amenity', 'building', 'historic', 'leisure']
                poi_name = None
                for poi_type in poi_types:
                    if addr.get(poi_type):
                        poi_name = addr.get(poi_type)
                        break
                
                # Use the name field if it's a specific place
                if item.get('name'):
                    name_parts.append(item.get('name'))
                elif poi_name:
                    name_parts.append(poi_name)
                
                # Add city/location context
                if addr.get('city') or addr.get('town') or addr.get('village'):
                    name_parts.append(addr.get('city') or addr.get('town') or addr.get('village'))
                elif addr.get('suburb'):
                    name_parts.append(addr.get('suburb'))
                    
                if addr.get('country'):
                    name_parts.append(addr.get('country'))
                
                display_name = ", ".join(name_parts)
                if not display_name:
                    # Fallback to the full display name
                    display_name = item.get('display_name', '')
                    
                results.append({
                    "name": display_name,
                    "display_name": item.get('display_name'), # Full name
                    "lat": float(item.get('lat')),
                    "lon": float(item.get('lon'))
                })
            return results
    except Exception as e:
        logger.error(f"Place search failed: {e}")
        
    return []
