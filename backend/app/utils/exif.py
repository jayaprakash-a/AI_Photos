from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime

def get_decimal_from_dms(dms, ref):
    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]

    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    
    if ref in ['S', 'W']:
        decimal = -decimal
        
    return decimal

def get_exif_data(image_path: str):
    """
    Extracts timestamp, latitude, and longitude from an image.
    Returns:
        dict: {'timestamp': datetime, 'latitude': float, 'longitude': float}
    """
    data = {'timestamp': None, 'latitude': None, 'longitude': None}
    
    try:
        image = Image.open(image_path)
        exif = image._getexif()

        if not exif:
            return data

        # Parse basic tags
        for tag, value in exif.items():
            decoded = TAGS.get(tag, tag)
            
            if decoded == 'DateTimeOriginal' or decoded == 'DateTime':
                # Prefer DateTimeOriginal, fallback later if needed
                if not data['timestamp'] and value:
                    try:
                        data['timestamp'] = datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                    except ValueError:
                        pass
        
        # Parse GPS
        gps_info = exif.get(34853) # 34853 is GPSInfo
        if gps_info:
            gps_data = {}
            for t in gps_info:
                sub_decoded = GPSTAGS.get(t, t)
                gps_data[sub_decoded] = gps_info[t]

            lat_dms = gps_data.get('GPSLatitude')
            lat_ref = gps_data.get('GPSLatitudeRef')
            lon_dms = gps_data.get('GPSLongitude')
            lon_ref = gps_data.get('GPSLongitudeRef')

            if lat_dms and lat_ref and lon_dms and lon_ref:
                data['latitude'] = get_decimal_from_dms(lat_dms, lat_ref)
                data['longitude'] = get_decimal_from_dms(lon_dms, lon_ref)
                
    except Exception as e:
        print(f"Error parsing EXIF for {image_path}: {e}")

    return data
