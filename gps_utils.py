import argparse
import pandas as pd
import numpy as np
import cv2
import googlemaps
from datetime import datetime
import io
from tqdm import tqdm, trange
import hashlib

def string_to_color(s):
    hash = hashlib.md5(s.encode()).hexdigest()
    color = tuple(int(hash[i:i+2], 16) for i in (0, 2, 4))
    factor = (180 * 3) / sum(color)
    color = tuple(int(c * factor) for c in color)
    color = tuple(min(c, 255) for c in color)
    return color

def get_map(size=(512,512), zoom=18, scale=2, gps_ids=None):
    maps_client = googlemaps.Client(key='AIzaSyDSg-8DixBmaPzyJrjYhQeKnCRk5OAsRkQ')
    response = maps_client.static_map(
        size=size,
        zoom=zoom,
        center=(39.351, -76.345),
        maptype='satellite',
        format="png32",
        scale=scale
    )
    image_stream = io.BytesIO()
    for chunk in response:
        if chunk: image_stream.write(chunk)
    image_data_np = np.frombuffer(image_stream.getvalue(), dtype=np.uint8)
    map = cv2.imdecode(image_data_np, cv2.IMREAD_UNCHANGED)
    if map.shape[2] == 4:
        map = cv2.cvtColor(map, cv2.COLOR_BGRA2BGR)
    if gps_ids is not None:
        #map color legend on map image as far right as possible
        for i, id in enumerate(gps_ids):
            color = string_to_color(id)
            if 'A' in id and '0' in id: #circle
                map = cv2.circle(map, (size[0]*2-100, 50+50*i), 5, color, -1)
            else: #rectange
                map = cv2.rectangle(map, (size[0]*2-100-5, 50+50*i-5), (size[0]*2-100+5, 50+50*i+5), color, -1)
            map = cv2.putText(map, id, (size[0]*2-90, 50+50*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return map

def to_implicit(lat, lon, zoom):
    implicit_height = (2*256)*(2**zoom)
    implicit_width  = (2*256)*(2**zoom)
    
    R = implicit_width/(2*np.pi)
    FE = 180
    lonRad = np.radians(lon + FE)
    implicit_x = lonRad * R
    
    latRad = np.radians(lat)
    verticalOffsetFromEquator = R * np.log(np.tan(np.pi / 4 + latRad / 2))
    implicit_y = implicit_height / 2 - verticalOffsetFromEquator
    
    return implicit_x, implicit_y

def to_pixel(lat, lon, center_lat, center_lon, zoom, width, height):
    cix, ciy = to_implicit(center_lat, center_lon, zoom)
    ix, iy = to_implicit(np.array(lat), np.array(lon), zoom)
    x = (ix - cix) + width/2
    y = (iy - ciy) + height/2
    return int(x), int(y)

def parse_gps(df, label):
    ms2pos = {}
    for row in df.iloc:
        ms = row['t']
        x, y = to_pixel(row['lt'], row['ln'], 39.351, -76.345, zoom=18, width=1024, height=1024)
        ms2pos[ms] = (label, x, y)
    return ms2pos

def main():
    parser = argparse.ArgumentParser(description="Plot CSV data on a map and create a video.")
    parser.add_argument("csv", type=str, help="Path to the CSV file.")
    parser.add_argument("--ids", type=str, nargs='+', help="IDs to plot.")
    args = parser.parse_args()

    #assign a color to each id
    colors = {id: string_to_color(id) for id in args.ids}

    df = pd.read_csv(args.csv)
    df = df[df['fix'] != 'network']
    if args.ids:
        df = df[df['id'].isin(args.ids)]

    map_img = get_map()
    history = []

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 5.0, (map_img.shape[1], map_img.shape[0]))

    num_rows = len(df)
    for i in trange(num_rows):
        row = df.iloc[i]
        history.append(row)
        frame = map_img.copy()
        for hrow in history[-10:]:
            label = hrow['id']
            x, y = to_pixel(hrow['lt'], hrow['ln'], 39.351, -76.345, zoom=18, width=1024, height=1024)
            frame = cv2.circle(frame, (x, y), 5, colors[label], -1)
        out.write(frame)

    out.release()

if __name__ == "__main__":
    main()
