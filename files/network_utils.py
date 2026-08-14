# ================================================================
# network_utils.py
# Shared utilities for real street-network distance computation and
# population-weighted customer location generation.
# ================================================================
import os
import pickle
import random
import numpy as np
import pandas as pd
import osmnx as ox
from shapely.geometry import Point
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def build_street_graphs(place, target_crs="EPSG:32630", cache_dir="osm_cache",
                        extra_latlon_points=None):
    """
    NOTE (bug fix): an earlier version of this function downloaded only
    the exact administrative boundary of `place` (Hackney). If a point
    such as the depot (Walthamstow) lies outside this boundary,
    nearest_nodes silently snaps it to the closest node *inside* Hackney
    -- meaning the true depot-to-customer distance would be
    underestimated. Passing extra_latlon_points (e.g.,
    [(51.5830, -0.0198)] for the depot) expands the query area enough
    to include these points as well.
    """
    os.makedirs(cache_dir, exist_ok=True)
    drive_path = os.path.join(cache_dir, "G_drive.pkl")
    bike_path = os.path.join(cache_dir, "G_bike.pkl")
    if os.path.exists(drive_path) and os.path.exists(bike_path):
        with open(drive_path, "rb") as f:
            G_drive = pickle.load(f)
        with open(bike_path, "rb") as f:
            G_bike = pickle.load(f)
        return G_drive, G_bike

    boundary = ox.geocode_to_gdf(place)
    polygon = boundary.iloc[0].geometry

    if extra_latlon_points:
        from shapely.geometry import MultiPoint
        extra_pts = [Point(lon, lat) for lat, lon in extra_latlon_points]
        polygon = MultiPoint([*list(polygon.exterior.coords), *[(p.x, p.y) for p in extra_pts]]).convex_hull
        # small surrounding buffer so connecting streets are also covered (~500 m)
        polygon = polygon.buffer(0.005)

    G_drive = ox.graph_from_polygon(polygon, network_type="drive", simplify=True)
    G_bike = ox.graph_from_polygon(polygon, network_type="bike", simplify=True)
    G_drive = ox.truncate.largest_component(G_drive, strongly=True)
    G_bike = ox.truncate.largest_component(G_bike, strongly=True)
    G_drive = ox.project_graph(G_drive, to_crs=target_crs)
    G_bike = ox.project_graph(G_bike, to_crs=target_crs)

    with open(drive_path, "wb") as f:
        pickle.dump(G_drive, f)
    with open(bike_path, "wb") as f:
        pickle.dump(G_bike, f)
    return G_drive, G_bike


def build_network_distance_matrix(G, points_xy):
    """Pairwise shortest-path distance matrix (km) over a street graph, via Dijkstra."""
    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]
    node_ids = ox.distance.nearest_nodes(G, X=xs, Y=ys)
    node_list = list(G.nodes())
    idx_of_node = {n: i for i, n in enumerate(node_list)}
    rows, cols, wts = [], [], []
    for u, v, data in G.edges(data=True):
        w = data.get("length", None)
        if w is None:
            continue
        rows.append(idx_of_node[u]); cols.append(idx_of_node[v]); wts.append(w)
    sparse_graph = csr_matrix((wts, (rows, cols)), shape=(len(node_list), len(node_list)))
    source_indices = [idx_of_node[n] for n in node_ids]
    dist_from_sources = dijkstra(sparse_graph, directed=True, indices=source_indices)
    n_pts = len(points_xy)
    dist_m = np.array([[dist_from_sources[i, source_indices[j]] if np.isfinite(dist_from_sources[i, source_indices[j]]) else np.inf
                         for j in range(n_pts)] for i in range(n_pts)])
    return dist_m / 1000.0, node_ids


def build_drone_distance_matrix(points_xy):
    """Pairwise Euclidean (straight-line) distance matrix (km), for drone routing."""
    pts = np.array(points_xy)
    n = len(pts)
    return np.array([[np.sqrt(((pts[i] - pts[j]) ** 2).sum()) / 1000.0 for j in range(n)] for i in range(n)])


def report_unreachable(dist_km, name, point_labels):
    bad = np.where(np.isinf(dist_km[1]))[0]
    if len(bad) > 0:
        print(f"WARNING: in the {name} network, {len(bad)} point(s) are unreachable from the UCC: "
              f"{[point_labels[i] for i in bad]}")
    else:
        print(f"OK: {name} network -- all points are reachable from the UCC.")


# ================================================================
# More realistic customer location generation: weighting by real LSOA
# population (2021 Census, England and Wales; ONS/Nomis), rather than
# uniform sampling across the whole borough. If the ONS data download
# fails for any reason (URL change, network outage), the function
# automatically falls back to the previous method (uniform random) with
# a clear warning -- the pipeline is never blocked by this failure.
#
# Sources:
#   LSOA boundaries: ONS Open Geography Portal (ArcGIS FeatureServer)
#   LSOA population: ONS Census 2021, Table TS001 (Number of usual
#     residents), bulk download via Nomis
# ================================================================
def generate_customer_locations_realistic(N_cust, place, seed=None, verbose=True):
    """
    Returns: (customers_latlon, method), where method is either
    'population_weighted' or 'uniform_random' (on fallback).
    """
    rng = random.Random(seed)
    try:
        import geopandas as gpd
        import requests
        import zipfile
        import io

        boundary = ox.geocode_to_gdf(place)
        polygon = boundary.iloc[0].geometry
        minx, miny, maxx, maxy = polygon.bounds

        # 1) fetch LSOA boundaries within Hackney's geographic extent
        lsoa_service = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
                         "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFC_V10/"
                         "FeatureServer/0/query")
        params = {
            "where": "1=1",
            "outFields": "LSOA21CD,LSOA21NM",
            "f": "geojson",
            "geometry": f"{minx},{miny},{maxx},{maxy}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326", "outSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
        }
        resp = requests.get(lsoa_service, params=params, timeout=60)
        resp.raise_for_status()
        lsoa_gdf = gpd.GeoDataFrame.from_features(resp.json()["features"], crs="EPSG:4326")
        lsoa_gdf = lsoa_gdf[lsoa_gdf.geometry.intersects(polygon)].reset_index(drop=True)
        if lsoa_gdf.empty:
            raise RuntimeError("No LSOA returned for this extent")

        # 2) download the TS001 population dataset and filter to Hackney's LSOAs
        zip_resp = requests.get(
            "https://www.nomisweb.co.uk/output/census/2021/census2021-ts001.zip", timeout=120)
        zip_resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(zip_resp.content))
        lsoa_csv_name = next(n for n in zf.namelist() if "lsoa" in n.lower() and n.lower().endswith(".csv"))
        pop_df = pd.read_csv(zf.open(lsoa_csv_name))
        code_col = next(c for c in pop_df.columns if "geography code" in c.lower() or c.lower() == "geography_code")
        pop_col = next(c for c in pop_df.columns if "observation" in c.lower() or "count" in c.lower() or "total" in c.lower())
        pop_df = pop_df[[code_col, pop_col]].rename(columns={code_col: "LSOA21CD", pop_col: "population"})

        lsoa_gdf = lsoa_gdf.merge(pop_df, on="LSOA21CD", how="left")
        lsoa_gdf["population"] = lsoa_gdf["population"].fillna(0)
        if lsoa_gdf["population"].sum() <= 0:
            raise RuntimeError("Zero population for all LSOAs -- population data merge failed")

        # 3) population-weighted sampling: first select an LSOA, then a uniform point within it
        weights = lsoa_gdf["population"].values / lsoa_gdf["population"].sum()
        customers_latlon = []
        attempts = 0
        while len(customers_latlon) < N_cust and attempts < N_cust * 50:
            attempts += 1
            idx = rng.choices(range(len(lsoa_gdf)), weights=weights, k=1)[0]
            geom = lsoa_gdf.geometry.iloc[idx]
            gminx, gminy, gmaxx, gmaxy = geom.bounds
            p = Point(rng.uniform(gminx, gmaxx), rng.uniform(gminy, gmaxy))
            if geom.contains(p) and polygon.contains(p):
                customers_latlon.append((p.y, p.x))
        if len(customers_latlon) < N_cust:
            raise RuntimeError("Population-weighted sampling did not reach the required number of points")

        if verbose:
            print(f"OK: generated {N_cust} customer locations weighted by real LSOA population "
                  f"(Census 2021, ONS/Nomis).")
        return customers_latlon, "population_weighted"

    except Exception as e:
        if verbose:
            print(f"WARNING: fetching ONS population data failed ({e}) -- "
                  f"falling back to uniform random sampling (previous method).")
        boundary = ox.geocode_to_gdf(place)
        polygon = boundary.iloc[0].geometry
        minx, miny, maxx, maxy = polygon.bounds
        customers_latlon = []
        while len(customers_latlon) < N_cust:
            p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if polygon.contains(p):
                customers_latlon.append((p.y, p.x))
        return customers_latlon, "uniform_random"
