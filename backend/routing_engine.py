import networkx as nx

# Initialize Chennai street graph
city_map = nx.Graph()
city_map.add_edge("T_Nagar", "Guindy", weight=15)
city_map.add_edge("Guindy", "Velachery", weight=10)
city_map.add_edge("T_Nagar", "Saidapet", weight=8)
city_map.add_edge("Saidapet", "Velachery", weight=12)

def calculate_safe_route(start: str, destination: str, flooded_areas: list):
    safe_map = city_map.copy()
    
    # Increase weight massively for flooded nodes to force rerouting
    for area in flooded_areas:
        if area in safe_map.nodes:
            for neighbor in safe_map.neighbors(area):
                safe_map[area][neighbor]['weight'] += 999 
                
    try:
        route = nx.shortest_path(safe_map, source=start, target=destination, weight='weight')
        return route
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []