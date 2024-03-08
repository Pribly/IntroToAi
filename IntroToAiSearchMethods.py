import heapq
import math
import csv
from queue import Queue
from collections import defaultdict
import time
import matplotlib.pyplot as plt


def read_file_from_csv(file_path):
    lists = defaultdict(list)
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            city, lat, lon = row
            lists[city] = (float(lat), float(lon))
    return lists

def read_adjacent_cities(file_path):
    adjacent_cities = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            city1, city2 = line.strip().split()
            adjacent_cities[city1].append(city2)
            adjacent_cities[city2].append(city1)
    return adjacent_cities


class Node:
    def __init__(self, city, lat, lon):
        self.city = city
        self.lat = lat
        self.lon = lon
        self.g = float('inf')
        self.h = 0
        self.f = 0
        self.parent = None
    def __lt__(self, other):
        return self.f < other.f
    
def distance(node1,node12,node2,node22):
    return(math.sqrt(pow((node1-node2),2)+pow((node12-node22),2)))

#
def heuristic(neighbor1,neighbor2,goal1,goal2):
    return distance(neighbor1,neighbor2,goal1,goal2)

#Created from Pseudocode
def breadth_first_search(start, goal, nodes, adjacent_cities,start_time):
    #Searches level by level until a solution is find 
    #Uses Que to maintain order of exploration
    open_queue = Queue()
    #Closed set keeps track of visited nodes
    closed_set = set()
    start.g = 0
    timeout=10
    

    open_queue.put(start)

    while not open_queue.empty():
        current_node = open_queue.get()

        if current_node.city == goal.city:
            path = []
            total_distance = 0
            while current_node:
                path.insert(0, (current_node.lat, current_node.lon))
                if current_node.parent:
                    total_distance += distance(current_node.lat, current_node.lon,
                                               current_node.parent.lat, current_node.parent.lon)
                current_node = current_node.parent
            return path, total_distance

        closed_set.add(current_node)

        for neighbor_city in adjacent_cities[current_node.city]:
            neighbor = nodes[neighbor_city]
            if neighbor not in closed_set and neighbor not in open_queue.queue:
                tentative_g = current_node.g + distance(current_node.lat, current_node.lon,
                                                        neighbor.lat, neighbor.lon)
                if tentative_g < neighbor.g:
                    neighbor.g = tentative_g
                    neighbor.parent = current_node
                    open_queue.put(neighbor)
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            return None, None  # Timeout reached
    return None, None  # No path found

#Created from Pseudocode
def depth_first_search(start, goal, nodes, adjacent_cities,start_time):
    open_stack = []
    closed_set = set()
    start.g = 0
    timeout=10
    #Explores as far into a branch before backtracking
    open_stack.append(start)

    while open_stack:
        current_node = open_stack.pop()

        if current_node.city == goal.city:
            path = []
            total_distance = 0
            while current_node:
                path.insert(0, (current_node.lat, current_node.lon))
                if current_node.parent:
                    total_distance += distance(current_node.lat, current_node.lon,
                                               current_node.parent.lat, current_node.parent.lon)
                current_node = current_node.parent
            return path, total_distance

        closed_set.add(current_node)

        for neighbor_city in reversed(adjacent_cities[current_node.city]):
            neighbor = nodes[neighbor_city]
            if neighbor not in closed_set and neighbor not in open_stack:
                tentative_g = current_node.g + distance(current_node.lat, current_node.lon,
                                                        neighbor.lat, neighbor.lon)
                if tentative_g < neighbor.g:
                    neighbor.g = tentative_g
                    neighbor.parent = current_node
                    open_stack.append(neighbor)
                    elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            return None, None  # Timeout reached

    return None, None  # No path found

def iterative_deepening_dfs(start, goal, nodes, adjacent_cities,start_time):
    max_depth=int(50)
    for depth_limit in range(max_depth):
        visited = set()
        path, total_distance = depth_limited_dfs(
            start, goal, nodes, adjacent_cities, depth_limit, visited, time.time()
        )
        if path:
            return path, total_distance
    return None, None  # No path found

def depth_limited_dfs(current_node, goal, nodes, adjacent_cities, depth_limit, visited, start_time):
    if time.time() - start_time > 10:
        return None, None  # Timeout reached

    if depth_limit < 0 or current_node in visited:
        return None, None  # No path found within depth limit or already visited

    visited.add(current_node)

    if current_node.city == goal.city:
        path = []
        total_distance = 0
        while current_node:
            path.insert(0, (current_node.lat, current_node.lon))
            if current_node.parent:
                total_distance += distance(
                    current_node.lat, current_node.lon, current_node.parent.lat, current_node.parent.lon
                )
            current_node = current_node.parent
        return path, total_distance

    for neighbor_city in adjacent_cities[current_node.city]:
        neighbor = nodes[neighbor_city]
        tentative_g = current_node.g + distance(
            current_node.lat, current_node.lon, neighbor.lat, neighbor.lon
        )
        if tentative_g < neighbor.g or neighbor not in visited:
            neighbor.g = tentative_g
            neighbor.parent = current_node

            result_path, result_distance = depth_limited_dfs(
                neighbor, goal, nodes, adjacent_cities, depth_limit - 1, visited, start_time
            )
            if result_path:
                return result_path, result_distance

    return None, None  # No path found

#Changed a couple lines from my A* Script
def best_first_search(start, goal, nodes, adjacent_cities,start_time):
    open_set = []
    closed_set = set()
    start.g = 0
    timeout=10

    heapq.heappush(open_set, (0, start))

    while open_set:
        current_node = heapq.heappop(open_set)[1]

        if current_node.city == goal.city:
            path = []
            total_distance = 0
            while current_node:
                path.insert(0, (current_node.lat, current_node.lon))
                if current_node.parent:
                    total_distance += distance(current_node.lat, current_node.lon,
                                               current_node.parent.lat, current_node.parent.lon)
                current_node = current_node.parent
            return path, total_distance

        closed_set.add(current_node)

        for neighbor_city in adjacent_cities[current_node.city]:
            neighbor = nodes[neighbor_city]
            if neighbor not in closed_set:
                tentative_g = current_node.g + distance(current_node.lat, current_node.lon,
                                                        neighbor.lat, neighbor.lon)
                if tentative_g < neighbor.g:
                    neighbor.g = tentative_g
                    neighbor.h = distance(neighbor.lat,neighbor.lon, goal.lat,goal.lon)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current_node

                    if neighbor not in open_set:
                        heapq.heappush(open_set, (neighbor.f, neighbor))
                        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            return None, None  # Timeout reached

    return None, None  # No path found



#Pulled from other Class
def astar(start, goal, nodes, adjacent_cities,start_time):
    open_set = []
    closed_set = set()
    start.g = 0
    timeout=10

    heapq.heappush(open_set, (0, start))

    while open_set:
        current_node = heapq.heappop(open_set)[1]

        if current_node.city == goal.city:
            path = []
            total_distance = 0
            while current_node:
                path.insert(0, (current_node.lat, current_node.lon))
                if current_node.parent:
                    total_distance += distance(current_node.lat, current_node.lon,
                                               current_node.parent.lat, current_node.parent.lon)
                current_node = current_node.parent
            return path, total_distance

        closed_set.add(current_node)

        for neighbor_city in adjacent_cities[current_node.city]:
            neighbor = nodes[neighbor_city]
            if neighbor in closed_set:
                continue

            tentative_g = current_node.g + distance(current_node.lat, current_node.lon,
                                                    neighbor.lat, neighbor.lon)
            if tentative_g < neighbor.g:
                neighbor.g = tentative_g
                neighbor.h = heuristic(neighbor.lat, neighbor.lon, goal.lat, goal.lon)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current_node

                if neighbor not in open_set:
                    heapq.heappush(open_set, (neighbor.f, neighbor))
                elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            return None, None  # Timeout reached

    return None, None  # No path found

def start_cityf(lists):
    start_city = input("Enter the name of your starting city: ")
    if start_city in lists:
        return str(start_city)
    else:
        print("City not valid, try again")
        start_cityf(lists)
        
def end_cityf(lists):
    goal_city=input(" Enter the name of your goal city: ")
    if goal_city in lists:
        return str(goal_city)
    else:
        print("City not valid, try again")
        end_cityf(lists)
        
        
def which_method(start_node,goal_node,nodes,adjacent_cities):
    method=input("Enter the method number you want to use: 1. breadth-first search 2. depth-first search 3. ID-DFS search 4. best-first search 5.A* ")
    start_time=time.time()
    if method == str(1):  # Breadth-First Search
        path, total_distance = breadth_first_search(start_node, goal_node, nodes,adjacent_cities,start_time)
        endtime=time.time()
        return path,total_distance,start_time,endtime
    elif method == str(2):  # Depth-First Search
        path, total_distance = depth_first_search(start_node, goal_node, nodes, adjacent_cities,start_time)
        endtime=time.time()
        return path,total_distance,start_time,endtime
    elif method == str(3):  # Iterative Deepening Depth-First Search
        path, total_distance =iterative_deepening_dfs(start_node, goal_node, nodes,adjacent_cities,start_time)
        endtime=time.time()
        return path,total_distance,start_time,endtime
    elif method == str(4):  # Best-First Search
        path, total_distance = best_first_search(start_node, goal_node, nodes,adjacent_cities,start_time)
        endtime=time.time()
        return path,total_distance,start_time,endtime
    elif method == str(5):  # A* Search
        path, total_distance= astar(start_node, goal_node, nodes,adjacent_cities,start_time)
        endtime=time.time()
        return path,total_distance,start_time,endtime
    else:
        print("Invalid method number entered.")
    
    
    
if __name__ == "__main__":
    file_path = "coordinates.csv"
    lists = read_file_from_csv(file_path)
    nodes = {city: Node(city, lat, lon) for city, (lat, lon) in lists.items()}
    
    adjacent_cities_file_path = "Adjacencies.txt"  # Change this to the actual path
    adjacent_cities = read_adjacent_cities(adjacent_cities_file_path)
    
    
    #starts="Topeka"
    #goals="Coldwater"
    starts=start_cityf(lists)
    start_node=Node(starts,lists[starts][0],lists[starts][1])
    goals=end_cityf(lists)
    goal_node=Node(goals,lists[goals][0],lists[goals][1])
    path,total_distance,start_time,endtime= which_method(start_node,goal_node,nodes,adjacent_cities)

# Print the result
    if path:
        print("Path:", path)
        print("Total Distance:", total_distance, "km")
        print("Time taken:", endtime - start_time, "seconds")
    else:
        print("No path found.")
    path = [(lat, lon) for lat, lon in path]

    for i in range(0, len(path) - 1):
       p1, p2 = path[i], path[i + 1]
       x_val = [p1[0], p2[0]]
       y_val = [p1[1], p2[1]]
       plt.plot(x_val, y_val, label=f"Segment {i + 1}")

        # Plot city labels for cities on the path
    for city, (lat, lon) in lists.items():
        if (lat, lon) in path:
            plt.text(lat, lon, city, fontsize=8, ha='right', va='bottom')

    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Path and City Labels')
    #plt.legend()
    plt.show()
