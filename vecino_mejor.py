import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count

def calculate_total_distance(route, dist_matrix):
    total_distance = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
    total_distance += dist_matrix[route[-1]][route[0]]
    return total_distance

def nearest_neighbor_algorithm(dist_matrix, start_node):
    n = len(dist_matrix)
    visited = [False] * n
    route = [start_node]
    visited[start_node] = True
    current_node = start_node

    while len(route) < n:
        min_distance = float('inf')
        next_node = None
        for i in range(n):
            if not visited[i] and dist_matrix[current_node][i] < min_distance:
                min_distance = dist_matrix[current_node][i]
                next_node = i
        visited[next_node] = True
        route.append(next_node)
        current_node = next_node

    route.append(start_node)
    return route

def two_opt_swap(route, i, k):
    new_route = route[:i] + route[i:k + 1][::-1] + route[k + 1:]
    return new_route

def lin_kernighan(route, dist_matrix):
    best_route = route
    best_cost = calculate_total_distance(best_route, dist_matrix)
    improvement = True

    while improvement:
        improvement = False
        for i in range(1, len(best_route) - 1):
            for k in range(i + 1, len(best_route)):
                new_route = two_opt_swap(best_route, i, k)
                new_cost = calculate_total_distance(new_route, dist_matrix)
                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    improvement = True
    return best_route, best_cost

def optimize_route_for_start_node(args):
    dist_matrix, start_node = args
    initial_route = nearest_neighbor_algorithm(dist_matrix, start_node)
    optimized_route, optimized_cost = lin_kernighan(initial_route, dist_matrix)
    return optimized_route, optimized_cost

def read_distance_matrix(filename):
    with open(filename, 'r') as file:
        distance_matrix = [list(map(float, line.strip().split())) for line in file]
    return np.array(distance_matrix)

def read_coordinates(filename):
    with open(filename, 'r') as file:
        coordinates = [tuple(map(float, line.strip().split())) for line in file]
    return coordinates

def plot_route(coordinates, route, title="Ruta Optimizada"):
    route_coords = [coordinates[i] for i in route] + [coordinates[route[0]]]
    x, y = zip(*route_coords)

    plt.figure(figsize=(10, 8))
    plt.plot(x, y, marker='o', linestyle='-', color='b')

    for i, (x_coord, y_coord) in enumerate(coordinates):
        plt.text(x_coord, y_coord, str(i), fontsize=10, color='red', ha='right', va='bottom')

    plt.title(title)
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid()
    plt.show()

def main():
    coord_filename = "Cordenadas/Coord1.txt"
    dist_filename = "Cordenadas/Dist1.txt"

    coordinates = read_coordinates(coord_filename)
    distance_matrix = read_distance_matrix(dist_filename)
    np.fill_diagonal(distance_matrix, 0)

    start_time = time.time()

    # Parallel processing using Pool
    with Pool(processes=7) as pool:
        results = pool.map(optimize_route_for_start_node, [(distance_matrix, i) for i in range(len(distance_matrix))])

    # Find the best route from all results
    best_route, best_cost = min(results, key=lambda x: x[1])

    end_time = time.time()

    print("Ruta optimizada:", best_route)
    print("Costo optimizado:", best_cost)
    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

    plot_route(coordinates, best_route, title="Ruta Optimizada")

if __name__ == "__main__":
    main()
