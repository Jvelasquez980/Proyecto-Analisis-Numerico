import numpy as np
import matplotlib.pyplot as plt
import time


def calculate_total_distance(route, dist_matrix):
    total_distance = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
    # Siempre cerrar el ciclo
    total_distance += dist_matrix[route[-1]][route[0]]
    return total_distance
    
def find_best_start_node(dist_matrix):
    """
    Encuentra el nodo inicial que genera la ruta de menor costo usando el algoritmo del vecino más cercano.
    """
    n = len(dist_matrix)
    best_route = None
    best_cost = float('inf')
    best_start_node = None

    for start_node in range(n):
        route = nearest_neighbor_algorithm(dist_matrix, start_node)
        cost = calculate_total_distance(route, dist_matrix)
        if cost < best_cost:
            best_cost = cost
            best_route = route
            best_start_node = start_node

    return best_start_node, best_route


def nearest_neighbor_algorithm(dist_matrix, start_node=0):
    """
    Genera una ruta inicial usando el algoritmo del vecino más cercano.
    """
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

    route.append(start_node)  # Regresar al inicio
    return route


def two_opt_swap(route, i, k):
    """
    Aplica un intercambio 2-opt en una ruta.
    """
    new_route = route[:i] + route[i:k + 1][::-1] + route[k + 1:]
    return new_route


def lin_kernighan(initial_route, dist_matrix):
    """
    Implementa la heurística Lin-Kernighan para optimizar una ruta.
    """
    best_route = initial_route
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
    return best_route


def read_distance_matrix(filename):
    """
    Lee la matriz de distancias desde un archivo.
    """
    with open(filename, 'r') as file:
        distance_matrix = [list(map(float, line.strip().split())) for line in file]
    return np.array(distance_matrix)


def read_coordinates(filename):
    """
    Lee las coordenadas de las ciudades desde un archivo.
    """
    with open(filename, 'r') as file:
        coordinates = [tuple(map(float, line.strip().split())) for line in file]
    return coordinates


def plot_route(coordinates, route, title="Ruta Optimizada"):
    """
    Dibuja la ruta en un gráfico e indica el número de los nodos.
    """
    route_coords = [coordinates[i] for i in route] + [coordinates[route[0]]]  # Regresar al inicio
    x, y = zip(*route_coords)

    plt.figure(figsize=(10, 8))
    plt.plot(x, y, marker='o', linestyle='-', color='b')

    # Agregar etiquetas para los nodos
    for i, (x_coord, y_coord) in enumerate(coordinates):
        plt.text(x_coord, y_coord, str(i), fontsize=10, color='red', ha='right', va='bottom')

    plt.title(title)
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid()
    plt.show()



def main():
    # Rutas a los archivos de entrada
    coord_filename = "Cordenadas\Coord1.txt"
    dist_filename = "Cordenadas\Dist1.txt"

    # Leer datos desde los archivos
    coordinates = read_coordinates(coord_filename)
    distance_matrix = read_distance_matrix(dist_filename)

    # Validación: Asegurarse de que la diagonal sea infinita
    np.fill_diagonal(distance_matrix, 0)

    # Comenzar el temporizador
    start_time = time.time()

    # Generar una ruta inicial usando el algoritmo del vecino más cercano
    initial_route = nearest_neighbor_algorithm(distance_matrix)

    costo_rutaInicial = calculate_total_distance(initial_route, distance_matrix)

    # Optimizar la ruta usando Lin-Kernighan
    optimized_route = lin_kernighan(initial_route, distance_matrix)

    # Calcular el costo optimizado
    optimized_cost = calculate_total_distance(optimized_route, distance_matrix)

    # Finalizar el temporizador
    end_time = time.time()
    # Validar que el nodo inicial sea igual al final
    if optimized_route[0] != optimized_route[-1]:
        optimized_route.append(optimized_route[0])

    # Mostrar resultados
    print("Ruta inicial:", initial_route)
    print("Ruta optimizada:", optimized_route)
    print("Costo optimizado:", optimized_cost)
    print("Costo ruta inicial", costo_rutaInicial)
    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

    # Graficar la ruta optimizada
    plot_route(coordinates, optimized_route, title="Ruta Optimizada")


if __name__ == "__main__":
    main()
