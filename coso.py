import numpy as np
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# Función para cargar distancias y coordenadas desde archivos
def cargar_datos(distancia_path, coordenadas_path):
    distancias = np.loadtxt(distancia_path)
    coordenadas = np.loadtxt(coordenadas_path)
    return distancias, coordenadas

# Función de la colonia de hormigas
def colonia_hormigas(distancias, coordenadas, num_hormigas, num_iteraciones, num_nodos, alpha, beta, rho, Q):
    # Inicialización de la feromona
    feromona = np.ones_like(distancias) / num_nodos

    mejor_ruta = None
    mejor_distancia = float('inf')

    for _ in range(num_iteraciones):
        rutas = []
        distancias_hormigas = []

        for _ in range(num_hormigas):
            ruta = explorar_ruta(feromona, distancias, num_nodos, alpha, beta)
            distancia = calcular_distancia(ruta, distancias)
            rutas.append(ruta)
            distancias_hormigas.append(distancia)

            if distancia < mejor_distancia:
                mejor_ruta = ruta
                mejor_distancia = distancia

        # Actualización de la feromona
        feromona = (1 - rho) * feromona  # Evaporación de la feromona
        for i in range(num_hormigas):
            for j in range(num_nodos - 1):
                feromona[rutas[i][j], rutas[i][j + 1]] += Q / distancias_hormigas[i]

    return mejor_ruta, mejor_distancia

# Función para explorar una ruta con una hormiga
def explorar_ruta(feromona, distancias, num_nodos, alpha, beta):
    ruta = [random.randint(0, num_nodos - 1)]
    while len(ruta) < num_nodos:
        probabilidades = calcular_probabilidades(feromona, distancias, ruta, num_nodos, alpha, beta)
        siguiente_nodo = seleccionar_nodo(probabilidades)
        ruta.append(siguiente_nodo)
    return ruta

# Función para calcular las probabilidades de selección del siguiente nodo
def calcular_probabilidades(feromona, distancias, ruta, num_nodos, alpha, beta):
    actual = ruta[-1]
    prob = []
    for i in range(num_nodos):
        if i not in ruta:
            pheromone = feromona[actual, i] ** alpha
            visibility = (1 / distancias[actual, i]) ** beta
            prob.append(pheromone * visibility)
        else:
            prob.append(0)
    prob_sum = sum(prob)
    return [p / prob_sum for p in prob]

# Función para seleccionar el siguiente nodo basado en probabilidades
def seleccionar_nodo(probabilidades):
    return np.random.choice(range(len(probabilidades)), p=probabilidades)

# Función para calcular la distancia total de una ruta
def calcular_distancia(ruta, distancias):
    distancia = 0
    for i in range(len(ruta) - 1):
        distancia += distancias[ruta[i], ruta[i + 1]]
    distancia += distancias[ruta[-1], ruta[0]]  # Regreso al nodo inicial
    return distancia

# Función para graficar la mejor ruta
def graficar_ruta(mejor_ruta, coordenadas):
    # Obtener las coordenadas de la ruta
    ruta_coordenadas = coordenadas[mejor_ruta]
    
    # Configuración de la figura
    plt.figure(figsize=(8, 6))
    
    # Graficar las líneas entre los nodos
    plt.plot(ruta_coordenadas[:, 0], ruta_coordenadas[:, 1], marker='o', linestyle='-', color='b', label='Ruta')
    
    # Añadir etiquetas a los nodos
    for idx, (x, y) in enumerate(ruta_coordenadas):
        plt.text(x, y, f'{mejor_ruta[idx]}', fontsize=10, color='red', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='red', boxstyle='circle'))
    
    # Personalización del gráfico
    plt.title("Ruta Óptima Encontrada")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    
    # Mostrar la gráfica
    plt.show()


# Función para ejecutar el algoritmo utilizando múltiples núcleos
def ejecutar_en_multiproceso(distancias, coordenadas, num_hormigas, num_iteraciones, num_nodos, alpha, beta, rho, Q, num_nucleos):
    iteraciones_por_nucleo = num_iteraciones // num_nucleos
    args = [(distancias, coordenadas, num_hormigas, iteraciones_por_nucleo, num_nodos, alpha, beta, rho, Q) for _ in range(num_nucleos)]
    
    with Pool(num_nucleos) as pool:
        resultados = pool.starmap(colonia_hormigas, args)

    # Obtener el mejor resultado de todos los núcleos
    mejor_ruta, mejor_distancia = min(resultados, key=lambda x: x[1])
    return mejor_ruta, mejor_distancia

# Asegurarse de que el código solo se ejecute cuando no sea importado como un módulo
if __name__ == "__main__":
    # Cargar datos de los archivos
    distancias, coordenadas = cargar_datos('Cordenadas\Dist4.txt', 'Cordenadas\Coord4.txt')

    # Parámetros
    num_hormigas = 200
    num_iteraciones = 90
    num_nodos = len(distancias)
    alpha = 1    # Influencia de la feromona
    beta = 3.3     # Influencia de la visibilidad (1/distance)
    rho = 0.7    # Factor de evaporación
    Q = 1.2        # Cantidad de feromona depositada por las hormigas
    num_nucleos = 7  # Usar todos los núcleos disponibles

    # Ejecutar el algoritmo con múltiples núcleos
    mejor_ruta, mejor_distancia = ejecutar_en_multiproceso(distancias, coordenadas, num_hormigas, num_iteraciones, num_nodos, alpha, beta, rho, Q, num_nucleos)
    if mejor_ruta[0] != mejor_ruta[len(mejor_ruta)-1]:
        mejor_ruta.append(mejor_ruta[0])
    # Mostrar la mejor ruta y distancia
    print(f"Mejor Ruta: {[int(nodo) for nodo in mejor_ruta]}")
    print(f"Distancia de la Mejor Ruta: {mejor_distancia}")

    # Graficar la mejor ruta
    graficar_ruta(mejor_ruta, coordenadas)
