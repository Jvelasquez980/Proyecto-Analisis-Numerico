import numpy as np
import random
import matplotlib.pyplot as plt

# Leer archivo de coordenadas
def leer_coordenadas(file_path):
    with open(file_path, 'r') as file:
        coordenadas = [tuple(map(float, line.strip().split())) for line in file]
    return coordenadas

# Leer archivo de distancias
def leer_distancias(file_path):
    with open(file_path, 'r') as file:
        distancias = [list(map(float, line.strip().split())) for line in file]
    return np.array(distancias)

class AlgoritmoGenetico:
    def __init__(self, distancias, tam_poblacion=100, num_generaciones=500, prob_mutacion=0.1):
        self.distancias = distancias
        self.tam_poblacion = tam_poblacion
        self.num_generaciones = num_generaciones
        self.prob_mutacion = prob_mutacion
        self.num_nodos = len(distancias)

    def inicializar_poblacion(self):
        # Generar rutas aleatorias como población inicial
        poblacion = [random.sample(range(self.num_nodos), self.num_nodos) for _ in range(self.tam_poblacion)]
        return poblacion

    def evaluar_fitness(self, ruta):
        # Calcular el costo total de una ruta, incluyendo el regreso al nodo inicial
        costo = sum(self.distancias[ruta[i]][ruta[i + 1]] for i in range(len(ruta) - 1))
        costo += self.distancias[ruta[-1]][ruta[0]]
        return costo

    def seleccionar_padres(self, poblacion, fitness):
        # Selección por torneo: selecciona dos rutas aleatorias y elige la mejor
        padres = []
        for _ in range(len(poblacion)):
            competidores = random.sample(range(len(poblacion)), 2)
            ganador = competidores[0] if fitness[competidores[0]] < fitness[competidores[1]] else competidores[1]
            padres.append(poblacion[ganador])
        return padres

    def cruzar_rutas(self, padre1, padre2):
        # Cruce de orden (Order Crossover)
        start, end = sorted(random.sample(range(len(padre1)), 2))
        hijo = [None] * len(padre1)
        hijo[start:end + 1] = padre1[start:end + 1]
        for nodo in padre2:
            if nodo not in hijo:
                hijo[hijo.index(None)] = nodo
        return hijo

    def mutar_ruta(self, ruta):
        # Mutación por intercambio de dos nodos
        if random.random() < self.prob_mutacion:
            i, j = random.sample(range(len(ruta)), 2)
            ruta[i], ruta[j] = ruta[j], ruta[i]
        return ruta

    def ejecutar(self):
        poblacion = self.inicializar_poblacion()
        mejor_ruta = None
        menor_costo = float('inf')

        for _ in range(self.num_generaciones):
            fitness = [self.evaluar_fitness(ruta) for ruta in poblacion]
            nueva_poblacion = []

            # Seleccionar padres y generar nueva población
            padres = self.seleccionar_padres(poblacion, fitness)
            for i in range(0, len(padres), 2):
                padre1, padre2 = padres[i], padres[(i + 1) % len(padres)]
                hijo1 = self.cruzar_rutas(padre1, padre2)
                hijo2 = self.cruzar_rutas(padre2, padre1)
                nueva_poblacion.extend([hijo1, hijo2])

            # Aplicar mutaciones
            poblacion = [self.mutar_ruta(ruta) for ruta in nueva_poblacion]

            # Actualizar la mejor solución
            for ruta in poblacion:
                costo = self.evaluar_fitness(ruta)
                if costo < menor_costo:
                    mejor_ruta = ruta
                    menor_costo = costo

        return mejor_ruta, menor_costo


# Leer datos
coordenadas = leer_coordenadas('Cordenadas\Coord1.txt')
distancias = leer_distancias('Cordenadas\Dist1.txt')

# Ejecutar algoritmo genético
genetico = AlgoritmoGenetico(distancias, tam_poblacion=200, num_generaciones=1000, prob_mutacion=0.05)
mejor_ruta, menor_costo = genetico.ejecutar()

print("Mejor ruta encontrada (GA):", mejor_ruta)
print("Costo total (GA):", menor_costo)


# Graficar resultados
def graficar_ruta(coordenadas, ruta, titulo="Ruta"):
    x, y = zip(*[coordenadas[nodo] for nodo in ruta + [ruta[0]]])
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    for i, (xi, yi) in enumerate(coordenadas):
        plt.text(xi, yi, f'{i}', fontsize=12, color='red')
    plt.title(titulo)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


graficar_ruta(coordenadas, mejor_ruta, titulo="Ruta óptima encontrada por GA")
