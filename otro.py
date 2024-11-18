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

class RecocidoSimulado:
    def __init__(self, distancias, temperatura_inicial=1000, enfriamiento=0.995, iteraciones_por_temperatura=100):
        self.distancias = distancias
        self.num_nodos = len(distancias)
        self.temperatura_inicial = temperatura_inicial
        self.enfriamiento = enfriamiento
        self.iteraciones_por_temperatura = iteraciones_por_temperatura

    def calcular_costo(self, ruta):
        # Calcula el costo de la ruta incluyendo el regreso al nodo inicial
        costo = sum(self.distancias[ruta[i]][ruta[i + 1]] for i in range(len(ruta) - 1))
        costo += self.distancias[ruta[-1]][ruta[0]]  # Regreso al nodo inicial
        return costo

    def generar_vecino(self, ruta):
        # Genera un vecino mediante intercambio de dos nodos
        i, j = random.sample(range(len(ruta)), 2)
        nueva_ruta = ruta[:]
        nueva_ruta[i], nueva_ruta[j] = nueva_ruta[j], nueva_ruta[i]
        return nueva_ruta

    def ejecutar(self):
        # Inicialización
        temperatura = self.temperatura_inicial
        ruta_actual = random.sample(range(self.num_nodos), self.num_nodos)
        costo_actual = self.calcular_costo(ruta_actual)
        mejor_ruta = ruta_actual[:]
        menor_costo = costo_actual

        while temperatura > 1:
            for _ in range(self.iteraciones_por_temperatura):
                # Generar un vecino y calcular su costo
                vecino = self.generar_vecino(ruta_actual)
                costo_vecino = self.calcular_costo(vecino)

                # Calcular probabilidad de aceptación
                delta = costo_vecino - costo_actual
                if delta < 0 or random.random() < np.exp(-delta / temperatura):
                    ruta_actual = vecino
                    costo_actual = costo_vecino

                # Actualizar la mejor solución encontrada
                if costo_actual < menor_costo:
                    mejor_ruta = ruta_actual[:]
                    menor_costo = costo_actual

            # Reducir la temperatura
            temperatura *= self.enfriamiento

        return mejor_ruta, menor_costo


# Leer datos
coordenadas = leer_coordenadas('Cordenadas\Coord1.txt')
distancias = leer_distancias('Cordenadas\Dist1.txt')

# Ejecutar recocido simulado
sa = RecocidoSimulado(distancias, temperatura_inicial=1000, enfriamiento=0.995, iteraciones_por_temperatura=500)
mejor_ruta, menor_costo = sa.ejecutar()

print("Mejor ruta encontrada (SA):", mejor_ruta)
print("Costo total (SA):", menor_costo)

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

graficar_ruta(coordenadas, mejor_ruta, titulo="Ruta óptima encontrada por Recocido Simulado")
