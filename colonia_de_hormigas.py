import numpy as np
import random
import matplotlib.pyplot as plt


# Leer archivo de coordenadas
def leer_coordenadas(file_path):
    with open('Cordenadas/Coord1.txt', 'r') as file:
        coordenadas = [tuple(map(float, line.strip().split())) for line in file]
    return coordenadas

# Leer archivo de distancias
def leer_distancias(file_path):
    with open('Cordenadas/Dist1.txt', 'r') as file:
        distancias = [list(map(float, line.strip().split())) for line in file]
    return np.array(distancias)

class ColoniaHormigas:
    def __init__(self, num_hormigas, num_iteraciones, alpha, beta, rho, distancias):
        self.num_hormigas = num_hormigas
        self.num_iteraciones = num_iteraciones
        self.alpha = alpha  # Influencia de las feromonas
        self.beta = beta    # Influencia de la visibilidad
        self.rho = rho      # Tasa de evaporación
        self.distancias = distancias
        self.num_nodos = len(distancias)
        self.feromonas = np.ones((self.num_nodos, self.num_nodos))  # Inicialización
    def calcular_probabilidades(self, nodo_actual, nodos_por_visitar):
        probabilidad = []
        for nodo in nodos_por_visitar:
            feromona = self.feromonas[nodo_actual][nodo] ** self.alpha
            visibilidad = (1 / self.distancias[nodo_actual][nodo]) ** self.beta
            probabilidad.append(feromona * visibilidad)
        total = sum(probabilidad)
        return [p / total for p in probabilidad]
    
    def actualizar_feromonas(self, rutas, costos):
        self.feromonas *= (1 - self.rho)  # Evaporación
        for ruta, costo in zip(rutas, costos):
            for i in range(len(ruta) - 1):
                self.feromonas[ruta[i]][ruta[i + 1]] += 1 / costo
            # Considerar el regreso al nodo inicial
            self.feromonas[ruta[-1]][ruta[0]] += 1 / costo
    def actualizar_feromonas(self, rutas, costos):
        self.feromonas *= (1 - self.rho)  # Evaporación
        for ruta, costo in zip(rutas, costos):
            for i in range(len(ruta) - 1):
                self.feromonas[ruta[i]][ruta[i + 1]] += 1 / costo
            # Considerar el regreso al nodo inicial
            self.feromonas[ruta[-1]][ruta[0]] += 1 / costo


    def generar_ruta(self, inicio):
        ruta = [inicio]
        nodos_por_visitar = set(range(self.num_nodos)) - {inicio}
        while nodos_por_visitar:
            nodo_actual = ruta[-1]
            probabilidades = self.calcular_probabilidades(nodo_actual, nodos_por_visitar)
            siguiente_nodo = random.choices(list(nodos_por_visitar), weights=probabilidades, k=1)[0]
            ruta.append(siguiente_nodo)
            nodos_por_visitar.remove(siguiente_nodo)
        ruta.append(inicio)  # Regresar al nodo inicial
        return ruta
    def ejecutar(self):
        mejor_ruta = None
        menor_costo = float('inf')

        for _ in range(self.num_iteraciones):
            rutas = []
            costos = []

            for _ in range(self.num_hormigas):
                inicio = random.randint(0, self.num_nodos - 1)
                ruta = self.generar_ruta(inicio)
                costo = sum(self.distancias[ruta[i]][ruta[i + 1]] for i in range(len(ruta) - 1))
                rutas.append(ruta)
                costos.append(costo)

                if costo < menor_costo:
                    mejor_ruta = ruta
                    menor_costo = costo

            self.actualizar_feromonas(rutas, costos)

        return mejor_ruta, menor_costo





coordenadas = leer_coordenadas('coordenadas.txt')
distancias = leer_distancias('distancias.txt')

aco = ColoniaHormigas(num_hormigas=10, num_iteraciones=100, alpha=1, beta=2, rho=0.1, distancias=distancias)
mejor_ruta, menor_costo = aco.ejecutar()

print("Mejor ruta:", mejor_ruta)
print("Menor costo:", menor_costo)

def vecino_mas_proximo(distancias, inicio=0):
    num_nodos = len(distancias)
    visitados = [inicio]
    costo_total = 0

    nodo_actual = inicio
    while len(visitados) < num_nodos:
        # Encuentra el vecino no visitado más cercano
        no_visitados = [i for i in range(num_nodos) if i not in visitados]
        siguiente_nodo = min(no_visitados, key=lambda nodo: distancias[nodo_actual][nodo])
        
        # Actualiza la ruta y el costo
        costo_total += distancias[nodo_actual][siguiente_nodo]
        visitados.append(siguiente_nodo)
        nodo_actual = siguiente_nodo

    # Regresa al nodo inicial
    costo_total += distancias[nodo_actual][inicio]
    visitados.append(inicio)
    
    return visitados, costo_total
# Leer datos
coordenadas = leer_coordenadas('coordenadas.txt')
distancias = leer_distancias('distancias.txt')

# Ejecutar el algoritmo
inicio = 0  # Nodo de inicio
ruta2, costo2 = vecino_mas_proximo(distancias, inicio=inicio)

print("Ruta encontrada:", ruta2)
print("Costo total:", costo2)



def graficar_ruta(coordenadas, ruta):
    x, y = zip(*[coordenadas[nodo] for nodo in ruta])
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    for i, (xi, yi) in enumerate(coordenadas):
        plt.text(xi, yi, f'{i}', fontsize=12, color='red')
    plt.title("Ruta del vecino más próximo")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

graficar_ruta(coordenadas, ruta2)
graficar_ruta(coordenadas, mejor_ruta)
