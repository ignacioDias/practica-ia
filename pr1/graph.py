# ---
# 📗 Graph.py — Clase auxiliar para representar un grafo con pesos de costo y tiempo
# ---

import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, nodes):
        """Inicializa un grafo con los nodos dados."""
        self.nodes = nodes
        self.graph = {node: {} for node in nodes}

    def add_path(self, node1, node2, cost, time):
        """Agrega una arista bidireccional con atributos de costo y tiempo."""
        self.graph[node1][node2] = {'cost': cost, 'time': time}
        self.graph[node2][node1] = {'cost': cost, 'time': time}

    def get_cost(self, origin, destination):
        """Devuelve el costo entre dos nodos, o infinito si no hay conexión."""
        return self.graph.get(origin, {}).get(destination, {}).get('cost', float('inf'))

    def get_time(self, origin, destination):
        """Devuelve el tiempo entre dos nodos, o infinito si no hay conexión."""
        return self.graph.get(origin, {}).get(destination, {}).get('time', float('inf'))

    def calculate_cost_and_time(self, path):
        """
        Calcula el costo y tiempo totales para una ruta (lista de nodos).
        Recorre cada par consecutivo de la secuencia.
        """
        total_cost, total_time = 0, 0
        for i in range(len(path) - 1):
            total_cost += self.get_cost(path[i], path[i + 1])
            total_time += self.get_time(path[i], path[i + 1])
        return total_cost, total_time

    def show(self):
        """Visualiza el grafo con etiquetas de costo/tiempo usando NetworkX."""
        G = nx.Graph()

        # Agregar todas las aristas con sus atributos
        for node, edges in self.graph.items():
            for vecino, data in edges.items():
                cost = data['cost']
                time = data['time']
                G.add_edge(node, vecino, label=f"{cost}/{time}")

        # Posiciones para los nodos
        pos = nx.spring_layout(G, seed=42)
        labels = nx.get_edge_attributes(G, 'label')

        # Dibujar nodos y aristas
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=1500, font_size=12, font_weight='bold', edge_color='gray')

        # Dibujar etiquetas de costo/tiempo
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')

        plt.title("Grafo con Costo/Tiempo por ruta")
        plt.show()
