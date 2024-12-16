import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import numpy as np
import networkx as nx
from controllers.ProbabilityTransitionController import (
    probabilityTransitionTable,
    graphProbability,
)
from controllers.Helpers import (
    cs_to_array,
    ns_to_array,
    reorder_cross_product,
    build_probabilities,
)

# Añadir al inicio de CutController.py
umbral_aceptable = 0.1  # Valor de EMD considerado aceptable
ventana_tamano = 5  # Tamaño de la ventana deslizante


def dfs(G, node, visited, end_node):
    visited.add(node)
    for neighbor in G.neighbors(node):
        if neighbor not in visited:
            dfs(G, neighbor, visited, end_node)

        if end_node in visited:
            return


def is_bipartite(G, start_node, end_node):
    visited = set()
    dfs(G, start_node, visited, end_node)

    return end_node not in visited


def cut_process(ns, cs, cs_value, probabilities, states):
    G = nx.DiGraph()
    add_connections(G, ns, cs)
    print("Grafo principal")
    draw_graph(G)

    nodes = list(G.nodes())
    start_node = nodes[0]
    if is_bipartite(G, start_node, start_node):
        print("El grafo ya está particionado, el proceso termina")
        return

    min_partition = {
        "partitioned_system": [],
        "partition": "",
        "edge_to_remove_1": "",
        "edge_to_remove_2": "",
        "emd": 0,
    }

    start_process(G, ns, cs, cs_value, min_partition, probabilities, states)

    print("**********************************************************")

    edge_to_remove_1 = min_partition["edge_to_remove_1"]
    edge_to_remove_2 = min_partition["edge_to_remove_2"]

    if min_partition["emd"] > 0 and edge_to_remove_1 != "" and edge_to_remove_2 != "":
        G.remove_edge(edge_to_remove_1, edge_to_remove_2)
        G.remove_edge(edge_to_remove_2, edge_to_remove_1)

    print("Grafo final")
    draw_graph(G)

    if min_partition["partition"]:
        print("Minima partición")
        print("emd", min_partition["emd"])
        print("partition", min_partition["partition"])
        graphProbability(
            min_partition["partitioned_system"], "blue", min_partition["partition"]
        )


def start_process(G, ns, cs, cs_value, min_partition, probabilities, states):
    memory = {}
    probabilities = build_probabilities(probabilities, len(states))
    original_system = probabilityTransitionTable(
        cs_to_array(cs, cs_value), ns_to_array(ns), probabilities, states
    )

    for i in range(len(ns)):
        nsN = ns[i] + "ᵗ⁺¹"
        for j in range(len(cs)):

            if ns[i] == cs[j]:
                continue

            csC = cs[j] + "ᵗ"

            print("Variable actual", nsN)
            print("cortando", csC, "de", nsN)

            cs_left_cut = cs[:j]
            cs_right_cut = cs[j + 1 :]
            cs_right_partition = cs_left_cut + cs_right_cut

            partition = f"(∅ᵗ⁺¹ | {csC}ᵗ) y ({ns}ᵗ⁺¹ | {cs_right_partition}ᵗ)"
            print("partition: ", partition)

            G.remove_edge(csC, nsN)
            G.remove_edge(nsN, csC)
            draw_graph(G)

            arr1 = np.array(cut("", csC, cs_value, memory, probabilities, states))
            arr2 = np.array(
                cut(ns, cs_right_partition, cs_value, memory, probabilities, states)
            )

            partitioned_system = []

            if len(arr1) > 0 and len(arr2) > 0:
                cross_product = np.kron(arr1, arr2)
                partitioned_system = reorder_cross_product(cross_product)
            elif len(arr1) > 0:
                partitioned_system = arr1
            elif len(arr2) > 0:
                partitioned_system = arr2

            # Calcular la Distancia de Wasserstein (EMD)
            emd_distance = wasserstein_distance(original_system, partitioned_system)
            print(f"Earth Mover's Distance: {emd_distance}")

            start_node = csC
            end_node = nsN
            if is_bipartite(G, start_node, end_node):
                print("Bipartición generada")
                if min_partition.get("partition") == "":
                    set_min_partition(
                        min_partition,
                        partition,
                        partitioned_system,
                        emd_distance,
                        csC,
                        nsN,
                    )

                if emd_distance == 0:
                    set_min_partition(
                        min_partition,
                        partition,
                        partitioned_system,
                        emd_distance,
                        csC,
                        nsN,
                    )
                    print("minima partición alcanzada")
                    return

                elif emd_distance <= min_partition.get("emd"):
                    set_min_partition(
                        min_partition,
                        partition,
                        partitioned_system,
                        emd_distance,
                        csC,
                        nsN,
                    )
                    print("minima partición actualizada")

                G.add_edge(csC, nsN)
                G.add_edge(nsN, csC)
                print("bipartición restarurada con costo:", emd_distance)
            else:
                print("No bipartición generada")
                if emd_distance > 0:
                    G.add_edge(csC, nsN)
                    G.add_edge(nsN, csC)
                    print("conexión restarurada con costo: ", emd_distance)
                else:
                    print("conexión elimina sin perdida de información")

            print("----------   ********** ------------")


def set_min_partition(
    min_partition,
    partition,
    partitioned_system,
    emd_distance,
    edge_to_remove_1,
    edge_to_remove_2,
):
    min_partition["partition"] = partition
    min_partition["partitioned_system"] = partitioned_system
    min_partition["emd"] = emd_distance
    min_partition["edge_to_remove_1"] = edge_to_remove_1
    min_partition["edge_to_remove_2"] = edge_to_remove_2


def cut(ns, cs, cs_value, memory, probabilities, states):
    if memory.get(cs) is not None and memory.get(cs).get(ns) is not None:
        if any(memory.get(cs).get(ns)):
            return memory.get(cs).get(ns)

    if len(ns) == 1:
        value = probabilityTransitionTable(
            cs_to_array(cs, cs_value), ns_to_array(ns), probabilities, states
        )
        return value

    value = []
    for i in range(0, len(ns)):
        if len(value) > 0:
            cross_product = np.kron(
                value, cut(ns[i], cs, cs_value, memory, probabilities, states)
            )
            value = reorder_cross_product(cross_product)

        else:
            value = np.array(cut(ns[i], cs, cs_value, memory, probabilities, states))

            if memory.get(cs) == None:
                memory[cs] = {}

            memory[cs][ns[i]] = value

    return value


def add_connections(G, ns, cs):
    for i in range(len(ns)):
        n = ns[i] + "ᵗ⁺¹"

        for j in range(len(cs)):
            if ns[i] == cs[j]:
                continue

            c = cs[j] + "ᵗ"
            G.add_node(n)
            G.add_node(c)
            G.add_edge(c, n)
            G.add_edge(n, c)


def draw_graph(G):
    pos = nx.circular_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=700,
        node_color="skyblue",
        font_size=10,
        font_color="black",
        arrowsize=20,
    )
    plt.show()

def calcularPesoConexion(origen, destino, G, probabilities, states):
    """Calcula el peso de una conexión basado en su criticidad"""
    try:
        # Obtener el índice correcto removiendo los símbolos de tiempo
        origen_idx = ord(origen.replace("ᵗ", "").replace("⁺¹", "")[0]) - 65
        destino_idx = ord(destino.replace("ᵗ", "").replace("⁺¹", "")[0]) - 65
        
        # Calcular frecuencia de transición
        frecuencia = sum(1 for state in states if origen_idx < len(state) and state[origen_idx] == 1)
        
        # Calcular impacto en distribución
        impacto = 0
        for prob_row in probabilities:
            if destino_idx < len(prob_row) and prob_row[destino_idx] > 0:
                impacto += prob_row[destino_idx]
        
        return frecuencia * impacto
    except Exception as e:
        print(f"Error calculando peso para {origen}-{destino}: {e}")
        return float('inf')  # Retornar un peso alto para conexiones problemáticas

def ordenarConexionesPorCriticidad(G, probabilities, states):
    """Ordena las conexiones por su peso de criticidad"""
    conexiones_peso = []
    for edge in G.edges():
        peso = calcularPesoConexion(edge[0], edge[1], G, probabilities, states)
        conexiones_peso.append((edge, peso))
    
    return sorted(conexiones_peso, key=lambda x: x[1])

def evaluate_partition(G, ns, cs, cs_value, probabilities, states):
    """Evalúa una partición del grafo y retorna el sistema particionado"""
    memory = {}
    original_system = probabilityTransitionTable(
        cs_to_array(cs, cs_value), ns_to_array(ns), probabilities, states
    )
    
    # Obtener componentes después del corte
    componentes = list(nx.connected_components(G.to_undirected()))
    if len(componentes) != 2:  # Si no es una bipartición válida
        return None
        
    # Separar los nodos en tiempo t y t+1
    nodos_t = [n for n in G.nodes() if "ᵗ" in n]
    nodos_t1 = [n for n in G.nodes() if "ᵗ⁺¹" in n]
    
    try:
        # Calcular distribuciones para cada componente
        arr1 = np.array(cut("", nodos_t[0].replace("ᵗ", ""), 
                           cs_value, memory, probabilities, states))
        arr2 = np.array(cut(ns, nodos_t[1].replace("ᵗ", ""), 
                           cs_value, memory, probabilities, states))
        
        # Combinar las distribuciones
        if len(arr1) > 0 and len(arr2) > 0:
            cross_product = np.kron(arr1, arr2)
            return reorder_cross_product(cross_product)
        elif len(arr1) > 0:
            return arr1
        elif len(arr2) > 0:
            return arr2
        else:
            return None
            
    except Exception as e:
        print(f"Error evaluando partición: {e}")
        return None

def cut_process_with_heuristic(ns, cs, cs_value, probabilities, states):
    G = nx.DiGraph()
    add_connections(G, ns, cs)
    print("Grafo principal")
    draw_graph(G)

    if is_bipartite(G, list(G.nodes())[0], list(G.nodes())[0]):
        print("El grafo ya está particionado")
        return

    # Obtener sistema original
    original_system = probabilityTransitionTable(
        cs_to_array(cs, cs_value), ns_to_array(ns), probabilities, states
    )

    # Obtener y ordenar conexiones existentes
    edges = [(u, v) for u, v in G.edges() if G.has_edge(u, v)]
    conexiones_ordenadas = []
    for edge in edges:
        peso = calcularPesoConexion(edge[0], edge[1], G, probabilities, states)
        conexiones_ordenadas.append((edge, peso))
    
    conexiones_ordenadas.sort(key=lambda x: x[1])
    
    min_partition = {
        "partitioned_system": [],
        "partition": "",
        "edge_to_remove_1": "",
        "edge_to_remove_2": "",
        "emd": float('inf')
    }

    # Procesar conexiones ordenadas
    for edge, peso in conexiones_ordenadas:
        if G.has_edge(edge[0], edge[1]):
            G.remove_edge(edge[0], edge[1])
            G.remove_edge(edge[1], edge[0])
            
            # Evaluar partición
            try:
                partitioned_system = evaluate_partition(G, ns, cs, cs_value, probabilities, states)
                if partitioned_system is not None:
                    emd_distance = wasserstein_distance(original_system, partitioned_system)
                    
                    if emd_distance < min_partition["emd"]:
                        set_min_partition(
                            min_partition,
                            f"Partition at {edge}",
                            partitioned_system,
                            emd_distance,
                            edge[0],
                            edge[1]
                        )
                        
                        if emd_distance == 0:
                            break
            except Exception as e:
                print(f"Error evaluando partición: {e}")
            
            # Restaurar si no es bipartición válida
            if not is_bipartite(G, edge[0], edge[1]):
                G.add_edge(edge[0], edge[1])
                G.add_edge(edge[1], edge[0])

    # Aplicar mejor partición encontrada
    if min_partition["edge_to_remove_1"] and min_partition["edge_to_remove_2"]:
        try:
            if G.has_edge(min_partition["edge_to_remove_1"], min_partition["edge_to_remove_2"]):
                G.remove_edge(min_partition["edge_to_remove_1"], min_partition["edge_to_remove_2"])
                G.remove_edge(min_partition["edge_to_remove_2"], min_partition["edge_to_remove_1"])
        except Exception as e:
            print(f"Error aplicando la partición final: {e}")

    print("Grafo final")
    draw_graph(G)

    if min_partition["partition"]:
        print("Minima partición")
        print("emd", min_partition["emd"])
        print("partition", min_partition["partition"])
        graphProbability(
            min_partition["partitioned_system"], 
            "blue", 
            min_partition["partition"]
        )