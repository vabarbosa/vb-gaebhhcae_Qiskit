import numpy as np
import time
# Import Qiskit
from qiskit import QuantumCircuit
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_provider import IBMProvider
from qiskit.visualization import *
provider = IBMProvider('')

from qiskit_aer import AerSimulator

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import statistics

from qiskit import QuantumRegister, QuantumCircuit,Aer, execute,ClassicalRegister

def compute_cost(n, w, G):
    """
    Calcula el costo de cada posible configuración de un sistema dado representado por un grafo y retorna el costo máximo y su configuración correspondiente.

    Parámetros
    ----------
    n : int
        El número de nodos en el grafo.
    w : numpy.ndarray
        Matriz de pesos de las aristas en el grafo.
    G : networkx.classes.graph.Graph
        Un objeto de grafo de NetworkX.
    pos : dict
        Un diccionario que mapea cada nodo a su posición en el grafo.

    Devoluciones
    -----------
    cost_dict : dict
        Un diccionario que mapea cada posible configuración del sistema a su costo asociado.
    best_cost_brute : int
        El costo máximo encontrado entre todas las configuraciones posibles.

    Ejemplo
    -------
    >>> cost_dict, best_cost_brute = compute_cost(n, w, G, pos)
    >>> print(best_cost_brute)
    15
    >>> print(cost_dict)
    {(0, 0, 0): 0, (0, 0, 1): 5, ..., (1, 1, 1): 15}
    """
    best_cost_brute = 0
    cost_dict = {}  # Diccionario para almacenar los casos y sus costos

    for b in range(2**n):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
        cost = 0
        for i in range(n):
            for j in range(n):
                cost = cost + w[i, j] * x[i] * (1 - x[j])
        if best_cost_brute < cost:
            best_cost_brute = cost
            xbest_brute = x

        # Almacena el caso y el costo en el diccionario
        cost_dict[tuple(x)] = cost

    colors = ["r" if xbest_brute[i] == 0 else "c" for i in range(n)]
    #draw_graph(G, colors, pos)

    return cost_dict, best_cost_brute

def generate_hamiltonian_graph(num_nodes, seed=0):
    """
    Genera un grafo Hamiltoniano no dirigido con un número dado de nodos y aristas adicionales aleatorias.

    Un grafo Hamiltoniano es un grafo que tiene un ciclo Hamiltoniano (un ciclo que visita cada nodo una vez).
    Esta función primero crea un ciclo que conecta todos los nodos, y luego agrega aristas adicionales de manera aleatoria.

    Parámetros
    ----------
    num_nodes : int
        El número de nodos para incluir en el grafo.
    seed : int, optional
        La semilla para el generador de números aleatorios. Por defecto es 0.

    Devoluciones
    ------------
    G : networkx.classes.graph.Graph
        Un objeto de grafo no dirigido de NetworkX con 'num_nodes' nodos, un ciclo Hamiltoniano,
        y aristas adicionales que se añaden aleatoriamente.

    Ejemplo
    -------
    >>> G = generate_hamiltonian_graph(5)
    >>> print(G.edges(data=True))
    [(0, 1, {'weight': 1.0}), (0, 4, {'weight': 1.0}), (1, 2, {'weight': 1.0}),
    (2, 3, {'weight': 1.0}), (3, 4, {'weight': 1.0})]
    """

    np.random.seed(seed)  # Fija la semilla del generador de números aleatorios

    # Crea un nuevo grafo no dirigido
    G = nx.Graph()

    # Agrega los nodos al grafo
    G.add_nodes_from(range(num_nodes))

    # Genera un camino Hamiltoniano
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1, weight=1.0)
    G.add_edge(num_nodes - 1, 0, weight=1.0)  # Cierra el ciclo para hacerlo Hamiltoniano

    # Agrega más aristas de manera aleatoria
    for i in range(num_nodes):
        for j in range(i + 2, num_nodes):
            if np.random.random() > 0.5:  # Añade la arista con una probabilidad del 50%
                G.add_edge(i, j, weight=1.0)

    return G

def compute_weight_matrix(G, n):
    """
    Calcula la matriz de pesos a partir de un grafo dado.

    Parámetros
    ----------
    G : networkx.classes.graph.Graph
        Un objeto de grafo de NetworkX.
    n : int
        El número de nodos en el grafo.

    Devoluciones
    -----------
    w : numpy.ndarray
        Una matriz de 'n' por 'n' que representa los pesos de las aristas en el grafo.
        Cada elemento w[i, j] es el peso de la arista entre los nodos i y j.

    Ejemplo
    -------
    >>> w = compute_weight_matrix(G, 5)
    >>> print(w)
    [[0. 1. 1. 1. 1.]
     [1. 0. 1. 1. 1.]
     [1. 1. 0. 1. 1.]
     [1. 1. 1. 0. 1.]
     [1. 1. 1. 1. 0.]]
    """
    # Calculando la matriz de pesos a partir del grafo aleatorio
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = temp["weight"]
    return w

def draw_graph(G, colors, pos):
    """
    Dibuja un grafo de NetworkX con nodos de colores y etiquetas de aristas.

    Parámetros
    ----------
    G : networkx.classes.graph.Graph
        Un objeto de grafo de NetworkX.
    colors : list
        Una lista de colores para los nodos. Cada nodo se dibuja con el color correspondiente de la lista.
    pos : dict
        Un diccionario que mapea cada nodo a su posición en el grafo.

    Ejemplo
    -------
    >>> draw_graph(G, ['red', 'red', 'cyan', 'cyan', 'cyan'], pos)
    """
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "peso")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

def string_to_tuple(s):
    """
    Convierte una cadena de texto en una tupla, interpretando cada carácter como un número entero.

    Parámetros
    ----------
    s : str
        La cadena de texto a convertir. Se espera que cada carácter de la cadena sea un dígito decimal.

    Devoluciones
    -----------
    tuple
        Una tupla de enteros correspondiente a los caracteres de la cadena.

    Ejemplo
    -------
    >>> t = string_to_tuple("1234")
    >>> print(t)
    (1, 2, 3, 4)
    """
    return tuple(int(char) for char in s)

def get_random_number(qc, qubit, reg):
    """
    Genera un número aleatorio (0 o 1) utilizando un circuito cuántico.

    Parámetros
    ----------
    qc : qiskit.circuit.quantumcircuit.QuantumCircuit
        El circuito cuántico en el que se realizará la operación.
    qubit : qiskit.circuit.quantumregister.QuantumRegister
        El registro cuántico que contiene los qubits en los que se realizará la operación.
    reg : qiskit.circuit.classicalregister.ClassicalRegister
        El registro clásico que se utilizará para la medición.

    Devoluciones
    -----------
    first_digit : int
        Un número aleatorio (0 o 1) generado a partir de la medida del estado del qubit.

    Ejemplo
    -------
    >>> qc = QuantumCircuit(1, 1)
    >>> qubit = QuantumRegister(1)
    >>> reg = ClassicalRegister(1)
    >>> random_number = get_random_number(qc, qubit, reg)
    >>> print(random_number)
    0
    """
    qc.h(qubit[0])
    qc.measure(qubit[0], reg[0])

    with qc.if_test((reg, 1)):
        qc.x(qubit[0])

    backend_sim = AerSimulator()
    reset_sim_job = backend_sim.run(qc, shots = 1)
    reset_sim_result = reset_sim_job.result()
    counts = reset_sim_result.get_counts(qc)

    key = list(counts.keys())[0]
    first_digit = int(key[0])  # Convierte el primer carácter de la clave a un entero
    return first_digit

def mutate_bit(qr, qc, bit):
    """
    Aplica una operación NOT (X) a un bit específico de un registro cuántico.

    Parámetros
    ----------
    qr : QuantumRegister
        El registro cuántico en el que se realizará la mutación.
    qc : QuantumCircuit
        El circuito cuántico que contiene el registro.
    bit : int
        El índice del bit que se va a mutar.

    Devoluciones
    ------------
    No devuelve nada; la función modifica el circuito cuántico directamente.
    """
    #print("mutate_bit")
    qc.x(qr[bit])


def mutate_multi_bit(qr, qc, probabilidad):
    """
    Aplica una operación NOT (X) a cada bit en un registro cuántico con cierta probabilidad.

    Parámetros
    ----------
    qr : QuantumRegister
        El registro cuántico en el que se realizarán las mutaciones.
    qc : QuantumCircuit
        El circuito cuántico que contiene el registro.
    probabilidad : float
        La probabilidad de que cada bit sea mutado.

    Devoluciones
    ------------
    No devuelve nada; la función modifica el circuito cuántico directamente.
    """
    for i in range(len(qr)):
        if np.random.random() > probabilidad:
            #print("multibit: ",i)
            mutate_bit(qr, qc, i)


def mutate_bit_random(qr, qc, probabilidad):#prob >0 and prob <1 a mas grande prob menos oportunidades de mutar
    """
    Aplica una operación NOT (X) a un bit aleatorio en un registro cuántico con cierta probabilidad.

    Parámetros
    ----------
    qr : QuantumRegister
        El registro cuántico en el que se realizará la mutación.
    qc : QuantumCircuit
        El circuito cuántico que contiene el registro.
    probabilidad : float
        La probabilidad de que el bit sea mutado.

    Devoluciones
    ------------
    No devuelve nada; la función modifica el circuito cuántico directamente.
    """

    if np.random.random() > probabilidad:
        #print("bit_random")
        qc.x(qr[np.random.randint(0, len(qr))])


def mutate_exchange(qr, qc, probabilidad):
    """
    Intercambia dos bits aleatorios en un registro cuántico con cierta probabilidad.

    Parámetros
    ----------
    qr : QuantumRegister
        El registro cuántico en el que se realizará el intercambio.
    qc : QuantumCircuit
        El circuito cuántico que contiene el registro.
    probabilidad : float
        La probabilidad de que se realice el intercambio.

    Devoluciones
    ------------
    No devuelve nada; la función modifica el circuito cuántico directamente.
    """
    if np.random.random() > probabilidad:
        #print("exchange")
        numero0 = np.random.randint(0, len(qr))
        numero1 = np.random.randint(0, len(qr))
        if numero0 != numero1:
            qc.swap(qr[numero0], qr[numero1])

def puntos(qr):
    """
    Genera dos números enteros aleatorios diferentes dentro del rango del tamaño del registro cuántico.

    Parámetros
    ----------
    qr : QuantumRegister
        El registro cuántico que se utiliza para determinar el rango de los números generados.

    Devoluciones
    ------------
    tuple
        Una tupla de dos enteros. El primer elemento de la tupla es siempre menor que el segundo.

    Ejemplo
    -------
    >>> qr = QuantumRegister(5)
    >>> p = puntos(qr)
    >>> print(p)
    (2, 4)  # Los valores exactos pueden variar debido a la generación aleatoria
    """
    numero = np.random.randint(1, len(qr) - 1)
    num = np.random.randint(1, len(qr) - 1)
    while numero == num:
        num = np.random.randint(1, len(qr) - 1)
    if num > numero:
        return numero, num
    if num <= numero:
        return num, numero

def escribir(qc, qreg, bin_string):
    """
    Escribe o 'resetea' un registro cuántico, basándose en una cadena de texto binaria.

    Parámetros
    ----------
    qc : QuantumCircuit
        El circuito cuántico que contiene el registro.
    qreg : QuantumRegister
        El registro cuántico que se va a escribir o resetear.
    bin_string : str
        La cadena de texto binaria que se utilizará para escribir o resetear el registro. Si se quiere escribir un
        número específico, esta cadena debe representar el número en binario. Si se quiere resetear el registro, esta
        cadena debe ser la representación binaria actual del estado del registro cuántico.

    Devoluciones
    ------------
    No devuelve nada; la función modifica el circuito cuántico directamente.

    Lanza
    -----
    ValueError
        Si el tamaño del QuantumRegister no es suficientemente grande para el string binario.

    Ejemplo
    -------
    >>> qr = QuantumRegister(5)
    >>> qc = QuantumCircuit(qr)
    >>> escribir(qc, qr, '10101')
    """
    if qreg.size < len(bin_string):
        raise ValueError("El QuantumRegister no es suficientemente grande para el string binario")

    bin_string = bin_string[::-1]

    for i, bit in enumerate(bin_string):
        if bit == '1':
            qc.x(i)

def create_qc(number_quantum_reg,qubits,number_classic_reg,bits):
    """facilita el crear circuitos cuanticos si quieres poner ancillas aconesjo hacerlo manualmente partiendo desde aqui"""
    # Create quantum registers
    qregs = [QuantumRegister(qubits, name=f'q{i}') for i in range(number_quantum_reg)]
    ancilla = QuantumRegister(1, name='ancilla')
    # Create classical registers
    cregs = [ClassicalRegister(bits, name=f'c{i}') for i in range(number_classic_reg)]
    cr_ancilla = ClassicalRegister(1, name='cr_ancilla')
    # Create a quantum circuit
    return QuantumCircuit(*qregs, ancilla, *cregs, cr_ancilla)

def lanzar_dinamico(circuito, backend,shots):
    """
    Ejecuta un circuito cuántico en un backend específico y devuelve los conteos de resultados.

    Parámetros
    ----------
    circuito : QuantumCircuit
        El circuito cuántico que se va a ejecutar.
    backend : Backend
        El backend de Qiskit en el que se ejecutará el circuito.

    Devoluciones
    ------------
    dict
        Un diccionario que contiene los conteos de resultados de la ejecución del circuito. Las claves son las
        cadenas de texto binarias que representan los resultados, y los valores son los números de veces que cada
        resultado ocurrió.

    Ejemplo
    -------
    >>> qc = QuantumCircuit(1)
    >>> qc.h(0)
    >>> backend = Aer.get_backend('qasm_simulator')
    >>> counts = lanzar_dinamico(qc, backend)
    >>> print(counts)
    {'0': 512, '1': 512}
    """
    backend_sim = backend
    #tp_circuit = transpile(circuito, backend=backend, optimization_level=0)
    tp_c = transpile(circuito,backend)
    reset_sim_job = backend_sim.run(tp_c, shots=shots, dynamic=True)
    reset_sim_result = reset_sim_job.result()#esto es para NO ruido
    return reset_sim_result.get_counts()
    """result_noise = sim_vigo.run(tcirc).result()#RUIDO
    return result_noise.get_counts(0)"""

def valores_hijos(resultados):#se supone que solo tenemos dos creg, es decir dos hijos
    """
    Extrae dos cadenas de texto binarias de los resultados de una ejecución de un circuito cuántico.

    Parámetros
    ----------
    resultados : dict
        El diccionario que contiene los conteos de resultados de la ejecución del circuito. Se espera que las claves
        sean cadenas de texto binarias separadas por espacios.

    Devoluciones
    ------------
    tuple
        Una tupla de dos tuplas de enteros. Cada tupla de enteros corresponde a los dígitos de una de las cadenas de
        texto binarias.

    Ejemplo
    -------
    >>> resultados = {'1 101 011': 1}
    >>> valores = valores_hijos(resultados)
    >>> print(valores)
    ((0, 1, 1), (1, 0, 1))
    """
    for key in resultados.keys():
        return string_to_tuple(key.split(' ')[2]), string_to_tuple(key.split(' ')[1])

def tuple_to_string(t):
    """
    Convierte una tupla de enteros en una cadena de texto, interpretando cada número entero como un carácter.

    Parámetros
    ----------
    t : tuple
        La tupla de enteros a convertir. Se espera que cada elemento de la tupla sea un número entero.

    Devoluciones
    ------------
    str
        Una cadena de texto correspondiente a los números de la tupla.

    Ejemplo
    -------
    >>> s = tuple_to_string((1, 2, 3, 4))
    >>> print(s)
    '1234'
    """
    return ''.join(str(num) for num in t)

def reset(qc, qr, cr_ancilla):
    """
    Resetea todos los qubits en un registro cuántico, midiendo cada qubit y luego aplicando una compuerta X si el resultado de la medición es 1.

    Parámetros
    ----------
    qc : QuantumCircuit
        El circuito cuántico donde se realizará el reseteo.

    qr : QuantumRegister
        El registro cuántico que contiene los qubits a resetear.

    cr_ancilla : ClassicalRegister
        El registro clásico auxiliar utilizado para almacenar los resultados de las mediciones.

    Ejemplo
    -------
    >>> from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
    >>> qr = QuantumRegister(3)
    >>> cr_ancilla = ClassicalRegister(1)
    >>> qc = QuantumCircuit(qr, cr_ancilla)
    >>> reset(qc, qr, cr_ancilla)
    """
    for qubit in range(0, qr.size):
        qc.measure(qr[qubit], cr_ancilla)
        with qc.if_test((cr_ancilla, 1)):
            qc.x(qr[qubit])

from qiskit import IBMQ
#provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
#backend = provider.get_backend('ibm_sherbrooke')
from qiskit.providers.fake_provider import FakeSherbrooke
backend = AerSimulator()
#backend = FakeManilaV2()
#backend = provider.get_backend('ibm_nairobi')
#backend = provider.get_backend('ibmq_kolkata')

def generate_random_number(n,backend):
    """Generate a random number in range [0, 2^n) using Qiskit."""
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.measure_all()

    #backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=1, memory=True).result()
    random_number = int(result.get_memory()[0], 2)  # Convert from binary to decimal
    return random_number

def generate_random_crossover_points(max_value,backend):
    """
    Genera dos números aleatorios distintos menores que max_value.

    Parámetros:
    max_value: El límite superior para los números generados.

    Retorna:
    Una tupla de dos números enteros distintos en orden ascendente.
    """

    # Asegurarse de que max_value es un número entero positivo
    assert isinstance(max_value, int) and max_value > 0, 'max_value debe ser un entero positivo'

    # Determinar cuántos bits se necesitan para representar max_value en binario
    num_bits = int(np.ceil(np.log2(max_value)))

    # Generar el primer número, reintentar si es mayor o igual a max_value
    num0 = generate_random_number(num_bits,backend)
    while num0 >= max_value:
        num0 = generate_random_number(num_bits,backend)

    # Generar el segundo número, reintentar si es igual al primero o es mayor o igual a max_value
    num1 = generate_random_number(num_bits,backend)
    while num1 == num0 or num1 >= max_value:
        num1 = generate_random_number(num_bits,backend)

    # Devolver los números en orden ascendente
    return min(num0, num1), max(num0, num1)

def crear_hijo(base, hasta, maximo, qreg_padre, qreg_madre, qreg_hijo, qc, qreg_ancilla, cr_ancilla):
    """
    Crea un nuevo registro cuántico hijo a partir de dos registros cuánticos padres utilizando el algoritmo genético.

    Parámetros
    ----------
    base : int
        El índice en el que comienza la sección intermedia del hijo.
    hasta : int
        El índice en el que termina la sección intermedia del hijo.
    maximo : int
        El tamaño total del registro cuántico.
    qreg_padre : QuantumRegister
        El registro cuántico del primer progenitor.
    qreg_madre : QuantumRegister
        El registro cuántico del segundo progenitor.
    qreg_hijo : QuantumRegister
        El registro cuántico del hijo que se va a crear.
    qc : QuantumCircuit
        El circuito cuántico en el que se realizarán las operaciones.
    qreg_ancilla : QuantumRegister
        El registro cuántico de ancilla utilizado para las operaciones.
    cr_ancilla : ClassicalRegister
        El registro clásico de ancilla utilizado para las operaciones.

    Devoluciones
    ------------
    No devuelve nada; la función modifica el QuantumCircuit directamente.

    Ejemplo
    -------
    >>> qr_padre = QuantumRegister(5, 'padre')
    >>> qr_madre = QuantumRegister(5, 'madre')
    >>> qr_hijo = QuantumRegister(5, 'hijo')
    >>> qc = QuantumCircuit(qr_padre, qr_madre, qr_hijo)
    >>> qr_ancilla = QuantumRegister(1, 'ancilla')
    >>> cr_ancilla = ClassicalRegister(1, 'cr_ancilla')
    >>> qc.add_register(qr_ancilla, cr_ancilla)
    >>> base = 1
    >>> hasta = 3
    >>> maximo = 5
    >>> crear_hijo(base, hasta, maximo, qr_padre, qr_madre, qr_hijo, qc, qr_ancilla, cr_ancilla)
    """

    #if get_random_number(qc, qreg_ancilla, cr_ancilla) == 0:#se genera un numero aleatorio que indicara si se usa al padre o a la madre

    if np.random.randint(0,2) ==0:#del 0 al 1 random
        padre_hijo(qc, qreg_padre, qreg_hijo, qreg_ancilla, cr_ancilla, 0, base)
    if np.random.randint(0,2) ==1:
        padre_hijo(qc, qreg_madre, qreg_hijo, qreg_ancilla, cr_ancilla, 0, base)

    padre_hijo(qc, qreg_padre, qreg_hijo, qreg_ancilla, cr_ancilla, base, hasta)#medio
    padre_hijo(qc, qreg_madre, qreg_hijo, qreg_ancilla, cr_ancilla, hasta, maximo)#fin

def padre_hijo(circuit, qreg_padre, qreg_hijo, qreg_ancilla, cr_ancilla, base, hasta):
    """
    Realiza el cruce entre el registro cuántico de un progenitor y el registro cuántico de un hijo.

    Parámetros
    ----------
    circuit : QuantumCircuit
        El circuito cuántico en el que se realizarán las operaciones.
    qreg_padre : QuantumRegister
        El registro cuántico del progenitor.
    qreg_hijo : QuantumRegister
        El registro cuántico del hijo.
    qreg_ancilla : QuantumRegister
        El registro cuántico de ancilla utilizado para las operaciones.
    cr_ancilla : ClassicalRegister
        El registro clásico de ancilla utilizado para las operaciones.
    base : int
        El índice en el que comienza la sección a cruzar.
    hasta : int
        El índice en el que termina la sección a cruzar.

    Devoluciones
    ------------
    No devuelve nada; la función modifica el QuantumCircuit directamente.

    Ejemplo
    -------
    >>> qr_padre = QuantumRegister(5, 'padre')
    >>> qr_hijo = QuantumRegister(5, 'hijo')
    >>> qc = QuantumCircuit(qr_padre, qr_hijo)
    >>> qr_ancilla = QuantumRegister(1, 'ancilla')
    >>> cr_ancilla = ClassicalRegister(1, 'cr_ancilla')
    >>> qc.add_register(qr_ancilla, cr_ancilla)
    >>> base = 1
    >>> hasta = 3
    >>> padre_hijo(qc, qr_padre, qr_hijo, qr_ancilla, cr_ancilla, base, hasta)
    """
    for i in range(base, hasta):
        circuit.cx(qreg_padre[i], qreg_hijo[i])
        circuit.x(qreg_padre[i])
        circuit.x(qreg_hijo[i])
        circuit.ccx(qreg_padre[i], qreg_hijo[i], qreg_ancilla[0])
        circuit.x(qreg_padre[i])
        circuit.x(qreg_hijo[i])
        circuit.cx(qreg_ancilla[0], qreg_hijo[i])
        circuit.x(qreg_hijo[i])
        circuit.x(qreg_ancilla[0])
        circuit.ccx(qreg_padre[i], qreg_ancilla[0], qreg_hijo[i])
        circuit.ccx(qreg_padre[i], qreg_hijo[i], qreg_ancilla[0])
        circuit.ccx(qreg_padre[i], qreg_ancilla[0], qreg_hijo[i])
        circuit.measure(qreg_ancilla[0], cr_ancilla[0])
        with circuit.if_test((cr_ancilla[0], 1)):
            circuit.x(qreg_ancilla[0])

def calculate_cut(partition_string, weight_matrix):
    """
    Calcula el número de aristas que cruzan la partición.

    Parámetros
    ----------
    partition_string : str
        Una cadena binaria que representa la partición.
        Cada carácter en la cadena representa un nodo en el grafo.
        Un '0' significa que el nodo está en un lado de la partición,
        y un '1' significa que está en el otro lado.
    weight_matrix : numpy.ndarray
        Una matriz que representa los pesos de las aristas en el grafo.
        Cada elemento w[i, j] es el peso de la arista entre los nodos i y j.

    Devoluciones
    -----------
    cut : int
        El número de aristas que cruzan la partición.
    """
    cut = 0
    for i in range(len(partition_string)):
        for j in range(i+1, len(partition_string)): # Evita contar dos veces la misma arista
            if partition_string[i] != partition_string[j]: # Si los nodos están en lados opuestos de la partición
                cut += weight_matrix[i, j] # Añade el peso de la arista al corte
    return cut

def qga(number_quantum_reg,qubits,number_classic_reg,bits,cost_dict,probabilidad,mejor,backend,inicial,weight_matrix):

    """
    qga(number_quantum_reg, qubits, number_classic_reg, bits, cost_dict, probabilidad, mejor, backend)

    Algoritmo genético cuántico que realiza una búsqueda de soluciones en un espacio cuántico.

    Parámetros:
    ------------
    number_quantum_reg: int
        Número de registros cuánticos a utilizar en el circuito.

    qubits: int
        Número de qubits en cada registro cuántico.

    number_classic_reg: int
        Número de registros clásicos a utilizar en el circuito.

    bits: int
        Número de bits en cada registro clásico.

    cost_dict: dict
        Un diccionario con los costos asociados a cada posible solución.

    probabilidad: float
        Probabilidad utilizada para las operaciones de mutación.

    mejor: tuple
        La mejor solución conocida hasta el momento.

    backend: Provider
        El backend en el que se ejecutará el circuito.

    inicial: tuple
        La solución inicial desde donde el algoritmo comenzará la búsqueda.

    Devuelve:
    ----------
    result: tuple
        La mejor solución encontrada por el algoritmo.

    Ejemplo:
    ---------
    >>> number_quantum_reg = 4
    >>> qubits = 6
    >>> number_classic_reg = 2
    >>> bits = 6
    >>> cost_dict = {(0, 1, 0, 0, 0, 1): 5.0, ...}
    >>> probabilidad = 0.1
    >>> mejor = (1, 0, 1, 1, 0, 0)
    >>> backend = Aer.get_backend('qasm_simulator')
    >>> qga(number_quantum_reg, qubits, number_classic_reg, bits, cost_dict, probabilidad, mejor, backend)
    (1, 0, 1, 1, 0, 0)
    """

    estadisticas = []
    shots = 10
    qregs = [QuantumRegister(qubits, name=f'q{i}') for i in range(number_quantum_reg)]
    ancilla = QuantumRegister(1, name='ancilla')

    # Create classical registers
    cregs = [ClassicalRegister(bits, name=f'c{i}') for i in range(number_classic_reg)]
    cr_ancilla = ClassicalRegister(1, name='cr_ancilla')
    qc = QuantumCircuit(*qregs, ancilla, *cregs, cr_ancilla)
    generation = 0
    best = False
 
    result=inicial
    escribir(qc,qc.qregs[0],tuple_to_string(inicial))
    for i in range(qc.qregs[1].size):
        qc.h(qc.qregs[1][i])
    #crear_hijo(base,hasta,qubits,qc.qregs[0],qc.qregs[1],qc.qregs[2],qc,qc.ancilla,qc.cr_ancilla)#usando al padre
    #crear_hijo(base,hasta,qubits,qc.qregs[1],qc.qregs[0],qc.qregs[3],qc,qc.ancilla,qc.cr_ancilla)#usando a la madre
    while generation <10:
        poblaciones_resultados = []
        print("Generation: ",generation)
        #f.write('Generation: '+str(generation) +'\n')
        crear_hijo(*puntos(qc.qregs[0]),qubits,qc.qregs[0],qc.qregs[1],qc.qregs[2],qc,ancilla,cr_ancilla)#usando al padre qc.qregs[-2]  # -2 since ancilla is the second last element added to qregs
        crear_hijo(*puntos(qc.qregs[0]),qubits,qc.qregs[1],qc.qregs[0],qc.qregs[3],qc,ancilla,cr_ancilla)#usando a la madre
        for i in range(qubits):
            qc.measure(qc.qregs[2][i],qc.cregs[0][i])
            qc.measure(qc.qregs[3][i],qc.cregs[1][i])
        hijo0,hijo1 = valores_hijos(lanzar_dinamico(qc,backend,shots))
        print("recien_crea2 hijo0: ",hijo0,"costo: ",calculate_cut(hijo0, weight_matrix))
        #f.write('recien creado hijo0 : '+", ".join(map(str, hijo0)) +'costo : '+ str(calculate_cut(hijo0, weight_matrix)) +'\n')
        print("recien_crea2 hijo1: ",hijo1,"costo: ",calculate_cut(hijo1, weight_matrix))
        #f.write('recien creado hijo0 : '+", ".join(map(str, hijo1)) +'costo : '+ str(calculate_cut(hijo1, weight_matrix)) +'\n')
        """h0 = calculate_cut(hijo0, weight_matrix)
        h1 = calculate_cut(hijo1, weight_matrix)
        inici = calculate_cut(inicial, weight_matrix)"""
        if cost_dict[hijo0]<cost_dict[hijo1] and cost_dict[inicial]<cost_dict[hijo1]:
        #if hijo0 < h1 and inici < h1:
            escribir(qc,qc.qregs[0],tuple_to_string(hijo1))#borramos la info del padre
            inicial = hijo1
            result = hijo1
        if cost_dict[hijo0]>cost_dict[hijo1] and cost_dict[inicial]<cost_dict[hijo0]:
        #if h0 > h1 and inici < h0:
            escribir(qc,qc.qregs[0],tuple_to_string(hijo0))#borramos la info del padre
            inicial = hijo0
            result = hijo0

        if  cost_dict[hijo0] == mejor:
            result = hijo0
            print("fin!!")
            poblaciones_resultados.append(calculate_cut(hijo0, weight_matrix))
            poblaciones_resultados.append(calculate_cut(hijo1, weight_matrix))
            best = True
            break

            #best = True

        if  cost_dict[hijo1] == mejor:
            result = hijo1
            #f.write('Primera linea de texto.\n')
            poblaciones_resultados.append(calculate_cut(hijo0, weight_matrix))
            poblaciones_resultados.append(calculate_cut(hijo1, weight_matrix))
            print("fin!!")
            best = True
            break



        """cual = np.random.randint(0, 4)#aleatorio cual de las 4 mutaciones
        functions = {
            0: lambda: mutate_bit(qc.qregs[0], qc, np.random.randint(0, len(qc.qregs[0]))),
            1: lambda: mutate_multi_bit(qc.qregs[0], qc, probabilidad),
            2: lambda: mutate_bit_random(qc.qregs[0], qc, probabilidad),
            3: lambda: mutate_exchange(qc.qregs[0], qc, probabilidad)
        }
        functions[cual]()"""
        """h0 = calculate_cut(hijo0, weight_matrix)
        h1 = calculate_cut(hijo1, weight_matrix)
        inici = calculate_cut(inicial, weight_matrix)"""
        #if h0 > h1 and inici < h0:
        if cost_dict[hijo0]>cost_dict[hijo1] and cost_dict[hijo0]> cost_dict[inicial]:
            """if cost_dict[hijo0] == mejor:
                best = True
                break"""
            escribir(qc,qc.qregs[0],tuple_to_string(hijo0))#borramos la info del padre
            inicial = hijo0
            result = hijo0
        #if h0 < h1 and inici < h1:
        if cost_dict[hijo0]<cost_dict[hijo1] and cost_dict[hijo1]> cost_dict[inicial]:
            """if cost_dict[hijo1] == mejor:
                best = True
                break"""

            escribir(qc,qc.qregs[0],tuple_to_string(hijo1))#borramos la info del padre
            inicial = hijo1
            result = hijo1
        #print("mutaciones_hijo0: ",hijo0,"costo: ",calculate_cut(hijo0, weight_matrix))
        #print("mutaciones_hijo1: ",hijo1,"costo: ",calculate_cut(hijo1, weight_matrix))

        print("padre: ",inicial)
        print("hijo0: ",hijo0)
        poblaciones_resultados.append(calculate_cut(hijo0, weight_matrix))
        print("hijo1: ",hijo1)
        poblaciones_resultados.append(calculate_cut(hijo1, weight_matrix))
        estadisticas.append(poblaciones_resultados)
        generation+=1
        print(generation)
    return result,best,estadisticas

def funcion(number_qubits,shots,backend,tupla):
    """
    Esta función realiza una serie de operaciones en un circuito cuántico, incluyendo inicialización de qubits,
    aplicaciones de puertas, mediciones y lanzamiento de un simulador cuántico.

    Parámetros
    ----------
    number_qubits : int
        El número de qubits para los registros cuánticos 'q0', 'q1', 'q2'.

    shots : int
        El número de veces que se debe ejecutar el circuito cuántico.

    backend : Backend
        La instancia de backend que se utilizará para ejecutar el circuito cuántico.

    Devoluciones
    -------
    dict
        Un diccionario que representa las cuentas de los resultados de las mediciones.

    Ejemplo
    -------
    >>> from qiskit import Aer
    >>> backend = Aer.get_backend('qasm_simulator')
    >>> counts = funcion(3, 100, backend)
    >>> print(counts)
    """
    q0 = QuantumRegister(number_qubits, 'q0')
    q1 = QuantumRegister(number_qubits, 'q1')
    q2 = QuantumRegister(number_qubits, 'q2')
    # Crear el qubit ancilla
    ancilla = QuantumRegister(1, 'ancilla')
    # Crear los registros clásicos
    c0 = ClassicalRegister(number_qubits, 'c0')
    c1 = ClassicalRegister(1, 'c1')
    # Crear el circuito cuántico
    qc = QuantumCircuit(q0, q1, q2, ancilla, c0, c1)
    #escribir = tuple_to_string((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))
    escribir = tuple_to_string(tupla)
    #escribir(qc,q0,"")
    # La cadena binaria que representa el estado al que quieres inicializar tu registro cuántico
    binary_string = escribir
    # Invierte la cadena, ya que los qubits en Qiskit se numeran de derecha a izquierda
    binary_string = binary_string[::-1]
    # Aplica una puerta X a cada qubit que quieras en el estado |1⟩
    for i in range(len(binary_string)):
        if binary_string[i] == '1':
            qc.x(q0[i])
    energia_inicial = calculate_cut(escribir, w)

    qc.h(q1)

    total = 0
    iteraciones = 0
    bucle = 0
    mejor = energia_inicial
    hijo=None
    while bucle<10:
        print("BUCLE: ",bucle)
        base,hasta = puntos(q0)

        crear_hijo(base,hasta,number_qubits,q0,q1,q2,qc,ancilla,c1)

        qc.measure(q2, c0)

        c=lanzar_dinamico(qc,backend,shots)#usando shots = 5 en los primeros ejemplos

        total = 0
        """AQUI CALCULAMOS LA MEDIA, METER ESTE CODIGO EN UNA FUNCION"""
        for key, value in c.items():
            #key = list(results.keys())[0]
        # Divide la clave por el espacio y toma el segundo elemento
            measurement = key.split(' ')[1]
            #print(measurement)
            #print(measurement)
            r = calculate_cut(measurement, w)
            total = total + (r*value)
        evaluacion_hijo = total / shots
        """HASTA AQUI!!!!"""
        #print("EVALUACION: ",evaluacion_hijo)
        if energia_inicial < evaluacion_hijo:
            #print("SIII")
            #mejor = 0
            for key, value in c.items():
                #key = list(results.keys())[0]
                # Divide la clave por el espacio y toma el segundo elemento
                measurement = key.split(' ')[1]
                #print("inical:", key.split(' ')[1])
                #print("obtenido: ",measurement)
                if energia_inicial < calculate_cut(measurement, w) and mejor < calculate_cut(measurement, w):
                    hijo=measurement
            #padre_hijo(qc, q2, q0, ancilla, c1, 0, 6)
            if hijo != None:
                #print("PONIENDO en q0 : ",str(hijo))
                reset(qc,q0,c1)#resetar el qreg
                reset(qc,ancilla,c1)#reset qreg de ancilla
                binary_string = hijo
                # Invierte la cadena, ya que los qubits en Qiskit se numeran de derecha a izquierda
                binary_string = binary_string[::-1]
                # Aplica una puerta X a cada qubit que quieras en el estado |1⟩
                for i in range(len(binary_string)):
                    if binary_string[i] == '1':
                        qc.x(q0[i])
                mejor = calculate_cut(hijo, w)
                #print("CONTAMOS CON ESTA: ",mejor)
                hijo = None
        total = 0
        bucle = bucle+1
    # Mide el registro cuántico q0 en el registro clásico c0
    qc.measure(q0, c0)
    job = execute(qc, backend, shots=1)
    # Obtiene los resultados
    result = job.result()
    # Obtiene las cuentas de las mediciones
    counts = result.get_counts(qc)
    return counts

def find_best_individual(individuals_dict, weight_matrix):
    best_individual = None
    best_value = -1
    for key, _ in individuals_dict.items():
        value = calculate_cut(key.split(' ')[1], weight_matrix)
        if value > best_value:
            best_individual = key.split(' ')[1]
            best_value = value
    return best_individual, best_value

def energia(dicionario,shots):
    total = 0
    for key, value in dicionario.items():
    # Divide la clave por el espacio y toma el segundo elemento
        measurement = key.split(' ')[1]
        r = calculate_cut(measurement, w)
        total = total + (r*value)
    return total / shots

def qga0(number_qubits,shots,backend,tupla):
    """
    Esta función realiza una serie de operaciones en un circuito cuántico, incluyendo inicialización de qubits,
    aplicaciones de puertas, mediciones y lanzamiento de un simulador cuántico.

    Parámetros
    ----------
    number_qubits : int
        El número de qubits para los registros cuánticos 'q0', 'q1', 'q2'.

    shots : int
        El número de veces que se debe ejecutar el circuito cuántico.

    backend : Backend
        La instancia de backend que se utilizará para ejecutar el circuito cuántico.

    Devoluciones
    -------
    dict
        Un diccionario que representa las cuentas de los resultados de las mediciones.

    Ejemplo
    -------
    >>> from qiskit import Aer
    >>> backend = Aer.get_backend('qasm_simulator')
    >>> counts = funcion(3, 100, backend)
    >>> print(counts)
    """
    poblaciones_resultados = []
    q0 = QuantumRegister(number_qubits, 'q0')
    q1 = QuantumRegister(number_qubits, 'q1')
    q2 = QuantumRegister(number_qubits, 'q2')
    # Crear el qubit ancilla
    ancilla = QuantumRegister(1, 'ancilla')
    # Crear los registros clásicos
    c0 = ClassicalRegister(number_qubits, 'c0')
    c1 = ClassicalRegister(1, 'c1')
    # Crear el circuito cuántico
    qc = QuantumCircuit(q0, q1, q2, ancilla, c0, c1)
    if isinstance(tupla, tuple):  # si el input es una tupla
        # aquí van las operaciones para el caso de tuplas
        escribir = tuple_to_string(tupla)
        #escribir(qc,q0,"")
        # La cadena binaria que representa el estado al que quieres inicializar tu registro cuántico
        binary_string = escribir
        # Invierte la cadena, ya que los qubits en Qiskit se numeran de derecha a izquierda
        binary_string = binary_string[::-1]
        # Aplica una puerta X a cada qubit que quieras en el estado |1⟩
        for i in range(len(binary_string)):
            if binary_string[i] == '1':
                qc.x(q0[i])
        energia_inicial = calculate_cut(escribir, w)
    else:# si el input es cualquier otro tipo de dato
        qc.h(q0)
        qc.measure(q0, c0)
        dic = lanzar_dinamico(qc,backend,int((2**number_qubits)/2))
        energia_inicial = energia(dic,int((2**number_qubits)/2))

    #print("Energia inicial: ",energia_inicial)
    # Aplica la puerta Hadamard a todos los qubits en q1
    qc.h(q1)
    iteraciones = 0
    bucle = 0
    while bucle<10:
        print("BUCLE: ",bucle)
        base,hasta = puntos(q0)
        #print("PUNTOS DE CORTE: ",base,hasta)
        crear_hijo(base,hasta,number_qubits,q0,q1,q2,qc,ancilla,c1)
        qc.measure(q2, c0)
        diccionario=lanzar_dinamico(qc,backend,shots)#usando shots = 5 en los primeros ejemplos
        #print("Printeamos:",diccionario)
        total = 0
        """AQUI CALCULAMOS LA MEDIA, METER ESTE CODIGO EN UNA FUNCION"""
        evaluacion_hijo = energia(diccionario,shots)
        """HASTA AQUI!!!!"""
        #print("EVALUACION: ",evaluacion_hijo)
        if energia_inicial < evaluacion_hijo:
            reset(qc,ancilla,c1)#reset qreg de ancilla
            reset(qc,q0,c1)
            #funcion que da de la lista el mejor
            binary_string,energia_inicial = find_best_individual(diccionario, w)
            # La cadena binaria que representa el estado al que quieres inicializar tu registro cuántico
            # Invierte la cadena, ya que los qubits en Qiskit se numeran de derecha a izquierda
            binary_string = binary_string[::-1]
            # Aplica una puerta X a cada qubit que quieras en el estado |1⟩
            for i in range(len(binary_string)):
                if binary_string[i] == '1':
                    qc.x(q0[i])
            evaluacion_hijo = energia_inicial
        """
        cual = np.random.randint(0, 4)#aleatorio cual de las 4 mutaciones
        functions = {
            0: lambda: mutate_bit(q0, qc, np.random.randint(0, len(qc.qregs[0]))),
            1: lambda: mutate_multi_bit(q0, qc, probabilidad),
            2: lambda: mutate_bit_random(q0, qc, probabilidad),
            3: lambda: mutate_exchange(q0, qc, probabilidad)
        }
        functions[cual]()"""
        poblaciones_resultados.append(energia_inicial)
        bucle = bucle+1
    # Mide el registro cuántico q0 en el registro clásico c0
    qc.measure(q0, c0)
    job = execute(qc, backend, shots=1)
    # Obtiene los resultados
    result = job.result()
    # Obtiene las cuentas de las mediciones
    counts = result.get_counts(qc)
    return counts,poblaciones_resultados

import datetime
import sys
import time

def obtener_valor(d):
    key = list(d.keys())[0]
    key_parts = key.split(' ')
    return key_parts[1]

# Obtener la hora actual y formatearla como cadena para usarla como nombre de archivo
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"{current_time}.txt"

# Abrir el archivo en modo de escritura
with open(filename, 'w') as f:
    # Redirigir la salida estándar al archivo
    sys.stdout = f

    n = 10  # numero de nodos
    grafo = generate_hamiltonian_graph(n,123)
    w = compute_weight_matrix(grafo,n)

    start= time.time()
    cost_dict, best_cost_brute = compute_cost(n, w, grafo )
    end = time.time()
    print("Fuerza bruta tarda : ",end-start)
    print("Mejor costo: ",best_cost_brute)

    start= time.time()
    result_funcion = funcion(n,10,backend,(0,0,0,0,0,0,0,0,0,0))
    end = time.time()
    print("funcion con 10 shots  tarda : ",end-start)
    print("resultados:",result_funcion)
    print(calculate_cut(obtener_valor(result_funcion),w))

    start= time.time()
    counts_qga0,poblaciones_resultados_qga0 = qga0(n,10,backend,(0,0,0,0,0,0,0,0,0,0))
    end = time.time()
    print("resultados:",poblaciones_resultados_qga0)
    print("counts:",counts_qga0)
    print(calculate_cut(obtener_valor(counts_qga0),w))
    print(" qga0 con 10 shots y todo a 0   tarda : ",end-start)

    start= time.time()
    counts,estats = qga0(n,10,backend,"h")
    end = time.time()
    print("qga0 con 5 shots  y todo hadamar tarda : ",end-start)
    print("counts:",counts)
    print(calculate_cut(obtener_valor(counts),w))

    start= time.time()
    counts = funcion(n,5,backend,(0,0,0,0,0,0,0,0,0,0))
    end = time.time()
    print(" funcion con 5 shots  tarda : ",end-start)
    print("counts:",counts)
    print(calculate_cut(obtener_valor(counts),w))

sys.stdout = sys.__stdout__

def graficos(estadisticas):
    mejores_aptitudes = [max(x) for x in estats]
    peores_aptitudes = [min(x) for x in estats]
    avg_aptitudes = [statistics.mean(x) for x in estats]
    # Creamos el gráfico para las aptitudes mejores y peores
    plt.figure()
    plt.plot(mejores_aptitudes, label='Mejor')
    plt.plot(peores_aptitudes, label='Peor')
    plt.xlabel('Generación')
    plt.ylabel('Aptitud')
    plt.title('Rendimiento del algoritmo genético - Mejor y peor')
    plt.legend()
    plt.show()
    # Creamos un nuevo gráfico para la aptitud media
    plt.figure()
    plt.plot(avg_aptitudes, label='Media', color='red')  # Especificamos que la línea de la media sea roja
    plt.xlabel('Generación')
    plt.ylabel('Aptitud media')
    plt.title('Rendimiento del algoritmo genético - Media')
    plt.legend()
    plt.show()


import statistics
graficos(estats)