from controllers.DescompositionController import decomposition
from controllers.CutController import cut_process,cut_process_with_heuristic

# Lista de estados del sistema
states = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]

probabilities = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
]

# Estado futuro del sistema
ns = "AC"

# Estado actual del sistema
cs = "ABC"

# Valor del estado actual del sistema
cs_value = [1, 0, 0]

decomposition(ns, cs, cs_value, probabilities, states)
#cut_process(ns, cs, cs_value, probabilities, states)

# Cambiar la llamada de cut_process a cut_process_with_heuristic
cut_process_with_heuristic(ns, cs, cs_value, probabilities, states)

"""
# pruebas decomposition
# Ejemplo 1 proyecto
ns = "ABC"
cs = "ABC"
cs_value = [1, 0, 0]

decomposition(ns, cs, cs_value)

# Ejemplo 2 proyecto
ns = "AB"
cs = "ABC"
cs_value = [1, 0, 0]

decomposition(ns, cs, cs_value)

# Ejemplo 3 proyecto
ns = "AC"
cs = "ABC"
cs_value = [1, 0, 0]

decomposition(ns, cs, cs_value)

# Ejemplo 4 proyecto
ns = "A"
cs = "AC"
cs_value = [1, 0, 0]

decomposition(ns, cs, cs_value)

# Ejemplo 5 proyecto
ns = "ABC"
cs = "AB"
cs_value = [1, 0, 0]

decomposition(ns, cs, cs_value)


# pruebas cut
states = [
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0],
    [1, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 0, 1],
    [0, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [0, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 1],
    [0, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 1, 1],
    [0, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
]


probabilities = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
]

cs_value = [1, 0, 0, 0, 1, 0]

# A
ns = "ABC"
cs = "ABC"

cut_process(ns, cs, cs_value, probabilities, states)

# B
ns = "B"
cs = "A"

cut_process(ns, cs, cs_value, probabilities, states)

# C
ns = "A"
cs = "B"

cut_process(ns, cs, cs_value, probabilities, states)

# D
ns = "C"
cs = "ABC"

cut_process(ns, cs, cs_value, probabilities, states)

# E
ns = "AB"
cs = "C"

cut_process(ns, cs, cs_value, probabilities, states)

# F
ns = "ABCD"
cs = "ABCD"

cut_process(ns, cs, cs_value, probabilities, states)

# G
ns = "ABCDEF"
cs = "ABCD"

cut_process(ns, cs, cs_value, probabilities, states)

# H
ns = "ABCDE"
cs = "ABC"

cut_process(ns, cs, cs_value, probabilities, states)

# H
ns = "ABCDE"
cs = "ABC"

cut_process(ns, cs, cs_value, probabilities, states)

# J
ns = "BC"
cs = "ABC"

cut_process(ns, cs, cs_value, probabilities, states)
"""
