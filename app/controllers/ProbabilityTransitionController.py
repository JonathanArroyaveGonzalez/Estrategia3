import matplotlib.pyplot as plt
import numpy as np


def graphProbability(array, color, label):
    labels = [bin(i)[2:] for i in range(len(array))]
    positions = np.arange(len(labels))
    plt.bar(positions, array, color=color, alpha=0.7, label=label)
    for i, valor in enumerate(array):
        plt.text(i, valor + 0.05, f"{valor:.2f}", ha="center", va="bottom")
    plt.xlabel("Next state")
    plt.ylabel("Probability")
    plt.xticks(positions, labels)
    plt.legend()
    plt.ylim(0, 1)
    plt.show()


def getIndicesToMargenalice(states, state):
    availableIndices = []
    indices = {}
    csValue = ""

    for i in range(len(state)):
        if state[i] != None:
            availableIndices.append(i)
            csValue = str(state[i]) + csValue

    for i in range(len(states)):
        key = ""
        for j in range(len(availableIndices)):
            key += str(states[i][availableIndices[j]])

        indices[key] = indices.get(key) + [i] if indices.get(key) else [i]

    if csValue == "":
        return indices, 0

    return indices, int(csValue, 2)


def margenaliceNextState(nsIndices, probabilites):
    nsTransitionTable = [[None] * len(nsIndices) for i in range(len(probabilites))]
    currentColumn = 0
    for indices in nsIndices.values():
        for i in range(len(nsTransitionTable)):
            probability = 0
            for j in range(len(indices)):
                probability += probabilites[i][indices[j]]

            nsTransitionTable[i][currentColumn] = probability

        currentColumn += 1

    return nsTransitionTable


def margenaliceCurrentState(csIndices, nsTransitionTable):
    csTransitionTable = [
        [None] * len(nsTransitionTable[0]) for i in range(len(csIndices))
    ]

    currentRow = 0
    for indices in csIndices.values():
        for i in range(len(csTransitionTable[0])):
            probability = 0
            for j in range(len(indices)):
                probability += nsTransitionTable[indices[j]][i]

            csTransitionTable[currentRow][i] = probability / len(indices)

        currentRow += 1

    return csTransitionTable


def probabilityTransitionTable(currentState, nextState, probabilities, states):
    result = []
    csTransitionTable = []
    csIndices, csValueIndex = getIndicesToMargenalice(states, currentState)
    missingCs = any(state is None for state in currentState)

    if missingCs:
        for i, state in enumerate(nextState):
            if state is not None:
                newNs = [None] * len(nextState)
                newNs[i] = nextState[i]

                nsIndices, _ = getIndicesToMargenalice(states, newNs)
                nsTransitionTable = margenaliceNextState(nsIndices, probabilities)
                csTransitionTable = margenaliceCurrentState(
                    csIndices, nsTransitionTable
                )
                csValue = csTransitionTable[csValueIndex]

                if len(result) > 0:
                    result = np.kron(result, csValue)
                else:
                    result = csValue

        # result = reorder_cross_product(result)

    else:
        nsIndices, _ = getIndicesToMargenalice(states, nextState)
        nsTransitionTable = margenaliceNextState(nsIndices, probabilities)

        csTransitionTable = margenaliceCurrentState(csIndices, nsTransitionTable)
        result = csTransitionTable[csValueIndex]

    return result
