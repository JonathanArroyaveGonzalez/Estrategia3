def reorder_cross_product(cross_product):
    len_cross = len(cross_product)

    if len_cross == 4:
        new_order = [0, 2, 1, 3]
        return cross_product[new_order]

    elif len_cross == 8:
        new_order = [0, 1, 2, 3, 4, 6, 5, 7]
        return cross_product[new_order]

    return cross_product


def ns_to_array(letras):
    ns_arr = [0] * len(letras)

    return ns_arr


def cs_to_array(cs, cs_value):
    # Crear un nuevo arreglo con None en las posiciones especificadas por cs
    cs_arr = [cs_value[i] if chr(65 + i) in cs else None for i in range(len(cs_value))]

    return cs_arr


def build_probabilities(probabilities, len_cs):
    extended_probabilities = [None] * len(probabilities)
    for i in range(len(probabilities)):
        extended_probabilities[i] = probabilities[i] + [0] * (
            len_cs - len(probabilities[i])
        )

    return extended_probabilities
