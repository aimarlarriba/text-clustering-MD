import re

# Lista básica de stopwords (palabras comunes que no aportan significado)
STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "en", "y", "a",
    "que", "es", "por", "para", "con", "no", "se", "su", "al", "lo", "como", "del"
}


def limpiar_texto(texto):
    """
    Limpia un texto individual:
    1. Comprueba si es nulo o vacío.
    2. Pasa a minúsculas.
    3. Quita puntuación y caracteres especiales (deja solo letras).
    4. Quita stopwords básicas.
    """
    if not texto or not isinstance(texto, str):
        return ""

    # Pasar a minúsculas
    texto = texto.lower()

    # Quitar signos de puntuación, números y caracteres raros
    texto = re.sub(r"[^a-záéíóúñ\s]", " ", texto)

    # Separar en palabras y filtrar stopwords
    palabras = texto.split()
    palabras_buenas = [p for p in palabras if p not in STOPWORDS and len(p) > 2]

    return " ".join(palabras_buenas)


def limpiar_corpus(lista_textos):
    """Aplica la limpieza a una lista de textos."""
    return [limpiar_texto(t) for t in lista_textos]
