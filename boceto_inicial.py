from src.preprocessing import (
    limpiar_corpus,
    VectorizadorTexto,
    guardar_vectorizador,
    cargar_vectorizador,
    reducir_a_2d,
)

# 1. Datos de juguete (textos de ejemplo para probar los primeros pasos)
textos_ejemplo = [
    "El Real Madrid gana el partido de fútbol con tres goles.",
    "El jugador de baloncesto encestó un triple en el último segundo.",
    "Partido de fútbol intenso entre los dos equipos de la liga.",
    "La inteligencia artificial y el aprendizaje automático analizan datos.",
    "El nuevo procesador y la tarjeta gráfica mejoran el rendimiento del ordenador.",
    "Algoritmos de minería de datos y clustering para clasificar información."
]

print("=== 1. TEXTOS ORIGINALES ===")
for i, t in enumerate(textos_ejemplo):
    print(f"[{i+1}] {t}")

# 2. Limpieza de texto
print("\n=== 2. TEXTOS TRAS LA LIMPIEZA ===")
textos_limpios = limpiar_corpus(textos_ejemplo)
for i, t in enumerate(textos_limpios):
    print(f"[{i+1}] {t}")

# 3. Vectorización (TF-IDF básico)
print("\n=== 3. CREANDO MATRIZ DE DATOS X ===")
vec = VectorizadorTexto(tipo="tfidf")
X = vec.fit_transform(textos_limpios)

print(f"Número de instancias (N): {X.shape[0]}")
print(f"Número de atributos / palabras (V): {X.shape[1]}")
print("Vocabulario aprendido:", list(vec.vocabulario.keys())[:10], "...")

# 4. Guardar el vectorizador (Persistencia para el módulo de inferencia)
print("\n=== 4. GUARDAR Y CARGAR (PERSISTENCIA) ===")
guardar_vectorizador(vec, "vectorizador_boceto.pkl")

# Cargamos el vectorizador como lo haría nuestro compañero en inferencia
vec_cargado = cargar_vectorizador("vectorizador_boceto.pkl")

# Probamos con un texto nuevo que no estaba en el entrenamiento
nuevo_texto = ["Un partido de baloncesto con muchos triples y canastas."]
nuevo_limpio = limpiar_corpus(nuevo_texto)
X_nuevo = vec_cargado.transform(nuevo_limpio)
print("Nuevo texto proyectado a vector con forma:", X_nuevo.shape)

# 5. Reducción a 2D con PCA (como se explica en las transparencias)
print("\n=== 5. REDUCCIÓN DIMENSIONAL CON PCA ===")
X_2d, pca = reducir_a_2d(X)
print("Forma de la matriz reducida:", X_2d.shape)
print("Varianza explicada por las 2 componentes:", pca.explained_variance_ratio_)
print("\n¡Primer boceto completado con éxito!")
