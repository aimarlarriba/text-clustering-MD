"""
text-clustering-MD: Ejemplo de Ejecución Didáctico (Toy-Example)
=============================================================================
Demostración paso a paso de los algoritmos y métricas desarrollados desde cero
(Scratch) sobre un conjunto de datos sintéticos controlados.

Cumple con el requisito del guión de prácticas:
"Ejecución: facilitar al usuario e.g. mediante ejemplo de ejecución (toy-example)"

Uso:
    python toy_example.py
"""

import numpy as np
from src.clustering.lance_williams import (
    obtener_parametros_lance_williams,
    actualizar_matriz_lance_williams,
)
from src.clustering.hierarchical import MotorJerarquico
from src.metricas_internas import (
    calcular_cpcc_scratch,
    calcular_silhouette_scratch,
    cortar_dendrograma_scratch,
    calcular_dunn_scratch,
)
from src.evaluacion import (
    calcular_metricas_externas,
    calcular_metricas_clasificacion_clustering_scratch,
)


def ejecutar_toy_example():
    print("=" * 70)
    print(" TOY-EXAMPLE: DEMOSTRACIÓN DE CLUSTERING JERÁRQUICO SCRATCH")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. FÓRMULA RECURSIVA DE LANCE-WILLIAMS (1967)
    # -------------------------------------------------------------------------
    print("\n--- 1. Actualización Recursiva de Lance-Williams (Matriz 3x3) ---")
    print("Instancias: A (cluster 0), B (cluster 1), C (cluster 2)")
    print("Distancias iniciales: d(A, B)=2.0, d(A, C)=6.0, d(B, C)=8.0")
    print("Fusionamos C0 y C1 en un único grupo {A, B}...")

    for enlace in ["single", "complete", "average"]:
        D_toy = np.array([
            [np.inf, 2.0, 6.0],
            [2.0, np.inf, 8.0],
            [6.0, 8.0, np.inf],
        ], dtype=float)
        tamanos = {0: 1, 1: 1, 2: 1}

        actualizar_matriz_lance_williams(
            D_toy, cluster1=0, cluster2=1, tipo_enlace=enlace, tamanos_clusters=tamanos
        )
        print(f"  - Enlace '{enlace:8s}' -> Nueva d({{A, B}}, C) = {D_toy[0, 2]:.4f} (Tamaño nuevo C0: {tamanos[0]})")

    # -------------------------------------------------------------------------
    # 2. MOTOR JERÁRQUICO AGLOMERATIVO (SAHN) SCRATCH
    # -------------------------------------------------------------------------
    print("\n--- 2. Motor Jerárquico Aglomerativo (15 Muestras en 3 Grupos) ---")
    np.random.seed(42)
    g1 = np.random.normal(loc=[-5.0, -5.0], scale=0.2, size=(5, 2))
    g2 = np.random.normal(loc=[0.0, 0.0], scale=0.2, size=(5, 2))
    g3 = np.random.normal(loc=[5.0, 5.0], scale=0.2, size=(5, 2))
    X_toy = np.vstack([g1, g2, g3])
    y_real = np.array([0] * 5 + [1] * 5 + [2] * 5)
    N = len(X_toy)

    # Matriz euclídea por pares
    diff = X_toy[:, None, :] - X_toy[None, :, :]
    D_toy = np.sqrt(np.sum(diff ** 2, axis=-1))

    motor = MotorJerarquico(tipo_enlace="complete")
    Z_toy = motor.fit(D_toy)

    print(f"Matriz de Enlace (Linkage Matrix) generada: {Z_toy.shape} (N-1 fusiones x 4 columnas)")
    print("Últimas 3 fusiones del dendrograma:")
    for paso in range(N - 4, N - 1):
        c1, c2, dist, n_elem = Z_toy[paso]
        print(f"  Paso {paso + 1:02d}: Fusiona C{int(c1)} y C{int(c2)} a distancia {dist:.4f} (tamaño: {int(n_elem)})")

    # -------------------------------------------------------------------------
    # 3. CORTE DEL DENDROGRAMA Y VALIDACIÓN INTERNA
    # -------------------------------------------------------------------------
    print("\n--- 3. Corte del Árbol Jerárquico y Validación Interna ---")
    pred_k3 = cortar_dendrograma_scratch(Z_toy, n_clusters=3)
    print(f"Corte exacto en K=3 clusters -> Asignaciones: {pred_k3}")

    cpcc = calcular_cpcc_scratch(D_toy, Z_toy)
    det_sil = calcular_silhouette_scratch(D_toy, pred_k3, devolver_detalle=True)
    dunn = calcular_dunn_scratch(D_toy, pred_k3)

    print(f"  - Coeficiente Cofenético (CPCC): {cpcc:.4f} (Preservación de la topología)")
    print(f"  - Silhouette Score global:       {det_sil['Silhouette Score']:.4f}")
    print(f"    * Cohesión media a(i):         {det_sil['Cohesion Media a(i)']:.4f}")
    print(f"    * Separación media b(i):       {det_sil['Separacion Media b(i)']:.4f}")
    print(f"  - Dunn Index:                    {dunn:.4f}")

    # -------------------------------------------------------------------------
    # 4. VALIDACIÓN EXTERNA (COMPARATIVA CON CLASES REALES)
    # -------------------------------------------------------------------------
    print("\n--- 4. Validación Externa frente a Clases Reales ---")
    ext = calcular_metricas_externas(y_real, pred_k3)
    clf = calcular_metricas_clasificacion_clustering_scratch(y_real, pred_k3, num_clases=3)

    print(f"  - Rand Index:            {ext['rand_index']:.4f}")
    print(f"  - Jaccard Index:         {ext['jaccard']:.4f}")
    print(f"  - Fowlkes-Mallows Index: {ext['fowlkes_mallows']:.4f}")
    print(f"  - Pureza (Accuracy):     {clf['accuracy_pureza'] * 100:.2f}%")
    print(f"  - Kappa de Cohen:        {clf['kappa_cohen']:.4f}")
    print(f"  - Macro F1-Score:        {clf['f1_macro']:.4f}")

    # -------------------------------------------------------------------------
    # 5. ASIGNACIÓN DE NUEVA MUESTRA NO VISTA E INFERENCIA INTRA-CLUSTER
    # -------------------------------------------------------------------------
    print("\n--- 5. Asignación de una Instancia no Vista ---")
    punto_nuevo = np.array([4.9, 5.1])
    dists_nuevo = np.linalg.norm(X_toy - punto_nuevo, axis=1)
    vecino_cercano = int(np.argmin(dists_nuevo))
    cluster_asignado = pred_k3[vecino_cercano]
    print(f"  Punto de test: {punto_nuevo}")
    print(f"  Vecino más cercano en train (1-NN global): Instancia #{vecino_cercano} a dist: {dists_nuevo[vecino_cercano]:.4f}")
    print(f"  Cluster asignado: C{cluster_asignado}")

    # Restringir la recuperación de instancias exclusivamente al cluster asignado
    indices_en_cluster = np.where(pred_k3 == cluster_asignado)[0]
    dists_intra = dists_nuevo[indices_en_cluster]
    orden_intra = np.argsort(dists_intra)

    print(f"  Instancias más cercanas DENTRO DEL CLUSTER C{cluster_asignado} (total en cluster: {len(indices_en_cluster)}):")
    for pos, idx_rel in enumerate(orden_intra[:3], 1):
        idx_global = indices_en_cluster[idx_rel]
        print(f"    {pos}. Instancia #{idx_global} (Coords: {X_toy[idx_global]}, Dist: {dists_intra[idx_rel]:.4f})")

    # -------------------------------------------------------------------------
    # 6. DEMOSTRACIÓN TEXTUAL REAL (TEXT CLUSTERING SOBRE MINI-CORPUS)
    # -------------------------------------------------------------------------
    print("\n--- 6. Demostración sobre Corpus Textual Clínico (Mini-Corpus) ---")
    textos_toy = [
        # Categoría A: Ideación / Desesperanza
        "I feel completely hopeless and I cannot see any reason to keep waking up.",
        "Everything hurts inside and I just want to fall asleep and never wake up again.",
        # Categoría B: Apoyo / Mensajes de ayuda
        "Please remember that you are not alone, there are hotlines and people who care.",
        "We want to listen and help you through this difficult time, please send me a message.",
        # Categoría C: Neutro / Conversacional
        "I went to the store today to buy some groceries for the weekend.",
        "The weather has been quite rainy all afternoon so I stayed home watching movies.",
    ]
    clases_toy_text = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    nombres_grupos = {0: "Ideación / Desesperanza", 1: "Apoyo / Red de Ayuda", 2: "Neutro / Cotidiano"}

    print(f"  Mini-corpus definido con {len(textos_toy)} textos en 3 categorías temáticas:")
    for idx_t, txt in enumerate(textos_toy):
        print(f"    [{idx_t}] ({nombres_grupos[clases_toy_text[idx_t]]}) \"{txt}\"")

    try:
        from src.embeddings import generar_embeddings, calcular_matriz_distancias_scratch
        from src.evaluacion import inferir_y_mostrar_cercanos

        print("\n  Generando embeddings densos con SBERT (all-MiniLM-L6-v2) normalizados L2...")
        X_txt = generar_embeddings(textos_toy, normalizar=True)
        D_cos_txt = calcular_matriz_distancias_scratch(X_txt, metrica="cosine")

        motor_txt = MotorJerarquico(tipo_enlace="complete")
        Z_txt = motor_txt.fit(D_cos_txt)
        pred_txt = cortar_dendrograma_scratch(Z_txt, n_clusters=3)

        sil_txt = calcular_silhouette_scratch(D_cos_txt, pred_txt)
        cpcc_txt = calcular_cpcc_scratch(D_cos_txt, Z_txt)
        ext_txt = calcular_metricas_externas(clases_toy_text, pred_txt)

        print(f"\n  Clustering Jerárquico Aglomerativo sobre texto (Scratch):")
        print(f"    - Asignación de clusters descubierta: {pred_txt}")
        print(f"    - Silhouette Score (Coseno, K=3):    {sil_txt:.4f}")
        print(f"    - Coeficiente Cofenético (CPCC):     {cpcc_txt:.4f}")
        print(f"    - Rand Index (vs categorías reales): {ext_txt['rand_index']:.4f}")

        # Inferencia semántica interactiva con un nuevo texto de prueba
        texto_prueba = "I need someone to talk to, I feel so sad and lonely tonight."
        print(f"\n  Inferencia semántica intra-cluster con nuevo texto: \"{texto_prueba}\"")
        vector_prueba = generar_embeddings([texto_prueba], normalizar=True)[0]
        cluster_recup, mas_cercanos = inferir_y_mostrar_cercanos(
            vector_prueba, X_txt, pred_txt, textos_toy, k=2, metrica="cosine"
        )
        print(f"  -> Asignado con éxito al Cluster C{cluster_recup}")
    except Exception as e:
        print(f"  [Aviso en demostración textual]: {e}")

    print("\n" + "=" * 70)
    print(" [OK] Demostración de toy-example ejecutada con éxito (2D sintético y Texto real).")
    print("=" * 70)


if __name__ == "__main__":
    ejecutar_toy_example()

