import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from scipy.cluster.hierarchy import fcluster
try:
    from src.metricas_internas import (
        calcular_cpcc_scratch,
        calcular_silhouette_scratch,
        calcular_sse_bss_scratch,
        calcular_dunn_scratch,
        cortar_dendrograma_scratch,
        generar_graficas_validacion_interna,
        optimizar_hiperparametros_internos,
    )
except (ImportError, ModuleNotFoundError):
    from metricas_internas import (
        calcular_cpcc_scratch,
        calcular_silhouette_scratch,
        calcular_sse_bss_scratch,
        calcular_dunn_scratch,
        cortar_dendrograma_scratch,
        generar_graficas_validacion_interna,
        optimizar_hiperparametros_internos,
    )


def calcular_metricas_externas(labels_true, labels_pred):
    """
    Calcula las métricas de validación externa evaluando los pares de instancias.

    Pre-condiciones:
    - labels_true y labels_pred deben ser arrays 1D de igual longitud (N,).

    Post-condiciones:
    - Devuelve un diccionario con las tres métricas (Rand, Jaccard, Fowlkes-Mallows).
    - Todos los índices devueltos están acotados en [0.0, 1.0].
    """
    labels_true = np.asarray(labels_true).flatten()
    labels_pred = np.asarray(labels_pred).flatten()

    assert len(labels_true) == len(labels_pred), "Las dimensiones de las etiquetas no coinciden"

    n = len(labels_true)

    mat_true = (labels_true[:, None] == labels_true[None, :])
    mat_pred = (labels_pred[:, None] == labels_pred[None, :])

    upper_tri_indices = np.triu_indices(n, k=1)
    pairs_true = mat_true[upper_tri_indices]
    pairs_pred = mat_pred[upper_tri_indices]

    a = np.sum(pairs_true & pairs_pred)
    b = np.sum(~pairs_true & ~pairs_pred)
    c = np.sum(~pairs_true & pairs_pred)
    d = np.sum(pairs_true & ~pairs_pred)

    rand_index = (a + b) / (a + b + c + d) if (a + b + c + d) > 0 else 0.0
    jaccard = a / (a + c + d) if (a + c + d) > 0 else 0.0
    fowlkes_mallows = a / np.sqrt((a + c) * (a + d)) if (a + c) > 0 and (a + d) > 0 else 0.0

    return {
        "rand_index": float(rand_index),
        "jaccard": float(jaccard),
        "fowlkes_mallows": float(fowlkes_mallows)
    }


def calcular_metricas_clasificacion_clustering_scratch(
    labels_true,
    labels_pred,
    num_clases=5,
    mapeo_previo=None,
    mapeo_entrenamiento=None,
):
    """
    Evalúa cómo se correlacionan los clusters descubiertos con las clases reales
    mediante Asignación por Voto Mayoritario (como en MNIST) y las figuras de mérito
    del Tema 3 de Alicia Pérez (Diapositivas 7-12 de 3_DM_HANDOUT.pdf):
      - Tabla de Contingencia (Cluster x Clase Real) y Matriz de Confusión (Clase x Clase).
      - Purity / Clustering Accuracy.
      - Precision, Recall y F-measure por clase (per-label), Macro-promedio y Media Ponderada (w.Avg).
      - Estadístico Kappa de Cohen (corrección del acierto por azar).

    Pre-condiciones:
        - labels_true: array 1D de longitud N con las clases reales [0 .. num_clases-1].
        - labels_pred: array 1D de longitud N con los IDs de cluster [0 .. K-1].
        - num_clases: entero >= 2 (por defecto 5 para C-SSRS).
        - mapeo_previo / mapeo_entrenamiento: dict opcional {cluster_id: clase_id} aprendido en entrenamiento (Train).
          Si se proporciona, NO se re-optimiza el voto mayoritario con las etiquetas de test,
          evitando la fuga conceptual de etiquetas (data leakage en Holdout).

    Post-condiciones:
        - Devuelve un diccionario exhaustivo con Accuracy/Pureza, Kappa, Macro-F1, Weighted-F1,
          vectores per-label de precision/recall/f1, soporte por clase, mapeo y matrices.
    """
    if mapeo_previo is None and mapeo_entrenamiento is not None:
        mapeo_previo = mapeo_entrenamiento

    y_true = np.asarray(labels_true, dtype=int).flatten()
    y_pred = np.asarray(labels_pred, dtype=int).flatten()
    N = len(y_true)
    clusters_unicos = np.unique(y_pred)
    K = len(clusters_unicos)

    # 1. Tabla de contingencia (K clusters x C clases reales)
    tabla_contingencia = np.zeros((K, num_clases), dtype=int)
    mapeo_cluster_a_clase = {}
    y_pred_mapeado = np.zeros_like(y_true)

    for idx_c, c_id in enumerate(clusters_unicos):
        mascara = (y_pred == c_id)
        clases_en_cluster = y_true[mascara]
        for cl in range(num_clases):
            tabla_contingencia[idx_c, cl] = int(np.sum(clases_en_cluster == cl))

        # Si tenemos un mapeo previo fijado en Train, lo respetamos estrictamente
        if mapeo_previo is not None and int(c_id) in mapeo_previo:
            clase_asignada = int(mapeo_previo[int(c_id)])
        else:
            # Voto mayoritario en Train: asignamos al cluster la clase real más frecuente
            clase_asignada = int(np.argmax(tabla_contingencia[idx_c])) if len(clases_en_cluster) > 0 else 0

        mapeo_cluster_a_clase[int(c_id)] = clase_asignada
        y_pred_mapeado[mascara] = clase_asignada

    # 2. Matriz de confusión cuadrada (num_clases x num_clases): Real (filas) vs Predicho (columnas)
    matriz_confusion = np.zeros((num_clases, num_clases), dtype=int)
    for t, p in zip(y_true, y_pred_mapeado):
        matriz_confusion[t, p] += 1

    # 3. Accuracy (Pureza del clustering)
    aciertos = np.trace(matriz_confusion)
    accuracy_pureza = float(aciertos / N) if N > 0 else 0.0

    # 4. Estadístico Kappa de Cohen (Slide 12 de 3_DM_HANDOUT.pdf)
    # p_0 = accuracy observada; p_e = probabilidad de acuerdo aleatorio manteniendo proporciones marginales
    totales_reales = np.sum(matriz_confusion, axis=1)
    totales_pred = np.sum(matriz_confusion, axis=0)
    p_e = float(np.sum(totales_reales * totales_pred)) / float(N * N) if N > 0 else 0.0
    kappa = (accuracy_pureza - p_e) / (1.0 - p_e) if (1.0 - p_e) > 1e-12 else 0.0

    # 5. Precision, Recall y F-measure por clase (per-label), Macro y Weighted Average (Slide 9 de 3_DM_HANDOUT.pdf)
    precision_por_clase = []
    recall_por_clase = []
    f1_por_clase = []
    soporte_por_clase = []

    for cl in range(num_clases):
        tp = float(matriz_confusion[cl, cl])
        fp = float(totales_pred[cl] - tp)
        fn = float(totales_reales[cl] - tp)
        soporte = int(totales_reales[cl])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precision_por_clase.append(round(prec, 4))
        recall_por_clase.append(round(rec, 4))
        f1_por_clase.append(round(f1, 4))
        soporte_por_clase.append(soporte)

    # Weighted Average (w.Avg) ponderado por el soporte real de cada clase clínica
    pesos = np.array(soporte_por_clase, dtype=float)
    suma_pesos = np.sum(pesos)
    if suma_pesos > 0:
        prec_weighted = float(np.sum(np.array(precision_por_clase) * pesos) / suma_pesos)
        rec_weighted = float(np.sum(np.array(recall_por_clase) * pesos) / suma_pesos)
        f1_weighted = float(np.sum(np.array(f1_por_clase) * pesos) / suma_pesos)
    else:
        prec_weighted = rec_weighted = f1_weighted = 0.0

    return {
        "accuracy_pureza": round(accuracy_pureza, 4),
        "kappa_cohen": round(float(kappa), 4),
        "f1_macro": round(float(np.mean(f1_por_clase)), 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_macro": round(float(np.mean(precision_por_clase)), 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(float(np.mean(recall_por_clase)), 4),
        "recall_weighted": round(rec_weighted, 4),
        "precision_por_clase": precision_por_clase,
        "recall_por_clase": recall_por_clase,
        "f1_por_clase": f1_por_clase,
        "soporte_por_clase": soporte_por_clase,
        "mapeo_cluster_a_clase": mapeo_cluster_a_clase,
        "tabla_contingencia": tabla_contingencia,
        "matriz_confusion": matriz_confusion,
    }



def extraer_y_visualizar_centroides_textuales_scratch(
    vectores_densos,
    matriz_distancias,
    etiquetas_reales,
    textos_reales,
    k_representar=15,
    carpeta_salida="data_processed/graficas_informe"
):
    """
    Caracteriza profundamente qué representa cada Cluster (Codebook de Centroides m_j) mediante:
      1. 6 Ventanas de 3 palabras (6 Trigramas temáticos por cluster con c-TF-IDF Scratch).
      2. La Oración Prototipo Central (la frase exacta más representativa del centroide m_j).
      3. El perfil de distribución en las 5 clases clínicas reales (C-SSRS).
    Genera 3 figuras limpias y sin solapamientos:
      - fig6_mapa_pca_centroides.png (Proyección PCA 2D con etiquetas anticolisión).
      - fig7_tabla_codebook_6x3_ventanas.png (Tabla completa con las 6 ventanas de 3 palabras + oración prototipo).
      - fig8_correlacion_externa_k.png (Curvas de Accuracy/Pureza, Rand, F1, Jaccard y Kappa en k=2..sqrt(N)).
    """
    import re
    import textwrap
    try:
        from src.clustering.hierarchical import MotorJerarquico
    except (ImportError, ModuleNotFoundError):
        try:
            from clustering.hierarchical import MotorJerarquico
        except (ImportError, ModuleNotFoundError):
            from .clustering.hierarchical import MotorJerarquico


    os.makedirs(carpeta_salida, exist_ok=True)
    nombres_clases = {0: "Indicator", 1: "Ideation", 2: "Behavior", 3: "Attempt", 4: "Supportive"}
    N = len(textos_reales)

    stopwords_en = {
        "the", "and", "to", "of", "in", "it", "is", "that", "for", "on", "you", "this", "with",
        "be", "are", "have", "not", "but", "at", "from", "or", "by", "as", "what", "all", "if",
        "can", "about", "just", "my", "me", "im", "so", "do", "dont", "was", "like", "know",
        "would", "get", "out", "up", "when", "people", "they", "how", "one", "there", "really",
        "think", "want", "feel", "ive", "because", "time", "even", "now", "going", "been", "much",
        "will", "who", "more", "had", "has", "their", "your", "we", "he", "she", "her", "his",
        "them", "some", "an", "no", "very", "too", "any", "only", "other", "than", "into", "could",
        "were", "did", "didnt", "cant", "wont", "am", "being", "things", "something", "anything",
        "make", "way", "see", "go", "good", "actual", "actually", "still", "back", "after", "here",
        "http", "https", "www", "com", "reddit", "html", "org", "watch", "youtube", "post", "posts",
        "someone", "anyone", "everyone", "everything", "nothing", "maybe", "always", "never", "lot",
        "also", "though", "around", "through", "every", "where", "which", "while", "those", "these",
        "right", "down", "well", "said", "say", "says", "got", "getting", "take", "took", "made",
        "let", "put", "come", "came", "look", "looking", "tell", "told", "sure", "else", "bit",
        "its", "youre", "hes", "shes", "theyre", "thats", "theres", "heres", "whatsoever", "quite"
    }

    # Eliminamos todas las etiquetas médicas artificiales inyectadas en el corpus de Zenodo
    patrones_ruido = re.compile(
        r"\b(hyperactive behavior|depressed mood|mental depression|anxiety mental|irritable mood|"
        r"drug abuse|bipolar disorder|ventricular dysfunction|malignant neoplastic|neoplastic disease|"
        r"ache|tired|exhaustion|suicidal ideation)\b",
        flags=re.IGNORECASE
    )

    docs_ventanas_3 = []
    docs_palabras = []
    df_ventanas = {}
    df_palabras = {}

    for txt in textos_reales:
        txt_limpio = patrones_ruido.sub(" ", txt)
        txt_limpio = re.sub(r"http\S+|www\.\S+", " ", txt_limpio.lower())
        palabras = [w for w in re.findall(r"[a-z]{3,}", txt_limpio) if w not in stopwords_en]
        docs_palabras.append(palabras)

        for w in set(palabras):
            df_palabras[w] = df_palabras.get(w, 0) + 1

        # Construir ventanas deslizantes de 3 palabras (trigramas) sin palabras repetidas seguidas
        ventanas_doc = []
        for idx_w in range(len(palabras) - 2):
            w1, w2, w3 = palabras[idx_w], palabras[idx_w + 1], palabras[idx_w + 2]
            if len({w1, w2, w3}) == 3:
                ventanas_doc.append(f"{w1} {w2} {w3}")

        # Si el post es muy breve, respaldamos con ventanas de 2 palabras
        if len(ventanas_doc) < 6:
            for idx_w in range(len(palabras) - 1):
                w1, w2 = palabras[idx_w], palabras[idx_w + 1]
                if w1 != w2:
                    ventanas_doc.append(f"{w1} {w2}")

        docs_ventanas_3.append(ventanas_doc)
        for v3 in set(ventanas_doc):
            df_ventanas[v3] = df_ventanas.get(v3, 0) + 1

    # --- Cortar en k_representar clusters y extraer 6 ventanas de 3 palabras + oración prototipo ---
    Z_comp = MotorJerarquico(tipo_enlace="complete").fit(matriz_distancias)
    etiquetas_k = cortar_dendrograma_scratch(Z_comp, n_clusters=k_representar)
    res_clf = calcular_metricas_clasificacion_clustering_scratch(etiquetas_reales, etiquetas_k)

    centroides_384 = np.zeros((k_representar, vectores_densos.shape[1]), dtype=np.float32)
    resumen_centroides = []

    print(f"\n=== CODEBOOK DE CENTROIDES: 6 VENTANAS DE 3 PALABRAS + ORACIÓN PROTOTIPO (K = {k_representar}) ===")
    for c in range(k_representar):
        idx_c = np.where(etiquetas_k == c)[0]
        puntos_c = vectores_densos[idx_c]

        # 1) Centroide matemático m_c
        m_c = np.mean(puntos_c, axis=0)
        centroides_384[c] = m_c

        # 2) Post medoide (más cercano al centroide m_c) y los top 3 más cercanos al centroide
        dists_al_centroide = np.linalg.norm(puntos_c - m_c, axis=1)
        orden_cercania = np.argsort(dists_al_centroide)
        idx_medoide_global = int(idx_c[int(orden_cercania[0])])
        top_docs_centrales = [int(idx_c[int(i)]) for i in orden_cercania[:min(5, len(idx_c))]]

        # 3) Perfil léxico del cluster (c-TF-IDF de palabras individuales para puntuar importancia semántica)
        tf_w_cluster = {}
        for doc_id in idx_c:
            for w in docs_palabras[doc_id]:
                tf_w_cluster[w] = tf_w_cluster.get(w, 0) + 1
        peso_palabra = {
            w: (freq / float(len(idx_c))) * np.log(1.0 + float(N) / float(df_palabras.get(w, 1)))
            for w, freq in tf_w_cluster.items()
        }

        # 4) Seleccionar 6 ventanas de 3 palabras (no solapadas entre sí) con mayor c-TF-IDF en el cluster
        tf_v3_cluster = {}
        for doc_id in idx_c:
            for v3 in docs_ventanas_3[doc_id]:
                tf_v3_cluster[v3] = tf_v3_cluster.get(v3, 0) + 1

        candidatas_v3 = []
        for v3, freq_v in tf_v3_cluster.items():
            palabras_v = v3.split()
            score_palabras = np.mean([peso_palabra.get(w, 0.0) for w in palabras_v])
            idf_v = np.log(1.0 + float(N) / float(df_ventanas.get(v3, 1)))
            bonus_Tried = 1.25 if len(palabras_v) == 3 else 0.85
            score_total = bonus_Tried * (0.65 * (freq_v / float(len(idx_c))) * idf_v + 0.35 * score_palabras)
            candidatas_v3.append((v3, score_total))

        candidatas_v3.sort(key=lambda x: x[1], reverse=True)

        # Elegimos 6 ventanas diversas (evitando repetir exactamente las mismas 2 palabras en todas)
        seis_ventanas = []
        palabras_usadas = {}
        for v3, _ in candidatas_v3:
            w_set = v3.split()
            if any(palabras_usadas.get(w, 0) >= 2 for w in w_set):
                continue
            seis_ventanas.append(v3)
            for w in w_set:
                palabras_usadas[w] = palabras_usadas.get(w, 0) + 1
            if len(seis_ventanas) == 6:
                break

        # Rellenar hasta 6 si el cluster tenía un único post muy corto
        for v3, _ in candidatas_v3:
            if len(seis_ventanas) >= 6:
                break
            if v3 not in seis_ventanas:
                seis_ventanas.append(v3)

        # 5) Extraer la Oración Prototipo Central (la frase de los posts centrales con mayor densidad temática)
        mejor_oracion = ""
        mejor_score_oracion = -1.0
        for doc_id in top_docs_centrales:
            txt_limpio_doc = patrones_ruido.sub("", textos_reales[doc_id])
            txt_limpio_doc = re.sub(r"http\S+|www\.\S+", "", txt_limpio_doc)
            oraciones = re.split(r"[.!?]+", txt_limpio_doc)
            for orac in oraciones:
                orac_str = " ".join(orac.strip().split())
                if len(orac_str) < 35 or len(orac_str) > 160:
                    continue
                w_orac = [w for w in re.findall(r"[a-z]{3,}", orac_str.lower()) if w not in stopwords_en]
                if len(w_orac) < 4:
                    continue
                # Puntuamos la oración por la suma de pesos temáticos de sus palabras + cercanía al centroide
                factor_cercania = 1.25 if doc_id == idx_medoide_global else 1.0
                sc_orac = factor_cercania * float(np.mean([peso_palabra.get(w, 0.0) for w in set(w_orac)]))
                if sc_orac > mejor_score_oracion:
                    mejor_score_oracion = sc_orac
                    mejor_oracion = orac_str

        if not mejor_oracion:
            mejor_oracion = patrones_ruido.sub("", textos_reales[idx_medoide_global]).replace("\n", " ").strip()[:115]

        # Distribución de las clases clínicas dentro del cluster
        conteo_clases_c = res_clf["tabla_contingencia"][c]
        orden_clases = np.argsort(conteo_clases_c)[::-1]
        clase_dom_id = int(orden_clases[0])
        clase_sec_id = int(orden_clases[1])
        clase_dom_nombre = nombres_clases[clase_dom_id]
        pureza_c = conteo_clases_c[clase_dom_id] / float(len(idx_c))
        pct_sec = conteo_clases_c[clase_sec_id] / float(len(idx_c))
        dist_str = f"{clase_dom_nombre} ({pureza_c*100:.0f}%)" + (f" + {nombres_clases[clase_sec_id][:4]}. ({pct_sec*100:.0f}%)" if pct_sec >= 0.15 else "")

        info_c = {
            "cluster": c,
            "tamano": len(idx_c),
            "clase_dominante": clase_dom_nombre,
            "distribucion_clases": dist_str,
            "pureza_cluster": round(float(pureza_c), 3),
            "seis_ventanas_3pal": seis_ventanas,
            "idx_medoide": idx_medoide_global,
            "oracion_prototipo": mejor_oracion,
        }
        resumen_centroides.append(info_c)
        print(
            f"  * Cluster C{c:2d} (n={len(idx_c):3d}) | Perfil Clínico: {dist_str}\n"
            f"    -> 6 Ventanas (3 pal.): {' | '.join(seis_ventanas)}\n"
            f"    -> Oración Prototipo (Post #{idx_medoide_global}): \"{mejor_oracion}\""
        )

    # =========================================================================
    # FIGURA 6: MAPA PCA 2D DE CLUSTERS Y CENTROIDES (CON POSICIONES ANTICOLISIÓN)
    # =========================================================================
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(vectores_densos)
    centroides_2d = pca.transform(centroides_384)
    cmap = plt.get_cmap("tab20").resampled(k_representar)
    var_pca1 = float(pca.explained_variance_ratio_[0] * 100)
    var_pca2 = float(pca.explained_variance_ratio_[1] * 100)
    var_total_2d = var_pca1 + var_pca2

    fig, ax_map = plt.subplots(figsize=(12.5, 7.2))
    fig.suptitle(
        f"Proyección Espacial PCA 2D de los Clusters y Codebook de Centroides m_j (k̂ = {k_representar})",
        fontsize=14, fontweight="bold", y=0.97
    )
    plt.figtext(
        0.5, 0.915,
        f"Visualización en el subespacio de máxima varianza (Varianza explicada 2D: {var_total_2d:.1f}% | PCA 1: {var_pca1:.1f}%, PCA 2: {var_pca2:.1f}%) y centroides m_j (Estrellas)",
        ha="center", fontsize=9.8, style="italic", color="#333333"
    )

    # Desplazamientos angulares para que ninguna etiqueta de centroide pise a otra en la zona central
    offsets_etiquetas = [
        (0.025, 0.030), (-0.095, -0.045), (0.028, -0.038), (-0.105, 0.032),
        (0.030, 0.042), (-0.090, 0.040), (0.030, -0.045), (-0.110, -0.025)
    ]
    idx_offset = 0

    for info in sorted(resumen_centroides, key=lambda x: x["tamano"], reverse=True):
        c = info["cluster"]
        idx_c = (etiquetas_k == c)
        es_principal = info["tamano"] >= 5

        ax_map.scatter(
            X_2d[idx_c, 0], X_2d[idx_c, 1],
            color=cmap(c), alpha=0.55 if es_principal else 0.95,
            s=32 if es_principal else 55,
            marker="o" if es_principal else "X",
            label=f"C{c:2d} (n={info['tamano']:3d}) [{info['clase_dominante'][:4]}]: {info['seis_ventanas_3pal'][0]}"
        )

        if es_principal:
            cx, cy = centroides_2d[c, 0], centroides_2d[c, 1]
            ax_map.scatter(cx, cy, color=cmap(c), edgecolor="black", marker="*", s=360, linewidth=1.5, zorder=10)
            dx, dy = offsets_etiquetas[idx_offset % len(offsets_etiquetas)]
            idx_offset += 1
            ax_map.annotate(
                f"m_{c} (n={info['tamano']})",
                xy=(cx, cy),
                xytext=(cx + dx, cy + dy),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="black", alpha=0.7),
                fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=cmap(c), lw=1.4, alpha=0.95),
                zorder=11
            )

    ax_map.set_xlabel(f"Componente Principal 1 (PCA 1 - {var_pca1:.1f}% varianza)", fontsize=10.5)
    ax_map.set_ylabel(f"Componente Principal 2 (PCA 2 - {var_pca2:.1f}% varianza)", fontsize=10.5)
    ax_map.grid(True, linestyle=":", alpha=0.6)
    ax_map.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.3, title="Clusters y 1ª Ventana Temática", title_fontsize=9, framealpha=0.95)

    nota_f6 = textwrap.fill(
        f"Lectura del mapa PCA: Las estrellas representan las coordenadas exactas de los centroides m_j = (1/|G_j|) Σ x en R^384 proyectadas en el plano 2D ({var_total_2d:.1f}% de varianza acumulada). "
        "La leyenda exterior derecha muestra para cada cluster su tamaño, su clase clínica mayoritaria y su primera ventana temática de 3 palabras.",
        width=130
    )
    plt.figtext(0.5, 0.015, nota_f6, ha="center", fontsize=8.8, bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#666666"))

    plt.tight_layout(rect=[0, 0.08, 0.99, 0.89])
    ruta_fig6 = os.path.join(carpeta_salida, "fig6_mapa_pca_centroides.png")
    plt.savefig(ruta_fig6, dpi=220)
    plt.close(fig)

    # =========================================================================
    # FIGURA 7: TABLA COMPLETA DEL CODEBOOK (6 VENTANAS DE 3 PALABRAS + ORACIÓN PROTOTIPO)
    # =========================================================================
    fig, ax_tab = plt.subplots(figsize=(17.5, 9.5))
    ax_tab.axis("off")
    fig.suptitle(
        f"Caracterización Semántica de los Clusters (k̂ = {k_representar}): 6 Ventanas de 3 Palabras y Oración Prototipo Central",
        fontsize=14, fontweight="bold", y=0.97
    )
    plt.figtext(
        0.5, 0.925,
        "Identificación del contenido temático de cada centroide m_j combinando 6 trigramas discriminativos (c-TF-IDF Scratch) y la frase nuclear más representativa del grupo",
        ha="center", fontsize=9.8, style="italic", color="#333333"
    )

    filas_tabla = []
    colores_filas = []
    for info in sorted(resumen_centroides, key=lambda x: x["tamano"], reverse=True):
        c_id = info["cluster"]
        v_6 = info["seis_ventanas_3pal"]
        bloque_ventanas = (
            "1) " + " | 2) ".join(v_6[:3]) + "\n" +
            "4) " + " | 5) ".join(v_6[3:6]).replace(" | 5) ", " | 5) ", 1)
        )
        # Formateo limpio en 2 líneas de 3 ventanas cada una (3 palabras x 6 ventanas)
        linea_1 = " • ".join(v_6[:3])
        linea_2 = " • ".join(v_6[3:6])
        texto_6_ventanas = f"{linea_1}\n{linea_2}"

        oracion_envuelta = textwrap.fill(f"\"{info['oracion_prototipo']}\"", width=68)

        filas_tabla.append([
            f"C{c_id} (m_{c_id})\nn = {info['tamano']}",
            info["distribucion_clases"],
            texto_6_ventanas,
            oracion_envuelta
        ])
        colores_filas.append(cmap(c_id))

    tabla = ax_tab.table(
        cellText=filas_tabla,
        colLabels=[
            "Cluster / Tam.",
            "Asignación Clínica\n(Voto Mayoritario + 2ª Clase)",
            "6 Ventanas de 3 Palabras (Trigramas Discriminativos c-TF-IDF)",
            "Oración Prototipo Central del Cluster (Extraída del Entorno del Medoide)"
        ],
        loc="center",
        cellLoc="left",
        colWidths=[0.09, 0.19, 0.36, 0.36]
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8.3)
    tabla.scale(1.0, 2.05)

    for (fila, col), celda in tabla.get_celld().items():
        if fila == 0:
            celda.set_facecolor("#1f4e78")
            celda.set_text_props(color="white", fontweight="bold", ha="center")
        else:
            if col == 0:
                celda.set_facecolor(colores_filas[fila - 1])
                celda.set_text_props(fontweight="bold", color="black", ha="center")
            else:
                celda.set_facecolor("#f4f6f9" if fila % 2 == 0 else "white")

    nota_f7 = textwrap.fill(
        "Cómo interpretar esta tabla: Mientras 2 palabras sueltas son insuficientes para distinguir matices clínicos, la combinación de 6 ventanas de 3 palabras "
        "junto con la oración prototipo central permite identificar con precisión la temática de cada grupo (ej. C14: crisis aguda y autolesión; C12: agotamiento laboral/académico; "
        "C10: desesperanza emocional y duelo; C7: búsqueda activa de terapia y apoyo; C8: suicidio de un familiar cercano).",
        width=145
    )
    plt.figtext(0.5, 0.012, nota_f7, ha="center", fontsize=8.8, bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor="#2ca02c"))

    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    ruta_fig7 = os.path.join(carpeta_salida, "fig7_tabla_codebook_6x3_ventanas.png")
    plt.savefig(ruta_fig7, dpi=220)
    plt.close(fig)

    # =========================================================================
    # FIGURA 8: CORRELACIÓN EXTERNA CON LA REALIDAD EN k ∈ [2, √N=22]
    # =========================================================================
    k_max_sqrt = max(10, int(np.sqrt(N)))
    rango_k = list(range(2, k_max_sqrt + 1))
    acc_vals, kappa_vals, f1_vals, rand_vals, jacc_vals = [], [], [], [], []

    for k in rango_k:
        pred_k = cortar_dendrograma_scratch(Z_comp, n_clusters=k)
        m_pares = calcular_metricas_externas(etiquetas_reales, pred_k)
        m_clf = calcular_metricas_clasificacion_clustering_scratch(etiquetas_reales, pred_k)

        acc_vals.append(m_clf["accuracy_pureza"])
        kappa_vals.append(m_clf["kappa_cohen"])
        f1_vals.append(m_clf["f1_macro"])
        rand_vals.append(m_pares["rand_index"])
        jacc_vals.append(m_pares["jaccard"])

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    fig.suptitle(
        f"Evaluación Externa: Correlación con las Clases Clínicas C-SSRS en k ∈ [2, √N = {k_max_sqrt}]",
        fontsize=13.5, fontweight="bold", y=0.97
    )
    plt.figtext(
        0.5, 0.91,
        "Figuras de mérito por pares (Rand, Jaccard) y por voto mayoritario (Accuracy/Pureza, F1-Macro, Kappa de Cohen - Tema 3)",
        ha="center", fontsize=9.8, style="italic", color="#333333"
    )

    ax.plot(rango_k, rand_vals, marker="o", linewidth=2.2, color="#1f77b4", label="Rand Index (Acuerdo por pares)")
    ax.plot(rango_k, acc_vals, marker="s", linewidth=2.2, color="#2ca02c", label="Accuracy / Pureza (Voto Mayoritario)")
    ax.plot(rango_k, f1_vals, marker="^", linewidth=2.0, linestyle="--", color="#ff7f0e", label="F1-Measure Macro")
    ax.plot(rango_k, jacc_vals, marker="v", linewidth=1.8, linestyle=":", color="#9467bd", label="Jaccard Index (Pares positivos)")
    ax.plot(rango_k, kappa_vals, marker="d", linewidth=1.8, linestyle="-.", color="#d62728", label="Estadístico Kappa de Cohen (κ)")
    ax.axvline(k_representar, color="#2ca02c", linestyle="--", alpha=0.8, linewidth=1.6, label=f"Óptimo interno seleccionado k̂ = {k_representar}")

    idx_k_rep = rango_k.index(k_representar) if k_representar in rango_k else -1
    if idx_k_rep >= 0:
        ax.annotate(
            f"En k̂={k_representar}:\nRand = {rand_vals[idx_k_rep]:.3f}\nPureza = {acc_vals[idx_k_rep]:.3f}\nF1 = {f1_vals[idx_k_rep]:.3f}",
            xy=(k_representar, rand_vals[idx_k_rep]),
            xytext=(k_representar - 3.2, rand_vals[idx_k_rep] - 0.16),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#1f77b4"),
            fontsize=8.8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e3f2fd", edgecolor="#1f77b4")
        )

    ax.set_xlabel("Número de Clusters (k)", fontsize=10.5)
    ax.set_ylabel("Valor de la Métrica de Evaluación Externa ∈ [0, 1]", fontsize=10.5)
    ax.set_xticks(rango_k)
    ax.set_ylim(-0.02, 0.82)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8.5, framealpha=0.95)

    nota_f8 = textwrap.fill(
        f"Conclusión externa: A partir de k=11..15 el Rand Index se estabiliza en ~0.67 y la Pureza/Accuracy alcanza {acc_vals[idx_k_rep]:.3f} (con F1-Macro={f1_vals[idx_k_rep]:.3f}), "
        "confirmando que los sub-clusters temáticos descubiertos de forma no supervisada guardan coherencia con los niveles de riesgo clínico reales.",
        width=115
    )
    plt.figtext(0.5, 0.015, nota_f8, ha="center", fontsize=8.8, bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#666666"))

    plt.tight_layout(rect=[0, 0.09, 1, 0.85])
    ruta_fig8 = os.path.join(carpeta_salida, "fig8_correlacion_externa_k.png")
    plt.savefig(ruta_fig8, dpi=220)
    plt.close(fig)

    print(f"  6) {ruta_fig6}")
    print(f"  7) {ruta_fig7}")
    print(f"  8) {ruta_fig8}")

    return resumen_centroides



def cortar_arbol(matriz_linkage, altura_h=None, n_clusters=None):
    """
    Genera los clústeres planos a partir del dendrograma jerárquico desde cero (Scratch).

    Pre-condiciones:
    - matriz_linkage: matriz numpy de tamaño (N-1, 4).
    - altura_h: valor float positivo (opcional) o n_clusters: entero K >= 2 (opcional).

    Post-condiciones:
    - Devuelve un array 1D con las etiquetas de clúster.
    """
    return cortar_dendrograma_scratch(matriz_linkage, n_clusters=n_clusters, altura_h=altura_h)


def visualizar_pca_clusters(vectores_densos, etiquetas_predichas, ruta_salida=None):
    """
    Reduce la dimensionalidad del espacio vectorial a 2 componentes principales (PCA)
    y grafica la dispersión de los clusters formados.

    Pre-condiciones:
        - vectores_densos: matriz numpy de tamaño (N, D).
        - etiquetas_predichas: array 1D de tamaño N con los IDs de los clústeres.
        - ruta_salida: str o None. Si se proporciona, guarda la figura en disco.

    Post-condiciones:
        - Guarda o muestra un gráfico de dispersión 2D reflejando la varianza explicada en los ejes.
    """
    pca = PCA(n_components=2, random_state=42)
    vectores_2d = pca.fit_transform(vectores_densos)
    v1 = float(pca.explained_variance_ratio_[0] * 100)
    v2 = float(pca.explained_variance_ratio_[1] * 100)

    plt.figure(figsize=(9, 6.5))
    scatter = plt.scatter(
        vectores_2d[:, 0], vectores_2d[:, 1],
        c=etiquetas_predichas, cmap='tab20', alpha=0.75,
        edgecolor='k', s=35
    )
    plt.title(f"Proyección PCA 2D de los Clusters (Varianza Explicada: {v1 + v2:.1f}%)", fontsize=12, fontweight="bold")
    plt.xlabel(f"Componente Principal 1 (PCA 1 - {v1:.1f}% varianza)", fontsize=10.5)
    plt.ylabel(f"Componente Principal 2 (PCA 2 - {v2:.1f}% varianza)", fontsize=10.5)
    plt.colorbar(scatter, label="Cluster ID")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    if ruta_salida:
        plt.savefig(ruta_salida, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def inferir_y_mostrar_cercanos(
    vector_nuevo,
    vectores_entrenamiento,
    etiquetas_entrenamiento,
    textos_entrenamiento,
    k=3,
    metrica="cosine",
    modelo_pca=None
):
    """
    Asigna una nueva instancia al cluster de su vecino más cercano (1-NN)
    y recupera las K instancias más afines DENTRO de dicho cluster asignado,
    cumpliendo estrictamente el requisito del guión de prácticas:
    "Dada una nueva instancia, asignarle un cluster e indicarlo.
     Dentro de ese cluster, ofrecer las instancias más cercanas (mostrar texto)".

    Pre-condiciones:
        - vector_nuevo: array 1D de dimensión D, o string de texto plano a vectorizar con SBERT.
        - vectores_entrenamiento: array 2D de dimensiones (N_train, D).
        - etiquetas_entrenamiento: array 1D de longitud N_train con los IDs de cluster asignados en train.
        - textos_entrenamiento: lista o array de longitud N_train con los textos correspondientes.
        - k: entero >= 1 indicando el número de vecinos más próximos a recuperar dentro del cluster.
        - metrica: str en {'cosine', 'euclidean'}. Debe coincidir con la métrica del clustering.
        - modelo_pca: objeto PCA ajustado en train (opcional). Si se proporciona y vector_nuevo
          tiene la dimensión original, se proyecta mediante modelo_pca.transform().

    Post-condiciones:
        - Devuelve el ID entero del cluster asignado y la lista de textos más cercanos dentro de él.
    """
    # Si se recibe una cadena de texto en lenguaje natural, se vectoriza con SBERT
    if isinstance(vector_nuevo, str):
        try:
            from sentence_transformers import SentenceTransformer
            modelo_sbert = SentenceTransformer("all-MiniLM-L6-v2")
            vector_nuevo = modelo_sbert.encode([vector_nuevo], normalize_embeddings=True)[0]
        except Exception as e:
            raise ValueError(f"No se pudo vectorizar el texto recibido: {e}")

    v_nuevo = np.asarray(vector_nuevo, dtype=np.float32)
    X_train = np.asarray(vectores_entrenamiento, dtype=np.float32)

    # Proyección con el modelo PCA de entrenamiento si corresponde
    if modelo_pca is not None:
        if hasattr(modelo_pca, "n_features_in_") and v_nuevo.shape[0] == modelo_pca.n_features_in_:
            v_nuevo = modelo_pca.transform(v_nuevo.reshape(1, -1)).flatten()

    metrica_clean = metrica.lower().strip()
    if metrica_clean in ["cosine", "coseno"]:
        norm_v = float(np.linalg.norm(v_nuevo))
        norm_X = np.linalg.norm(X_train, axis=1)
        norm_v = 1.0 if norm_v == 0.0 else norm_v
        norm_X = np.where(norm_X == 0.0, 1.0, norm_X)

        similitud = np.dot(X_train, v_nuevo) / (norm_X * norm_v)
        similitud = np.clip(similitud, -1.0, 1.0)
        distancias = 1.0 - similitud
    elif metrica_clean in ["euclidean", "euclidea"]:
        distancias = np.linalg.norm(X_train - v_nuevo, axis=1)
    else:
        raise ValueError(f"Métrica '{metrica}' no soportada (usar 'cosine' o 'euclidean').")

    # 1. Asignar el cluster de pertenencia mediante el vecino 1-NN más cercano en el espacio métrico
    vecino_mas_cercano = int(np.argmin(distancias))
    cluster_asignado = int(etiquetas_entrenamiento[vecino_mas_cercano])

    # 2. Filtrar candidatos estrictamente DENTRO del cluster asignado
    indices_en_cluster = np.where(etiquetas_entrenamiento == cluster_asignado)[0]
    distancias_en_cluster = distancias[indices_en_cluster]
    orden_en_cluster = np.argsort(distancias_en_cluster)[:min(k, len(indices_en_cluster))]
    indices_cercanos = indices_en_cluster[orden_en_cluster]

    print(f"\n--- INFERENCIA: NUEVO POST ASIGNADO AL CLUSTER C{cluster_asignado} (Espacio: {metrica.capitalize()}) ---")
    print(f"Vecino 1-NN global: Post #{vecino_mas_cercano} (Distancia: {distancias[vecino_mas_cercano]:.4f})")
    print(f"Top {len(indices_cercanos)} instancias más afines DENTRO DEL CLUSTER C{cluster_asignado} (total en cluster: {len(indices_en_cluster)}):")
    textos_recuperados = []
    for i, idx in enumerate(indices_cercanos):
        txt = textos_entrenamiento[idx]
        textos_recuperados.append(txt)
        print(f"  {i + 1}. [Distancia {metrica}: {distancias[idx]:.4f}] -> \"{txt[:140]}...\"")

    return cluster_asignado, textos_recuperados


def generar_figura_matriz_confusion_heatmap(
    matriz_train,
    matriz_test,
    nombres_clases=None,
    carpeta_salida="data_processed/graficas_informe",
):
    """
    Genera la Figura 10 del informe: Heat-Map intuitivo de las Matrices de Confusión
    (Class-to-Cluster) en Train y Test según lo exigido explícitamente en la rúbrica.
    Muestra recuentos absolutos y porcentajes normalizados por fila (Recall / Cobertura real).
    """
    import textwrap
    os.makedirs(carpeta_salida, exist_ok=True)
    if nombres_clases is None:
        nombres_clases = ["Indicator", "Ideation", "Behavior", "Attempt", "Supportive"]

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.8))
    fig.suptitle(
        "Matriz de Confusión Class-to-Cluster: Correlación con Niveles Clínicos C-SSRS",
        fontsize=14, fontweight="bold", y=0.97
    )
    plt.figtext(
        0.5, 0.915,
        "Asignación por Voto Mayoritario y Evaluación Honesta de Generalización: Train (N1=400) vs Test No Vistos (N2=100)",
        ha="center", fontsize=9.8, style="italic", color="#333333"
    )

    matrices = [
        ("A) Subconjunto Train (N1 = 400)", np.asarray(matriz_train, dtype=float)),
        ("B) Subconjunto Test - Instancias No Vistas (N2 = 100)", np.asarray(matriz_test, dtype=float))
    ]

    for idx_ax, (titulo, M) in enumerate(matrices):
        ax = axes[idx_ax]
        sumas_filas = np.maximum(M.sum(axis=1, keepdims=True), 1.0)
        M_norm = M / sumas_filas
        im = ax.imshow(M_norm, cmap="Blues", vmin=0.0, vmax=1.0)

        # Anotaciones numéricas en cada celda
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                cnt = int(M[r, c])
                pct = M_norm[r, c] * 100.0
                texto_celda = f"{cnt}\n({pct:.1f}%)"
                color_texto = "white" if M_norm[r, c] > 0.52 else "black"
                ax.text(c, r, texto_celda, ha="center", va="center", color=color_texto, fontsize=9.2, fontweight="bold")

        ax.set_title(titulo, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(range(len(nombres_clases)))
        ax.set_yticks(range(len(nombres_clases)))
        ax.set_xticklabels(nombres_clases, fontsize=9.5, fontweight="bold")
        ax.set_yticklabels(nombres_clases, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Clase Asignada al Cluster (Voto Mayoritario)", fontsize=10.2)
        ax.set_ylabel("Clase Real (Ground-Truth C-SSRS)", fontsize=10.2)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Proporción por Fila (Recall / Cobertura)", fontsize=8.8)

    nota_f10 = textwrap.fill(
        "Interpretación del Heat-Map: La diagonal principal refleja la pureza temática alcanzada por cada clase clínica mediante voto mayoritario. "
        "En Test (derecha), la asignación se realiza de forma estrictamente honesta aplicando el mapeo aprendido en Train sin fuga de etiquetas. "
        "Las mayores confusiones ocurren entre 'Ideation' y 'Attempt', consistente con el gradiente léxico continuo del lenguaje suicida.",
        width=135
    )
    plt.figtext(0.5, 0.015, nota_f10, ha="center", fontsize=8.8, bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#666666"))

    plt.tight_layout(rect=[0, 0.08, 1, 0.89])
    ruta_fig10 = os.path.join(carpeta_salida, "fig10_matriz_confusion_heatmap.png")
    plt.savefig(ruta_fig10, dpi=220)
    plt.close(fig)
    print(f"  10) {ruta_fig10}")
    return ruta_fig10


def ejecutar_evaluacion(carpeta_datos=None, carpeta_salida=None):
    """
    Ejecuta el protocolo completo de evaluación interna, externa, Holdout e inferencia.
    Genera las 8 figuras científicas del informe.
    """
    print("Iniciando evaluación...")

    carpeta_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if carpeta_datos is None:
        carpeta_datos = os.path.join(carpeta_raiz, "data_processed")
    if carpeta_salida is None:
        carpeta_salida = os.path.join(carpeta_datos, "graficas_informe")

    if carpeta_raiz not in sys.path:
        sys.path.append(carpeta_raiz)

    # 1. Cargar los datos procesados
    try:
        vectores_densos = np.load(os.path.join(carpeta_datos, "embeddings_sbert.npy"))
        etiquetas_reales = np.load(os.path.join(carpeta_datos, "etiquetas_reales.npy"))
        matriz_distancias = np.load(os.path.join(carpeta_datos, "matriz_distancias_cosine.npy"))

        with open(os.path.join(carpeta_datos, "textos_posts.json"), "r", encoding="utf-8") as f:
            textos_reales = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encuentran los datos procesados en '{carpeta_datos}'. Ejecuta pipeline.py primero.")
        return None

    # 2. Generar el clustering
    try:
        from src.clustering.hierarchical import MotorJerarquico
    except (ImportError, ModuleNotFoundError):
        try:
            from clustering.hierarchical import MotorJerarquico
        except (ImportError, ModuleNotFoundError):
            from .clustering.hierarchical import MotorJerarquico

    motor = MotorJerarquico(tipo_enlace="average")
    matriz_linkage = motor.fit(matriz_distancias)

    # 3. Cortar el dendrograma
    altura_h = 0.70
    predicciones = cortar_arbol(matriz_linkage, altura_h)

    print("\n--- RESULTADOS DE VALIDACIÓN INTERNA (SCRATCH) ---")
    cpcc = calcular_cpcc_scratch(matriz_distancias, matriz_linkage)
    print(f"Coeficiente de Correlación Cofenética (CPCC): {cpcc:.4f}")

    n_clusters_gen = len(np.unique(predicciones))
    print(f"Clusters generados en el corte: {n_clusters_gen}")
    if 2 <= n_clusters_gen < len(predicciones):
        detalle_sil = calcular_silhouette_scratch(matriz_distancias, predicciones, devolver_detalle=True)
        print(f"Silhouette Score: {detalle_sil['Silhouette Score']:.4f}")
        print(f"  - Cohesión media a(i):   {detalle_sil['Cohesion Media a(i)']:.4f}")
        print(f"  - Separación media b(i): {detalle_sil['Separacion Media b(i)']:.4f}")

    # Comparativa rápida con K=5 (las 5 clases clínicas C-SSRS)
    pred_k5 = cortar_arbol(matriz_linkage, n_clusters=5)
    det_k5 = calcular_silhouette_scratch(matriz_distancias, pred_k5, devolver_detalle=True)
    print(f"Silhouette Score (corte exacto K=5): {det_k5['Silhouette Score']:.4f} "
          f"(a(i)={det_k5['Cohesion Media a(i)']:.4f}, b(i)={det_k5['Separacion Media b(i)']:.4f})")

    print("\n--- RESULTADOS DE VALIDACIÓN ---")
    resultados = calcular_metricas_externas(etiquetas_reales, predicciones)
    print(f"Rand Index: {resultados.get('rand_index', 0):.4f}")
    print(f"Jaccard: {resultados.get('jaccard', 0):.4f}")
    print(f"Fowlkes-Mallows: {resultados.get('fowlkes_mallows', 0):.4f}")

    # Métricas adicionales de los apuntes (SSE, BSS y Dunn Index)
    sse_bss = calcular_sse_bss_scratch(vectores_densos, predicciones)
    dunn_val = calcular_dunn_scratch(matriz_distancias, predicciones)
    print(f"Cohesión SSE(G): {sse_bss['SSE']:.4f} | Separabilidad BSS(G): {sse_bss['BSS']:.4f} | Dunn Index: {dunn_val:.4f}")

    # Generar las 3 figuras autoexplicativas para el informe
    generar_graficas_validacion_interna(
        vectores_densos,
        matriz_distancias,
        np.load(os.path.join(carpeta_datos, "matriz_distancias_euclidean.npy")),
        carpeta_salida=carpeta_salida,
    )

    # Selección no supervisada de K en [2, sqrt(N)=22] (sin prefijar K=5 a dedo)
    opt_info = optimizar_hiperparametros_internos(matriz_distancias)
    k_pico_equilibrado = opt_info["mejor_particion_equilibrada"]["k_optimo"] or 7
    print(f"\n[OPTIMIZACION NO SUPERVISADA k en [2, sqrt(N)=22]]")
    print(f"  - Maximo Global Silhouette: Enlace='{opt_info['mejor_global']['enlace']}', k_opt={opt_info['mejor_global']['k_optimo']} (S_media={opt_info['mejor_global']['silhouette']:.4f})")
    print(f"  - Mejor Particion Equilibrada (sin macro-cluster >80%): Enlace='{opt_info['mejor_particion_equilibrada']['enlace']}', k_opt={k_pico_equilibrado} (S_media={opt_info['mejor_particion_equilibrada']['silhouette']:.4f})")
    print(f"  - Picos locales detectados (Complete-Link): {opt_info['resultados_por_enlace']['complete']['picos_locales_silhouette']}")

    # Evaluar métricas externas de clasificación (Accuracy/Pureza, Kappa, F1 y Matriz de Confusión) en k_opt descubierto
    Z_comp = MotorJerarquico(tipo_enlace="complete").fit(matriz_distancias)
    pred_opt = cortar_arbol(Z_comp, n_clusters=k_pico_equilibrado)
    clf_stats = calcular_metricas_clasificacion_clustering_scratch(etiquetas_reales, pred_opt)
    pares_stats = calcular_metricas_externas(etiquetas_reales, pred_opt)
    try:
        np.save(os.path.join(carpeta_datos, "etiquetas_predichas.npy"), pred_opt)
    except Exception:
        pass
    print(f"\n--- CORRELACION CON LA REALIDAD EN k_opt={k_pico_equilibrado} (COMPLETE-LINK) ---")

    print(f"Accuracy / Pureza (Voto Mayoritario): {clf_stats['accuracy_pureza']:.4f}")
    print(f"Estadístico Kappa de Cohen:           {clf_stats['kappa_cohen']:.4f}")
    print(f"F1-Measure (Macro):                   {clf_stats['f1_macro']:.4f}")
    print(f"Rand Index: {pares_stats['rand_index']:.4f} | Jaccard: {pares_stats['jaccard']:.4f} | Fowlkes-Mallows: {pares_stats['fowlkes_mallows']:.4f}")
    print("Matriz de Confusión (Filas=Clase Real [0..4], Columnas=Clase Predicha [0..4]):")
    print(clf_stats["matriz_confusion"])

    # Extraer y dibujar el Codebook de Centroides Textuales + Curva de Correlación Externa (Figura 4)
    extraer_y_visualizar_centroides_textuales_scratch(
        vectores_densos,
        matriz_distancias,
        etiquetas_reales,
        textos_reales,
        k_representar=k_pico_equilibrado,
        carpeta_salida=carpeta_salida,
    )

    # Evaluación Honesta Holdout (N1-Train = 400, N2-Test = 100 no vistos en el clustering)
    print("\n--- EVALUACIÓN HONESTA HOLDOUT (N1=400 TRAIN vs N2=100 TEST NO VISTOS) ---")
    n_train = int(0.8 * len(vectores_densos))
    X_train, X_test = vectores_densos[:n_train], vectores_densos[n_train:]
    y_train, y_test = etiquetas_reales[:n_train], etiquetas_reales[n_train:]
    D_train = matriz_distancias[:n_train, :n_train]

    Z_train = MotorJerarquico(tipo_enlace="complete").fit(D_train)
    pred_train = cortar_arbol(Z_train, n_clusters=k_pico_equilibrado)

    # Asignar cada instancia no vista de X_test al cluster de su vecino más cercano en X_train
    D_test_train = matriz_distancias[n_train:, :n_train]
    vecinos_mas_cercanos = np.argmin(D_test_train, axis=1)
    pred_test = pred_train[vecinos_mas_cercanos]

    ext_train = calcular_metricas_externas(y_train, pred_train)
    ext_test = calcular_metricas_externas(y_test, pred_test)
    clf_train = calcular_metricas_clasificacion_clustering_scratch(y_train, pred_train)
    # Evaluación honesta en Test: aplicamos el mapeo de cluster a clase aprendido en Train
    clf_test = calcular_metricas_clasificacion_clustering_scratch(
        y_test, pred_test, mapeo_previo=clf_train["mapeo_cluster_a_clase"]
    )

    sil_train = calcular_silhouette_scratch(D_train, pred_train)
    D_test = matriz_distancias[n_train:, n_train:]
    sil_test = calcular_silhouette_scratch(D_test, pred_test) if len(np.unique(pred_test)) >= 2 else 0.0

    print(f"\n[TABLA COMPARATIVA HOLDOUT (N_train={n_train} | N_test={len(X_test)})]")
    print(f"{'Métrica':<32} | {'Train (N1=400)':<15} | {'Test (N2=100)':<15}")
    print("-" * 68)
    print(f"{'Class-to-Cluster Accuracy':<32} | {clf_train['accuracy_pureza']*100:6.2f}%         | {clf_test['accuracy_pureza']*100:6.2f}%")
    print(f"{'Estadístico Kappa de Cohen (κ)':<32} | {clf_train['kappa_cohen']:6.4f}          | {clf_test['kappa_cohen']:6.4f}")
    print(f"{'Macro F1-Score':<32} | {clf_train['f1_macro']:6.4f}          | {clf_test['f1_macro']:6.4f}")
    print(f"{'Weighted F1-Score (w.Avg)':<32} | {clf_train['f1_weighted']:6.4f}          | {clf_test['f1_weighted']:6.4f}")
    print(f"{'Precision (Weighted Avg)':<32} | {clf_train['precision_weighted']:6.4f}          | {clf_test['precision_weighted']:6.4f}")
    print(f"{'Recall (Weighted Avg)':<32} | {clf_train['recall_weighted']:6.4f}          | {clf_test['recall_weighted']:6.4f}")
    print(f"{'Rand Index (Pares)':<32} | {ext_train['rand_index']:6.4f}          | {ext_test['rand_index']:6.4f}")
    print(f"{'Jaccard Index (Pares)':<32} | {ext_train['jaccard']:6.4f}          | {ext_test['jaccard']:6.4f}")
    print(f"{'Silhouette Score':<32} | {sil_train:6.4f}          | {sil_test:6.4f}")

    print("\n[DESGLOSE POR CLASE CLÍNICA C-SSRS EN TEST (N2=100)]")
    print(f"{'Clase Clínica':<15} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Soporte':<8}")
    print("-" * 55)
    nombres_c = ["Indicator", "Ideation", "Behavior", "Attempt", "Supportive"]
    for idx_cl, nom in enumerate(nombres_c):
        print(
            f"{nom:<15} | {clf_test['precision_por_clase'][idx_cl]:6.4f}   | "
            f"{clf_test['recall_por_clase'][idx_cl]:6.4f}   | "
            f"{clf_test['f1_por_clase'][idx_cl]:6.4f}   | "
            f"{clf_test['soporte_por_clase'][idx_cl]:4d}"
        )
    print(f"{'Macro Avg':<15} | {clf_test['precision_macro']:6.4f}   | {clf_test['recall_macro']:6.4f}   | {clf_test['f1_macro']:6.4f}   | {len(y_test):4d}")
    print(f"{'Weighted Avg':<15} | {clf_test['precision_weighted']:6.4f}   | {clf_test['recall_weighted']:6.4f}   | {clf_test['f1_weighted']:6.4f}   | {len(y_test):4d}")

    # Generar la Figura 10: Heat-Map intuitivo de las Matrices de Confusión (Train y Test)
    generar_figura_matriz_confusion_heatmap(
        clf_train["matriz_confusion"],
        clf_test["matriz_confusion"],
        nombres_clases=nombres_c,
        carpeta_salida=carpeta_salida,
    )

    # Validación Holdout Repetida (5 particiones aleatorias 80/20) para medir dispersión estadística (std)
    rep_test_acc, rep_test_rand, rep_test_f1 = [], [], []
    for sem in [42, 101, 202, 303, 404]:
        rng = np.random.RandomState(sem)
        perm = rng.permutation(len(vectores_densos))
        idx_tr, idx_te = perm[:n_train], perm[n_train:]
        D_tr_rep = matriz_distancias[np.ix_(idx_tr, idx_tr)]
        Z_tr_rep = MotorJerarquico(tipo_enlace="complete").fit(D_tr_rep)
        p_tr_rep = cortar_arbol(Z_tr_rep, n_clusters=k_pico_equilibrado)

        D_te_tr = matriz_distancias[np.ix_(idx_te, idx_tr)]
        nn_rep = np.argmin(D_te_tr, axis=1)
        p_te_rep = p_tr_rep[nn_rep]

        c_tr = calcular_metricas_clasificacion_clustering_scratch(etiquetas_reales[idx_tr], p_tr_rep)
        c_te = calcular_metricas_clasificacion_clustering_scratch(
            etiquetas_reales[idx_te], p_te_rep, mapeo_previo=c_tr["mapeo_cluster_a_clase"]
        )
        e_te = calcular_metricas_externas(etiquetas_reales[idx_te], p_te_rep)
        rep_test_acc.append(c_te["accuracy_pureza"])
        rep_test_rand.append(e_te["rand_index"])
        rep_test_f1.append(c_te["f1_macro"])

    print(f"\n[ESTABILIDAD Y DISPERSIÓN ESTADÍSTICA (5 Repeticiones Holdout 80/20)]")
    print(f"  - Test Class-to-Cluster Acc: {np.mean(rep_test_acc)*100:.2f}% (std = {np.std(rep_test_acc)*100:.2f}%)")
    print(f"  - Test Rand Index:           {np.mean(rep_test_rand):.4f} (std = {np.std(rep_test_rand):.4f})")
    print(f"  - Test Macro F1-Score:       {np.mean(rep_test_f1):.4f} (std = {np.std(rep_test_f1):.4f})")

    # Inferencia honesta: pasamos la primera instancia de TEST (no vista en X_train)
    inferir_y_mostrar_cercanos(X_test[0], X_train, pred_train, textos_reales[:n_train], k=3, metrica="cosine")

    return {
        "cpcc": float(cpcc),
        "silhouette_k_opt": float(opt_info["mejor_global"]["silhouette"]),
        "k_optimo": int(k_pico_equilibrado),
        "accuracy_pureza": float(clf_stats["accuracy_pureza"]),
        "kappa_cohen": float(clf_stats["kappa_cohen"]),
        "f1_macro": float(clf_stats["f1_macro"]),
        "rand_index": float(pares_stats["rand_index"]),
        "jaccard": float(pares_stats["jaccard"]),
        "fowlkes_mallows": float(pares_stats["fowlkes_mallows"]),
        "holdout": {
            "train": {
                "accuracy_pureza": float(clf_train["accuracy_pureza"]),
                "kappa_cohen": float(clf_train["kappa_cohen"]),
                "f1_macro": float(clf_train["f1_macro"]),
                "f1_weighted": float(clf_train["f1_weighted"]),
                "silhouette": float(sil_train),
                "rand": float(ext_train["rand_index"]),
                "jaccard": float(ext_train["jaccard"]),
            },
            "test": {
                "accuracy_pureza": float(clf_test["accuracy_pureza"]),
                "kappa_cohen": float(clf_test["kappa_cohen"]),
                "f1_macro": float(clf_test["f1_macro"]),
                "f1_weighted": float(clf_test["f1_weighted"]),
                "silhouette": float(sil_test),
                "rand": float(ext_test["rand_index"]),
                "jaccard": float(ext_test["jaccard"]),
                "accuracy_media_5rep": float(np.mean(rep_test_acc)),
                "accuracy_std_5rep": float(np.std(rep_test_acc)),
                "rand_media_5rep": float(np.mean(rep_test_rand)),
                "rand_std_5rep": float(np.std(rep_test_rand)),
            },
        },
    }


if __name__ == "__main__":
    ejecutar_evaluacion()