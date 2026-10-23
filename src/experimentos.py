"""
text-clustering-MD: Experimentos Científicos y Evidencias Cuantitativas
=============================================================================
Aporta las evidencias experimentales exigidas explícitamente en el guión de prácticas
y en las observaciones de la rúbrica de evaluación:

1. Dimensión del Espacio de Atributos:
   ¿Conviene hacer el clustering antes o después de reducir la dimensión?
   ¿Cuál es la dimensión que mejor calidad ofrece según class-to-cluster accuracy?
   (Evaluación sistemática de D in [2, 5, 10, 25, 50, 100, 200, 384]).

2. Pre-proceso y Normalización:
   Evidencias experimentales de la repercusión de la normalización L2 unitaria
   frente a embeddings crudos no normalizados (contraste relativo, CPCC, silueta).

3. Representación Óptima (Aproximaciones Dispersas vs Densas):
   Comparativa formal entre TF-IDF (disperso de alta dimensión) y SBERT (denso contextual).

4. Referencia Random (Baseline nulo):
   Contraste estadístico de las métricas obtenidas frente a una asignación aleatoria uniforme.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    ensure_directories,
)
from src.embeddings import (
    generar_embeddings,
    calcular_matriz_distancias_scratch,
)
from src.clustering.hierarchical import MotorJerarquico
from src.metricas_internas import (
    calcular_cpcc_scratch,
    calcular_silhouette_scratch,
    cortar_dendrograma_scratch,
)
from src.evaluacion import (
    calcular_metricas_externas,
    calcular_metricas_clasificacion_clustering_scratch,
)


def experimento_impacto_dimensionalidad(X, y_true, k_optimo=15, dimensiones=(2, 5, 10, 25, 50, 100, 200, 384)):
    """
    Evalúa empíricamente si conviene hacer el clustering antes o después de reducir
    la dimensión del espacio mediante PCA, y qué dimensión ofrece mejor class-to-cluster accuracy.
    """
    print("\n[EXPERIMENTO 1] Impacto de la Dimensión del Espacio (PCA vs Original 384D)...")
    N, D_orig = X.shape
    resultados_dim = []

    for d in dimensiones:
        t0 = time.time()
        if d < D_orig:
            pca = PCA(n_components=d, random_state=42)
            X_red = pca.fit_transform(X)
            varianza_exp = float(np.sum(pca.explained_variance_ratio_))
        else:
            X_red = X
            varianza_exp = 1.0

        # Calcular distancias en el espacio de dimension d
        D_cos = calcular_matriz_distancias_scratch(X_red, metrica="cosine")
        motor = MotorJerarquico(tipo_enlace="complete")
        Z = motor.fit(D_cos)

        pred = cortar_dendrograma_scratch(Z, n_clusters=k_optimo)
        cpcc = calcular_cpcc_scratch(D_cos, Z)
        sil = calcular_silhouette_scratch(D_cos, pred)
        clf = calcular_metricas_clasificacion_clustering_scratch(y_true, pred)
        ext = calcular_metricas_externas(y_true, pred)
        t_total = time.time() - t0

        registro = {
            "dimension_D": int(d),
            "varianza_explicada_acumulada": round(varianza_exp, 4),
            "cpcc": round(float(cpcc), 4),
            "silhouette": round(float(sil), 4),
            "accuracy_pureza": round(float(clf["accuracy_pureza"]), 4),
            "f1_macro": round(float(clf["f1_macro"]), 4),
            "rand_index": round(float(ext["rand_index"]), 4),
            "tiempo_seg": round(t_total, 3),
        }
        resultados_dim.append(registro)
        print(f"  D = {d:3d} (Varianza: {varianza_exp*100:5.1f}%) -> Silueta: {sil:6.4f} | Pureza: {clf['accuracy_pureza']*100:5.2f}% | CPCC: {cpcc:6.4f} | Tiempo: {t_total:5.2f}s")

    return resultados_dim


def experimento_impacto_normalizacion(textos, y_true, X_norm=None, k_optimo=15):
    """
    Aporta evidencias cuantitativas sobre el efecto de la normalización L2 unitaria
    frente a representaciones no normalizadas (contraste métrico, CPCC y pureza).
    """
    print("\n[EXPERIMENTO 2] Repercusión Cuantitativa de la Normalización L2...")
    # 1. Sin normalizar
    print("  Codificando SBERT sin normalización L2...")
    X_raw = generar_embeddings(textos, normalizar=False)
    D_euc_raw = calcular_matriz_distancias_scratch(X_raw, metrica="euclidean")
    D_cos_raw = calcular_matriz_distancias_scratch(X_raw, metrica="cosine")

    # 2. Con normalización
    if X_norm is None:
        print("  Codificando SBERT con normalización L2...")
        X_norm = generar_embeddings(textos, normalizar=True)
    D_euc_norm = calcular_matriz_distancias_scratch(X_norm, metrica="euclidean")
    D_cos_norm = calcular_matriz_distancias_scratch(X_norm, metrica="cosine")


    # Contraste relativo C_R = (d_max - d_min) / d_min
    filas, cols = np.triu_indices(len(textos), k=1)

    def stats_dist(D_mat):
        pares = D_mat[filas, cols]
        d_min, d_max = float(np.min(pares)), float(np.max(pares))
        c_r = (d_max - d_min) / (d_min if d_min > 1e-6 else 1e-6)
        return {"media": float(np.mean(pares)), "std": float(np.std(pares)), "contraste_relativo": float(c_r)}

    # Clustering en ambos
    Z_raw = MotorJerarquico(tipo_enlace="complete").fit(D_cos_raw)
    pred_raw = cortar_dendrograma_scratch(Z_raw, n_clusters=k_optimo)
    clf_raw = calcular_metricas_clasificacion_clustering_scratch(y_true, pred_raw)
    sil_raw = calcular_silhouette_scratch(D_cos_raw, pred_raw)
    cpcc_raw = calcular_cpcc_scratch(D_cos_raw, Z_raw)

    Z_norm = MotorJerarquico(tipo_enlace="complete").fit(D_cos_norm)
    pred_norm = cortar_dendrograma_scratch(Z_norm, n_clusters=k_optimo)
    clf_norm = calcular_metricas_clasificacion_clustering_scratch(y_true, pred_norm)
    sil_norm = calcular_silhouette_scratch(D_cos_norm, pred_norm)
    cpcc_norm = calcular_cpcc_scratch(D_cos_norm, Z_norm)

    res = {
        "sin_normalizar": {
            "distancia_euclidea": stats_dist(D_euc_raw),
            "distancia_coseno": stats_dist(D_cos_raw),
            "cpcc": round(float(cpcc_raw), 4),
            "silhouette": round(float(sil_raw), 4),
            "accuracy_pureza": round(float(clf_raw["accuracy_pureza"]), 4),
            "f1_macro": round(float(clf_raw["f1_macro"]), 4),
        },
        "con_normalizacion_l2": {
            "distancia_euclidea": stats_dist(D_euc_norm),
            "distancia_coseno": stats_dist(D_cos_norm),
            "cpcc": round(float(cpcc_norm), 4),
            "silhouette": round(float(sil_norm), 4),
            "accuracy_pureza": round(float(clf_norm["accuracy_pureza"]), 4),
            "f1_macro": round(float(clf_norm["f1_macro"]), 4),
        },
    }

    print(f"  [Sin Normalizar] Contraste Euclídeo: {res['sin_normalizar']['distancia_euclidea']['contraste_relativo']:.2f} | Silueta: {sil_raw:.4f} | Pureza: {clf_raw['accuracy_pureza']*100:.2f}%")
    print(f"  [Con Normaliz.]  Contraste Euclídeo: {res['con_normalizacion_l2']['distancia_euclidea']['contraste_relativo']:.2f} | Silueta: {sil_norm:.4f} | Pureza: {clf_norm['accuracy_pureza']*100:.2f}%")
    return res


def experimento_representacion_tfidf_vs_sbert(textos, y_true, k_optimo=15):
    """
    Compara la aproximación dispersa clásica (TF-IDF) frente a la densa contextual (SBERT).
    Evalúa dimensionalidad, esparsidad, CPCC, Silueta y Pureza.
    """
    print("\n[EXPERIMENTO 3] Comparativa Representación Dispersa (TF-IDF) vs Densa (SBERT)...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2)
    X_tfidf_sparse = vectorizer.fit_transform(textos)
    X_tfidf = X_tfidf_sparse.toarray().astype(np.float32)
    t_tfidf = time.time() - t0

    N, D_vocab = X_tfidf.shape
    esparsidad = float(np.sum(X_tfidf == 0.0) / (N * D_vocab)) * 100.0

    # Distancia Coseno para TF-IDF
    D_cos_tfidf = calcular_matriz_distancias_scratch(X_tfidf, metrica="cosine")
    motor_tfidf = MotorJerarquico(tipo_enlace="complete")
    Z_tfidf = motor_tfidf.fit(D_cos_tfidf)
    pred_tfidf = cortar_dendrograma_scratch(Z_tfidf, n_clusters=k_optimo)

    cpcc_tfidf = calcular_cpcc_scratch(D_cos_tfidf, Z_tfidf)
    sil_tfidf = calcular_silhouette_scratch(D_cos_tfidf, pred_tfidf)
    clf_tfidf = calcular_metricas_clasificacion_clustering_scratch(y_true, pred_tfidf)
    ext_tfidf = calcular_metricas_externas(y_true, pred_tfidf)

    res = {
        "tfidf_disperso": {
            "dimension": int(D_vocab),
            "esparsidad_porcentaje": round(esparsidad, 2),
            "tiempo_extraccion_seg": round(t_tfidf, 4),
            "cpcc": round(float(cpcc_tfidf), 4),
            "silhouette": round(float(sil_tfidf), 4),
            "accuracy_pureza": round(float(clf_tfidf["accuracy_pureza"]), 4),
            "f1_macro": round(float(clf_tfidf["f1_macro"]), 4),
            "rand_index": round(float(ext_tfidf["rand_index"]), 4),
        }
    }
    print(f"  [TF-IDF Disperso] Dim: {D_vocab} | Esparsidad: {esparsidad:.1f}% | Silueta: {sil_tfidf:.4f} | Pureza: {clf_tfidf['accuracy_pureza']*100:.2f}% | CPCC: {cpcc_tfidf:.4f}")
    return res


def experimento_referencia_random(y_true, k_optimo=15, n_iteraciones=30):
    """
    Establece la Referencia Random (Baseline nulo) asignando etiquetas aleatorias
    uniformes para demostrar la significancia estadística del clustering frente al azar.
    """
    print("\n[EXPERIMENTO 4] Referencia Random (Baseline Nulo de Comparación)...")
    N = len(y_true)
    rands, jaccards, purities, kappas, f1s = [], [], [], [], []

    for _ in range(n_iteraciones):
        pred_rnd = np.random.randint(0, k_optimo, size=N)
        ext = calcular_metricas_externas(y_true, pred_rnd)
        clf = calcular_metricas_clasificacion_clustering_scratch(y_true, pred_rnd)
        rands.append(ext["rand_index"])
        jaccards.append(ext["jaccard"])
        purities.append(clf["accuracy_pureza"])
        kappas.append(clf["kappa_cohen"])
        f1s.append(clf["f1_macro"])

    res = {
        "rand_index_media": round(float(np.mean(rands)), 4),
        "rand_index_std": round(float(np.std(rands)), 4),
        "jaccard_media": round(float(np.mean(jaccards)), 4),
        "purity_media": round(float(np.mean(purities)), 4),
        "kappa_media": round(float(np.mean(kappas)), 4),
        "f1_macro_media": round(float(np.mean(f1s)), 4),
    }
    print(f"  [Random Baseline] Rand Index: {res['rand_index_media']:.4f} (std={res['rand_index_std']:.4f}) | Pureza: {res['purity_media']*100:.2f}% | Kappa: {res['kappa_media']:.4f}")
    return res


def generar_figura_evidencias_experimentales(res_dim, res_norm, res_tfidf, sbert_stats, carpeta_salida=REPORTS_DIR):
    """
    Genera la Figura 9 del informe: Evidencias Experimentales de Dimensión, Preproceso y Representación.
    """
    ensure_directories()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # Panel A: Impacto de la Dimensión PCA (Curva de Silueta y Pureza vs D)
    dims = [r["dimension_D"] for r in res_dim]
    sils = [r["silhouette"] for r in res_dim]
    accs = [r["accuracy_pureza"] * 100 for r in res_dim]

    color_sil = "#1f77b4"
    color_acc = "#d95f02"

    ax1 = axes[0]
    ax1.plot(dims, sils, marker="o", color=color_sil, linewidth=2.2, label="Silhouette Score")
    ax1.set_xlabel("Dimensión del Espacio de Atributos (D)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Silhouette Score", color=color_sil, fontsize=10, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_sil)
    ax1.set_title("A) Impacto de la Dimensión (PCA vs Original 384D)\n¿Conviene reducir dimensión antes del clustering?", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(dims, accs, marker="s", color=color_acc, linewidth=2.0, linestyle="--", label="Pureza (Accuracy %)")
    ax1_twin.set_ylabel("Pureza / Clustering Accuracy (%)", color=color_acc, fontsize=10, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor=color_acc)

    # Panel B: Comparativa de Representaciones (TF-IDF vs SBERT Raw vs SBERT Norm)
    modelos = ["TF-IDF\n(Disperso)", "SBERT Raw\n(Sin Norm)", "SBERT Norm\n(L2 Unitario)"]
    pures = [
        res_tfidf["tfidf_disperso"]["accuracy_pureza"] * 100,
        res_norm["sin_normalizar"]["accuracy_pureza"] * 100,
        res_norm["con_normalizacion_l2"]["accuracy_pureza"] * 100,
    ]
    sil_vals = [
        res_tfidf["tfidf_disperso"]["silhouette"],
        res_norm["sin_normalizar"]["silhouette"],
        res_norm["con_normalizacion_l2"]["silhouette"],
    ]

    x = np.arange(len(modelos))
    ancho = 0.35

    ax2 = axes[1]
    rects1 = ax2.bar(x - ancho/2, pures, ancho, label="Pureza / Accuracy (%)", color="#2ca02c", alpha=0.85)
    ax2.set_ylabel("Pureza (%)", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(modelos, fontsize=9.5, fontweight="bold")
    ax2.set_title("B) Comparativa de Representaciones Vectoriales\n¿Cuál es la representación óptima?", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    ax2_twin = ax2.twinx()
    rects2 = ax2_twin.bar(x + ancho/2, sil_vals, ancho, label="Silhouette Score", color="#7570b3", alpha=0.85)
    ax2_twin.set_ylabel("Silhouette Score", fontsize=10, fontweight="bold")

    for r in rects1:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax2_twin.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel C: Repercusión de la Normalización L2 en el Contraste Relativo
    cats = ["Euclídea\nSin Norm", "Euclídea\nNorm L2", "Coseno\nNorm L2"]
    cr_vals = [
        res_norm["sin_normalizar"]["distancia_euclidea"]["contraste_relativo"],
        res_norm["con_normalizacion_l2"]["distancia_euclidea"]["contraste_relativo"],
        res_norm["con_normalizacion_l2"]["distancia_coseno"]["contraste_relativo"],
    ]
    colores_c = ["#e7298a", "#1b9e77", "#386cb0"]

    axes[2].bar(cats, cr_vals, color=colores_c, alpha=0.88, width=0.55, edgecolor="black")
    axes[2].set_ylabel("Contraste Relativo C_R = (d_max - d_min) / d_min", fontsize=10, fontweight="bold")
    axes[2].set_title("C) Contraste Métrico y Fenómeno de Concentración\n(Aggarwal et al., 2001)", fontsize=11, fontweight="bold")
    axes[2].grid(axis="y", linestyle=":", alpha=0.6)

    for i, v in enumerate(cr_vals):
        axes[2].text(i, v + max(cr_vals)*0.02, f"C_R={v:.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    ruta_fig9 = os.path.join(carpeta_salida, "fig9_evidencias_dimension_y_normalizacion.png")
    plt.savefig(ruta_fig9, dpi=200)
    plt.close(fig)
    print(f"\n[OK] Figura 9 generada con éxito en '{ruta_fig9}'.")


def ejecutar_todos_los_experimentos():
    """
    Ejecuta el protocolo experimental completo para alimentar el informe científico-técnico
    y guarda los resultados en 'data_processed/evidencias_experimentales.json'.
    """
    print("\n" + "=" * 70)
    print(" EJECUTANDO BANCO DE PRUEBAS EXPERIMENTAL COMPLETO")
    print(" (Evidencias de Dimensión, Normalización, TF-IDF y Referencia Random)")
    print("=" * 70)

    from src.config import EMBEDDINGS_FILE, LABELS_FILE, POSTS_TEXT_FILE, METRICS_PROFILING_FILE
    if not EMBEDDINGS_FILE.exists() or not LABELS_FILE.exists() or not POSTS_TEXT_FILE.exists():
        print(f"[ERROR] No se encontraron los datos procesados en '{PROCESSED_DATA_DIR}'. Ejecuta 'python main.py' primero.")
        return None

    X = np.load(str(EMBEDDINGS_FILE))
    y_true = np.load(str(LABELS_FILE))
    with open(str(POSTS_TEXT_FILE), "r", encoding="utf-8") as f:
        textos = json.load(f)

    # 1. Dimensión del espacio PCA vs Original
    res_dim = experimento_impacto_dimensionalidad(X, y_true, k_optimo=15)

    # 2. Normalización L2
    res_norm = experimento_impacto_normalizacion(textos, y_true, X_norm=X, k_optimo=15)


    # 3. TF-IDF vs SBERT
    res_tfidf = experimento_representacion_tfidf_vs_sbert(textos, y_true, k_optimo=15)

    # 4. Referencia Random
    res_random = experimento_referencia_random(y_true, k_optimo=15)

    # Guardar evidencias en JSON para el informe
    informe_evidencias = {
        "impacto_dimension_pca": res_dim,
        "impacto_normalizacion": res_norm,
        "comparativa_tfidf_vs_sbert": res_tfidf,
        "referencia_random": res_random,
    }

    ruta_json = os.path.join(str(PROCESSED_DATA_DIR), "evidencias_experimentales.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(informe_evidencias, f, ensure_ascii=False, indent=2)

    # Generar Figura 9
    generar_figura_evidencias_experimentales(res_dim, res_norm, res_tfidf, sbert_stats=None, carpeta_salida=str(REPORTS_DIR))

    print("\n" + "=" * 70)
    print(f" [OK] Evidencias experimentales guardadas en '{ruta_json}'.")
    print("=" * 70)
    return informe_evidencias


if __name__ == "__main__":
    ejecutar_todos_los_experimentos()
