"""
text-clustering-MD: Punto de Entrada Principal (Main Pipeline Orchestrator)
=============================================================================
Orquestador integral del proyecto de Minería de Datos / Text Clustering:
1. Fase de Representación y Espacios Métricos (SBERT + Distancias Coseno / Euclídea).
2. Fase de Clustering Jerárquico Aglomerativo Scratch (Lance-Williams 1967).
3. Fase de Validación Interna (CPCC, Silhouette, SSE, BSS, Dunn, Selección k̂).
4. Fase de Validación Externa (Rand, Jaccard, Fowlkes-Mallows, Purity, Kappa, F1).
5. Fase Semántica, Holdout e Inferencia Prototípica (Codebook y Figuras 1-8).

Uso:
    python main.py                         # Ejecuta el pipeline completo de inicio a fin
    python main.py --step embeddings       # Ejecuta solo la generación de embeddings y distancias
    python main.py --step evaluacion       # Ejecuta clustering, validación y figuras
    python main.py --test                  # Ejecuta tests y toy-examples de verificación
    python main.py --n_muestras 500        # Configura el número de muestras (por defecto 500)
"""

import os
import sys
import time
import warnings
import argparse
from pathlib import Path

# Silenciar advertencias informativas no críticas de Hugging Face Hub
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
warnings.filterwarnings("ignore", message=".*You are sending unauthenticated requests.*")
try:
    from huggingface_hub.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# Añadir la raíz del proyecto al sys.path para importaciones estándar
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_N_SAMPLES,
    ensure_directories,
)


def ejecutar_paso_embeddings(csv_path=None, n_muestras=DEFAULT_N_SAMPLES, modelo=DEFAULT_MODEL_NAME):
    """Fase 1: Carga de dataset, SBERT embeddings, matrices scratch y profiling."""
    print("\n" + "=" * 70)
    print(" [FASE 1] REPRESENTACIÓN VECTORIAL Y ESPACIOS MÉTRICOS")
    print("=" * 70)

    from pipeline import cargar_datos
    from src.embeddings import (
        generar_embeddings,
        guardar_embeddings,
        calcular_matriz_distancias_scratch,
        guardar_matriz_distancias,
        comparar_espacios_metricos,
        perfilar_coste_computacional,
    )
    import json
    import numpy as np

    ruta_csv = str(csv_path or RAW_DATA_FILE)
    carpeta_salida = str(PROCESSED_DATA_DIR)
    ensure_directories()

    # 1. Cargar textos
    textos, etiquetas_reales = cargar_datos(ruta_csv, n_muestras)
    N = len(textos)
    if N == 0:
        print(f"[ERROR] No hay textos disponibles en '{ruta_csv}'. Abortando Fase 1.")
        return False

    # 2. Generar embeddings
    print(f"\n[1/5] Generando embeddings densos contextuales ({modelo})...")
    t0 = time.time()
    X = generar_embeddings(textos, nombre_modelo=modelo, normalizar=True)
    print(f"      Embeddings generados en {time.time() - t0:.2f}s | Forma: {X.shape}")

    # 3. Calcular matrices de distancia Scratch
    print("\n[2/5] Calculando matrices de distancias en NumPy (Scratch)...")
    t0 = time.time()
    D_cos = calcular_matriz_distancias_scratch(X, metrica="cosine")
    t_cos = time.time() - t0
    t0 = time.time()
    D_euc = calcular_matriz_distancias_scratch(X, metrica="euclidean")
    t_euc = time.time() - t0
    print(f"      Distancia Coseno:   {t_cos:.4f}s")
    print(f"      Distancia Euclídea: {t_euc:.4f}s")

    # 4. Comparativa de contraste métrico (Aggarwal et al., 2001)
    print("\n[3/5] Evaluando contraste métrico y preservación topológica...")
    ruta_grafico = os.path.join(carpeta_salida, "comparativa_distancias.png")
    stats = comparar_espacios_metricos(X, D_cos, D_euc, ruta_grafico=ruta_grafico)
    print(f"      Contraste Coseno:   {stats['cos_contraste_relativo']:.2f}")
    print(f"      Contraste Euclídeo: {stats['euc_contraste_relativo']:.2f}")
    print(f"      Correlación Pearson (Coseno vs Euclídeo): {stats['corr_pearson']:.4f}")

    # 5. Profiling O(N^2)
    print("\n[4/5] Profiling empírico de complejidad temporal y de memoria...")
    profiling = perfilar_coste_computacional(X, lista_n=(100, 200, 300, 400, 500))

    # Guardar métricas y artefactos
    print("\n[5/5] Serializando artefactos intermedios en data_processed/...")
    informe_cuantitativo = {
        "estadisticas_espacios_metricos": stats,
        "profiling_complejidad_computacional": profiling,
    }
    with open(os.path.join(carpeta_salida, "metricas_espacios_y_profiling.json"), "w", encoding="utf-8") as f:
        json.dump(informe_cuantitativo, f, ensure_ascii=False, indent=2)

    guardar_embeddings(X, os.path.join(carpeta_salida, "embeddings_sbert.npy"))
    guardar_matriz_distancias(D_cos, os.path.join(carpeta_salida, "matriz_distancias_cosine.npy"))
    guardar_matriz_distancias(D_euc, os.path.join(carpeta_salida, "matriz_distancias_euclidean.npy"))
    np.save(os.path.join(carpeta_salida, "etiquetas_reales.npy"), etiquetas_reales)
    with open(os.path.join(carpeta_salida, "textos_posts.json"), "w", encoding="utf-8") as f:
        json.dump(textos, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Fase 1 completada con éxito. Artefactos listos en '{carpeta_salida}'.")
    return True


def ejecutar_paso_evaluacion():
    """Fase 2 & 3: Clustering, validación interna/externa, Holdout, figuras 1-8 e inferencia."""
    print("\n" + "=" * 70)
    print(" [FASE 2 Y 3] CLUSTERING JERÁRQUICO, VALIDACIÓN CIENTÍFICA E INFERENCIA")
    print("=" * 70)

    from src.evaluacion import ejecutar_evaluacion
    ensure_directories()
    resultados = ejecutar_evaluacion(
        carpeta_datos=str(PROCESSED_DATA_DIR),
        carpeta_salida=str(REPORTS_DIR),
    )
    if resultados:
        print("\n" + "=" * 70)
        print(" [RESUMEN FINAL DE MÉTRICAS CIENTÍFICAS]")
        print("=" * 70)
        print(f"  - Coeficiente Cofenético (CPCC - Sokal & Rohlf): {resultados['cpcc']:.4f}")
        print(f"  - Número Óptimo de Clusters Descubierto (k̂):     {resultados['k_optimo']}")
        print(f"  - Silhouette Score en k̂:                         {resultados['silhouette_k_opt']:.4f}")
        print(f"  - Pureza / Clustering Accuracy:                  {resultados['accuracy_pureza'] * 100:.2f}%")
        print(f"  - Estadístico Kappa de Cohen:                    {resultados['kappa_cohen']:.4f}")
        print(f"  - Macro F1-Score:                                {resultados['f1_macro']:.4f}")
        print(f"  - Rand Index:                                    {resultados['rand_index']:.4f}")
        print(f"  - Jaccard Index:                                 {resultados['jaccard']:.4f}")
        print(f"  - Fowlkes-Mallows Index:                         {resultados['fowlkes_mallows']:.4f}")
        print(f"  - Holdout Train Pureza:                          {resultados['holdout']['train']['accuracy_pureza'] * 100:.2f}%")
        print(f"  - Holdout Test Pureza (no vistos):               {resultados['holdout']['test']['accuracy_pureza'] * 100:.2f}% (std 5-rep: {resultados['holdout']['test']['accuracy_std_5rep'] * 100:.2f}%)")
        print(f"  - Holdout Test Weighted F1 (w.Avg):              {resultados['holdout']['test']['f1_weighted']:.4f}")
        print(f"  - Holdout Train Rand:                            {resultados['holdout']['train']['rand']:.4f}")
        print(f"  - Holdout Test Rand (no vistos):                 {resultados['holdout']['test']['rand']:.4f}")
        print(f"\n[OK] 10 Figuras científicas disponibles en '{REPORTS_DIR}' (incluyendo el Heat-Map Fig 10).")
    return resultados is not None


def ejecutar_paso_experimentos():
    """Fase 4: Ejecuta los experimentos científicos (Dimensión, Normalización, TF-IDF, Random y K-Means)."""
    print("\n" + "=" * 70)
    print(" [FASE 4] EVIDENCIAS EXPERIMENTALES (DIMENSIÓN, PREPROCESO Y BASELINES)")
    print("=" * 70)
    from src.experimentos import ejecutar_todos_los_experimentos
    evidencias = ejecutar_todos_los_experimentos()
    return evidencias is not None


def ejecutar_toy():
    """Ejecuta la demostración didáctica paso a paso sobre datos de juguete (toy-example)."""
    from toy_example import ejecutar_toy_example
    ejecutar_toy_example()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline integral de Text Clustering (text-clustering-MD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "embeddings", "evaluacion", "experimentos", "toy"],
        default="all",
        help="Fase del pipeline a ejecutar ('all', 'embeddings', 'evaluacion', 'experimentos', 'toy')",
    )
    parser.add_argument(
        "--n_muestras",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Número de posts de Reddit a procesar (por defecto {DEFAULT_N_SAMPLES})",
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Modelo de Sentence-Transformers a utilizar (por defecto '{DEFAULT_MODEL_NAME}')",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(RAW_DATA_FILE),
        help="Ruta al archivo CSV con el corpus clínico de Reddit",
    )
    parser.add_argument(
        "--toy",
        "--toy-example",
        "--test",
        action="store_true",
        dest="toy",
        help="Ejecuta la demostración didáctica de algoritmos scratch sobre datos sintéticos (toy-example)",
    )
    parser.add_argument(
        "--infer",
        "--texto",
        type=str,
        default=None,
        dest="infer",
        help="Asigna un nuevo texto clínico a su cluster correspondiente y muestra los vecinos más cercanos dentro de él",
    )

    args = parser.parse_args()
    t_global = time.time()

    if args.infer:
        print("\n" + "=" * 70)
        print(" [INFERENCIA INTERACTIVA DE NUEVA INSTANCIA]")
        print("=" * 70)
        import json
        import numpy as np
        from src.evaluacion import inferir_y_mostrar_cercanos
        from src.clustering.hierarchical import MotorJerarquico
        from src.metricas_internas import cortar_dendrograma_scratch

        emb_path = PROCESSED_DATA_DIR / "embeddings_sbert.npy"
        dist_path = PROCESSED_DATA_DIR / "matriz_distancias_cosine.npy"
        text_path = PROCESSED_DATA_DIR / "textos_posts.json"

        if not emb_path.exists() or not dist_path.exists() or not text_path.exists():
            print("[ERROR] Artefactos procesados no encontrados. Ejecuta 'python main.py' primero.")
            sys.exit(1)

        X = np.load(str(emb_path))
        D = np.load(str(dist_path))
        with open(str(text_path), "r", encoding="utf-8") as f:
            textos = json.load(f)

        Z = MotorJerarquico(tipo_enlace="complete").fit(D)
        etiquetas = cortar_dendrograma_scratch(Z, n_clusters=15)

        print(f"Texto a inferir: \"{args.infer}\"")
        inferir_y_mostrar_cercanos(args.infer, X, etiquetas, textos, k=3, metrica="cosine")
        sys.exit(0)

    if args.toy or args.step == "toy":
        exito = ejecutar_toy()
        sys.exit(0 if exito else 1)

    # Verificar si existen los artefactos de Fase 1 para no repetir innecesariamente
    fase1_necesaria = args.step in ["all", "embeddings"]
    embeddings_listos = (
        (PROCESSED_DATA_DIR / "embeddings_sbert.npy").exists()
        and (PROCESSED_DATA_DIR / "matriz_distancias_cosine.npy").exists()
    )

    if fase1_necesaria:
        # Si se pide explícitamente embeddings o si no existen los archivos, ejecutar fase 1
        if args.step == "embeddings" or not embeddings_listos:
            exito1 = ejecutar_paso_embeddings(csv_path=args.csv, n_muestras=args.n_muestras, modelo=args.modelo)
            if not exito1:
                sys.exit(1)
        else:
            print(f"\n[INFO] Artefactos de embeddings ya existentes en '{PROCESSED_DATA_DIR}'.")
            print("       (Para regenerarlos, ejecuta con '--step embeddings')")

    if args.step in ["all", "evaluacion"]:
        exito2 = ejecutar_paso_evaluacion()
        if not exito2:
            sys.exit(1)

    if args.step in ["all", "experimentos"]:
        exito3 = ejecutar_paso_experimentos()
        if not exito3:
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f" PIPELINE COMPLETADO EXITOSAMENTE en {time.time() - t_global:.2f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

