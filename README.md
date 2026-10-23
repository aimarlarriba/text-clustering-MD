# text-clustering-MD: Minería de Datos y Clustering No Supervisado de Texto Clínico

> **Proyecto de Minería de Datos (Master / Grado en Inteligencia Artificial / Ciencia de Datos)**  
> Implementación desde cero (*Scratch*) de algoritmos de clustering jerárquico aglomerativo (Lance-Williams 1967), embeddings densos contextuales (Sentence-BERT), validación interna/externa multidimensional, y perfilado semántico de temas clínicos en Reddit (C-SSRS).

---

## 📋 Tabla de Contenidos
1. [Descripción General](#-descripción-general)
2. [Estructura del Proyecto](#-estructura-del-proyecto)
3. [Fundamentos Teóricos y Científicos](#-fundamentos-teóricos-y-científicos)
4. [Instalación y Requisitos](#-instalación-y-requisitos)
5. [Guía de Ejecución](#-guía-de-ejecución)
6. [Catálogo de Gráficas y Figuras Generadas](#-catálogo-de-gráficas-y-figuras-generadas)
7. [Suite de Tests Unitarios](#-suite-de-tests-unitarios)
8. [Referencias Bibliográficas](#-referencias-bibliográficas)

---

## 🔬 Descripción General

El objetivo de este proyecto es analizar, agrupar y caracterizar semánticamente publicaciones clínicas de usuarios de Reddit procedentes del corpus **Zenodo C-SSRS** (Gaur et al., 2019). 

El sistema implementa un flujo completo de aprendizaje no supervisado sin dependencias de alto nivel para los algoritmos centrales:
- **Representación vectorial**: Generación de representaciones semánticas densas contextuales de $384$ dimensiones mediante Sentence-BERT (`all-MiniLM-L6-v2`).
- **Espacios métricos Scratch**: Cálculo vectorial optimizado en NumPy de matrices de proximidad (Distancia Coseno y Distancia Euclídea) y análisis empírico del fenómeno de pérdida de contraste en alta dimensionalidad (Aggarwal et al., 2001).
- **Clustering Jerárquico Aglomerativo (SAHN) Scratch**: Algoritmo aglomerativo completo implementado con la fórmula recursiva de actualización de **Lance-Williams (1967)** con soporte para enlace simple (*Single*), completo (*Complete*) y promedio (*Average / UPGMA*).
- **Validación Interna Objetiva**: Coeficiente de Correlación Cofenética (CPCC - Sokal & Rohlf 1962), perfilado de Silueta (*Silhouette Score*), compromiso Cohesión vs Separabilidad ($SSE$ / $BSS$) y análisis de índice de Dunn.
- **Validación Externa y Correlación Clínica**: Rand Index, Jaccard Index, Fowlkes-Mallows, Pureza (*Clustering Accuracy* por voto mayoritario), Kappa de Cohen, Macro-F1 y matriz de confusión frente a las 5 clases clínicas reales de riesgo suicida C-SSRS.
- **Validación Honesta Holdout e Inferencia**: Partición 80% Train / 20% Test, asignación por vecino más cercano de instancias no vistas y recuperación semántica de las $K$ publicaciones más próximas a un nuevo texto.

---

## 📁 Estructura del Proyecto

El repositorio está estructurado bajo las mejores prácticas de la ingeniería de software y ciencia de datos:

```text
text-clustering-MD/
│
├── data/                                 # Datos crudos (Raw Data)
│   ├── .gitkeep
│   └── 500_Reddit_users_posts_labels.csv # Dataset clínico anotado (C-SSRS)
│
├── data_processed/                       # Artefactos serializados y resultados
│   ├── .gitkeep
│   ├── embeddings_sbert.npy              # Matriz de embeddings densos (500 x 384)
│   ├── etiquetas_reales.npy              # Clases de riesgo reales [0..4]
│   ├── matriz_distancias_cosine.npy      # Matriz NxN de distancias Coseno
│   ├── matriz_distancias_euclidean.npy   # Matriz NxN de distancias Euclídeas
│   ├── textos_posts.json                 # Textos normalizados de cada usuario
│   ├── comparativa_distancias.png        # Gráfica de contraste Coseno vs Euclídeo
│   ├── metricas_espacios_y_profiling.json# Métricas y profiling O(N^2)
│   │
│   └── graficas_informe/                 # 9 Figuras científicas autoexplicativas
│       ├── .gitkeep
│       ├── fig1_cpcc_distorsion_lance_williams.png
│       ├── fig2_curva_silhouette_picos.png
│       ├── fig3_compromiso_sse_bss.png
│       ├── fig4_perfil_silueta_k5_vs_k15.png
│       ├── fig5_dendrograma_jerarquico_corte.png
│       ├── fig6_mapa_pca_centroides.png
│       ├── fig7_tabla_codebook_6x3_ventanas.png
│       ├── fig8_correlacion_externa_k.png
│       └── fig9_evidencias_dimension_y_normalizacion.png
│
├── src/                                  # Código fuente modular
│   ├── __init__.py                       # Inicializador del paquete src
│   ├── config.py                         # Configuración centralizada de rutas y constantes
│   ├── embeddings.py                     # SBERT, distancias scratch y profiling O(N^2)
│   ├── metricas_internas.py              # CPCC, Silueta, SSE/BSS, corte y dendrograma
│   ├── evaluacion.py                     # Validación externa, Holdout, Codebook e inferencia
│   ├── experimentos.py                   # Experimentos de dimensión, normalización y baselines
│   │
│   └── clustering/                       # Algoritmos de agrupamiento aglomerativo
│       ├── __init__.py                   # Exportaciones del submódulo clustering
│       ├── hierarchical.py               # MotorJerarquico (SAHN iterativo)
│       └── lance_williams.py             # Parámetros y actualización de Lance-Williams
│
├── main.py                               # Orquestador principal CLI unificado
├── toy_example.py                        # Demostración paso a paso (toy-example)
├── pipeline.py                           # Pipeline de extracción y preprocesado

├── ejecutar_pipeline.ps1                 # Script de ejecución para Windows PowerShell
├── ejecutar_pipeline.bat                 # Script de ejecución para Windows CMD
├── ejecutar_pipeline.sh                  # Script de ejecución para Linux/macOS Bash
├── requirements.txt                      # Dependencias del entorno de Python
├── .gitignore                            # Exclusión de datos confidenciales y binarios
└── README.md                             # Documentación técnica y académica
```

---

## 🧠 Fundamentos Teóricos y Científicos

1. **Ontología Clínica C-SSRS (Gaur et al., 2019 - WWW '19)**:
   - El corpus clasifica a los usuarios en 5 niveles ordinales de riesgo:
     `0: Indicator`, `1: Ideation`, `2: Behavior`, `3: Attempt`, `4: Supportive`.
   - Permite contrastar de forma empírica si las agrupaciones semánticas no supervisadas reflejan la progresión clínica del riesgo.

2. **Representaciones Contextuales SBERT (Reimers & Gurevych, 2019 - EMNLP)**:
   - Se utiliza la arquitectura siamesa `all-MiniLM-L6-v2` para proyectar el texto en $\mathbb{R}^{384}$.
   - A diferencia de TF-IDF clásico o BERT estándar por pares, SBERT permite codificar textos en tiempo lineal $O(N)$ y mantener una geometría semántica densa de alta calidad.

3. **Contraste Métrico en Alta Dimensión (Aggarwal et al., 2001 - ICDT)**:
   - En espacios de dimensión $D \ge 384$, la distancia euclídea sufre el fenómeno de concentración de distancias (pérdida de contraste relativo):
     $$\lim_{D \to \infty} \frac{d_{max} - d_{min}}{d_{min}} \to 0$$
   - La distancia del coseno demuestra un mayor contraste relativo y mayor inmunidad a la longitud del documento.

4. **Fórmula Recursiva de Lance & Williams (1967)**:
   - Permite recalcular la distancia entre un cluster fusionado $C_i \cup C_j$ y cualquier otro cluster $C_k$ en tiempo $O(1)$:
     $$d(C_i \cup C_j, C_k) = \alpha_i \cdot d(C_i, C_k) + \alpha_j \cdot d(C_j, C_k) + \beta \cdot d(C_i, C_j) + \gamma \cdot |d(C_i, C_k) - d(C_j, C_k)|$$
   - **Single Link**: $\alpha_i = 0.5, \alpha_j = 0.5, \beta = 0, \gamma = -0.5 \implies \min(d_{ik}, d_{jk})$
   - **Complete Link**: $\alpha_i = 0.5, \alpha_j = 0.5, \beta = 0, \gamma = 0.5 \implies \max(d_{ik}, d_{jk})$
   - **Average Link (UPGMA)**: $\alpha_i = \frac{n_i}{n_i + n_j}, \alpha_j = \frac{n_j}{n_i + n_j}, \beta = 0, \gamma = 0$

5. **Coeficiente de Correlación Cofenética (CPCC - Sokal & Rohlf, 1962)**:
   - Mide la correlación lineal de Pearson entre las distancias originales por pares $D_{ij}$ y las distancias cofenéticas $C_{ij}$ impuestas por el dendrograma en el momento de la primera fusión:
     $$\text{CPCC} = \frac{\sum (D_{ij} - \bar{D})(C_{ij} - \bar{C})}{\sqrt{\sum (D_{ij} - \bar{D})^2 \sum (C_{ij} - \bar{C})^2}}$$

---

## 💻 Instalación y Requisitos

### Requisitos Previos
- Python 3.10 o superior (compatible con Python 3.10, 3.11, 3.12 y 3.13).

### Pasos de Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/aimarlarriba/text-clustering-MD.git
   cd text-clustering-MD
   ```

2. Crear y activar un entorno virtual:
   - En **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - En **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Guía de Ejecución

El proyecto ofrece múltiples alternativas de ejecución automatizada y multiplataforma:

### Opción 1: Ejecución Completa con `main.py` (Recomendada)
Ejecuta el flujo completo (embeddings, matrices de proximidad, clustering jerárquico, validación interna/externa, Holdout y generación de las 8 figuras):
```bash
python main.py
```

### Opción 2: Ejecución según Sistema Operativo
- **En Windows (PowerShell)**:
  ```powershell
  .\ejecutar_pipeline.ps1
  ```
- **En Windows (CMD)**:
  ```cmd
  ejecutar_pipeline.bat
  ```
- **En Linux / macOS**:
  ```bash
  chmod +x ejecutar_pipeline.sh
  ./ejecutar_pipeline.sh
  ```

### Opción 3: Ejecución Modular por Fases
- **Solo Fase 1 (Embeddings y Matrices de Distancia Scratch)**:
  ```bash
  python main.py --step embeddings --n_muestras 500
  ```
- **Solo Fase 2 y 3 (Clustering, Validación y Figuras)**:
  ```bash
  python main.py --step evaluacion
  ```
- **Ajustar el número de muestras**:
  ```bash
  python main.py --n_muestras 200
  ```

---

## 📊 Catálogo de Gráficas y Figuras Generadas

Todas las figuras generadas se almacenan automáticamente en alta resolución en `data_processed/graficas_informe/`:

| Figura | Archivo | Descripción Metodológica |
|---|---|---|
| **Fig. 1** | `fig1_cpcc_distorsion_lance_williams.png` | Comparativa de preservación métrica (CPCC) y distorsión jerárquica para Single, Complete y Average Linkage. |
| **Fig. 2** | `fig2_curva_silhouette_picos.png` | Evolución de la silueta media en el rango $k \in [2, \sqrt{N}=22]$ con detección no supervisada de picos locales. |
| **Fig. 3** | `fig3_compromiso_sse_bss.png` | Compromiso clásico entre Cohesión Interna ($SSE$) y Separabilidad Externa ($BSS$). |
| **Fig. 4** | `fig4_perfil_silueta_k5_vs_k15.png` | Perfil detallado de silueta muestra a muestra comparando la partición clínica ($K=5$) vs la partición óptima descubierta ($k̂=15$). |
| **Fig. 5** | `fig5_dendrograma_jerarquico_corte.png` | Dendrograma jerárquico completo y detalle truncado en el punto de corte horizontal exacto. |
| **Fig. 6** | `fig6_mapa_pca_centroides.png` | Proyección bidimensional PCA de los clusters con centroides $m_j$ y etiquetas anticolisión. |
| **Fig. 7** | `fig7_tabla_codebook_6x3_ventanas.png` | **Codebook Semántico**: Tabla formal con 6 ventanas de 3 palabras (trigramas c-TF-IDF) y la oración prototipo de cada centroide. |
| **Fig. 8** | `fig8_correlacion_externa_k.png` | Curvas de validación externa (Rand, Jaccard, Fowlkes-Mallows, Purity, Cohen's Kappa, Macro-F1) en función de $K$. |
| **Fig. 9** | `fig9_evidencias_dimension_y_normalizacion.png` | **Evidencias Experimentales**: Curva de calidad vs dimensión ($D \in [2..384]$), comparativa TF-IDF vs SBERT, y efecto de normalización en el contraste. |
| **Extra** | `comparativa_distancias.png` | Histograma de contraste relativo y correlación entre Distancia Coseno y Euclídea (Aggarwal et al., 2001). |

---

## 🧪 Ejemplo Didáctico de Ejecución (Toy-Example)

Cumpliendo con la especificación del guión de prácticas (*"Ejecución: facilitar al usuario e.g. mediante ejemplo de ejecución e.g. (toy-example)"*), se incluye un script autocontenido que ejecuta paso a paso todas las rutinas *Scratch* sobre datos sintéticos en menos de 1 segundo:
1. Actualización de matriz $3 \times 3$ con Lance-Williams (Single, Complete, Average).
2. Ajuste del `MotorJerarquico` sobre $15$ muestras y visualización de la matriz de enlace.
3. Corte del dendrograma y cálculo de CPCC, Silhouette y Dunn.
4. Validación externa (Rand, Jaccard, Purity, Kappa, F1) frente a etiquetas reales de juguete.
5. Inferencia de cluster y vecinos cercanos sobre una nueva instancia no vista.

Para ejecutar la demostración:
```bash
python toy_example.py
```
o mediante el flag `--toy`:
```bash
python main.py --toy
```


---

## 📚 Referencias Bibliográficas

- **Gaur, M., Alambo, A., Sain, J. P., Ugander, J., & Sheth, A. (2019)**. *"Knowledge-aware assessment of severity of suicide risk for early intervention"*. In The World Wide Web Conference (WWW '19), pp. 514–525.
- **Lance, G. N., & Williams, W. T. (1967)**. *"A general theory of classificatory sorting strategies: 1. Hierarchical systems"*. The Computer Journal, 9(4), pp. 373–380.
- **Reimers, N., & Gurevych, I. (2019)**. *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"*. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP).
- **Sokal, R. R., & Rohlf, F. J. (1962)**. *"The comparison of dendrograms by objective methods"*. Taxon, 11(2), pp. 33–40.
- **Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001)**. *"On the surprising behavior of distance metrics in high dimensional space"*. In International Conference on Database Theory (ICDT), pp. 420–434.
