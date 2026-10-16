# AMIA 2025 · TP1 · LDA/QDA y optimización matemática de modelos

Trabajo práctico 1 de Análisis Matemático para Inteligencia Artificial (CEIA, FIUBA).

Parte del código base de la cátedra (clasificador bayesiano, `QDA`, `TensorizedQDA` y las tres variantes Cholesky) y lo optimiza en dos frentes: tensorización y factorización de Cholesky.

## Contenido

| Archivo | Qué tiene |
|---|---|
| `AMIA_2025_TP1.ipynb` | Notebook con el código y las respuestas a las consignas |
| `requirements.txt` | Dependencias para correrlo en local |

## Variantes implementadas

| Variante | Qué cambia |
|---|---|
| `FasterQDA` | Saca el loop de observaciones; pasa por una matriz `(k, n, n)` y se queda con la diagonal |
| `EfficientQDA` | Calcula solo la diagonal con `sum(A ⊙ Bᵀ)`, sin formar la matriz `n × n` |
| `TensorizedChol` | Apila los `L⁻¹` de `QDA_Chol3` y vectoriza sobre clases |
| `EfficientChol` | Vectoriza también sobre observaciones: forma cuadrática como `‖L⁻¹(x − μ)‖²` |

Al final del notebook está la comparación de las nueve variantes (tiempo de fit y predict, memoria y accuracy) con el mismo protocolo de benchmark.

## Cómo correrlo

En Colab: subir el notebook y ejecutar todas las celdas.

En local:

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook AMIA_2025_TP1.ipynb
```

El dataset Letters se descarga de OpenML la primera vez, así que hace falta conexión. Los tiempos dependen de la máquina y de las versiones de NumPy y SciPy; el notebook imprime las versiones usadas.
