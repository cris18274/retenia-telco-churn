# RetenIA - Retención inteligente y explicable

Aplicación académica para el problema **Telco Customer Churn**, que combina comparación de modelos, Grid Search, Random Search, Optuna-TPE, NSGA-II, SHAP y LIME, un panel de Streamlit y una propuesta comercial.

## Estado verificado

- Notebook **ejecutado correctamente** en el entorno de creación (todos sus bloques). Dataset: 7.043 filas, 21 columnas; 11 blancos de `TotalCharges` transformados a `NaN` e imputados **dentro del pipeline**.
- Split de 80% train / 20% test estratificado, seed=42. Test **aislado** de la optimización.
- Regresión logística y Random Forest comparados con 3 folds estratificados.
- Grid Search, Random Search y TPE usan el **mismo modelo base**, 24 evaluaciones por estrategia y los mismos 3 folds; se registran tiempos por ejecución.
- NSGA-II evalúa F1 (max.) y tasa de contactos (min.) en 32 propuestas; frente no dominado y criterio de selección explícitos.
- Test final, ranking top 10%, importancia RF y SHAP completados. **SHAP está en log-odds**, no en puntos porcentuales.
- LIME: el código intenta utilizar **`lime.lime_tabular.LimeTabularExplainer` oficial**. Como el entorno de creación no permitía instalar paquetes nuevos, los resultados ilustrativos incluidos usan **una implementación alternativa propia de la metodología LIME**, implementada y declarada en `explain_utils.py`, con regresión lineal local ponderada. Para cumplir una exigencia estricta del paquete oficial, instale `lime`, vuelva a ejecutar el notebook y compruebe que el modo sea `LIME oficial`.
- El programa Streamlit fue preparado y compilado sintácticamente, pero **no probado en un servidor Streamlit** en este entorno (la dependencia no está disponible). Tampoco se ha publicado una URL de despliegue.

## Instalación local (Python 3.11+ sugerido)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Abra la URL local que indique Streamlit. La aplicación carga `artifacts/model.joblib` y archivos ya entrenados; **no vuelve a entrenar modelos**. Antes de presentarse, verificar todos los paneles con el entorno de la máquina de exposición.

## Reproducir resultados

Abrir `RetenIA_Notebook_Completo.ipynb` en Jupyter y ejecutar de arriba a abajo (reemplaza los artefactos). El código de preparación y entrenamiento está **dentro del notebook** y `explain_utils.py`. Toma el CSV del directorio del proyecto. La ejecución puede variar en tiempos y pequeñas diferencias numéricas según plataforma y dependencias.

## Desplegar en GitHub + Streamlit Community Cloud

1. Crear un repositorio propio y subir `app.py`, `explain_utils.py`, `requirements.txt`, el CSV y el directorio `artifacts/` (JSON, CSV, NPY y JOBLIB), sin agregar información privada real.
2. En [share.streamlit.io](https://share.streamlit.io/) crear una app nueva conectada al repositorio, branch `main`, archivo principal `app.py`.
3. Verificar dependencias (en particular `lime`, `shap`, `scikit-learn`, `optuna` en notebook), predicción, visualizaciones y enlace público. **El despliegue y la URL requieren acceso a la cuenta del estudiante**: no están hechos aquí.
4. Reemplazar `ENLACE_PENDIENTE` por el enlace real en la presentación y entregar el repositorio. No publicar datos de clientes reales ni secretos en código.

## Qué se entrega

| Archivo | Función |
|---|---|
| `RetenIA_Notebook_Completo.ipynb` | Notebook original ampliado y ejecutado: modelos, optimización, Pareto, SHAP/LIME, negocio |
| `app.py` | Aplicación con cuatro pestañas: EDA, desempeño, predicción+XAI y negocio |
| `explain_utils.py` | LIME oficial si está instalado; alternativa pedagógica explícita |
| `artifacts/` | Modelos completos, métricas, frente, figuras, ranking demostrativo |
| `Presentacion_Comercial_RetenIA.pptx`/`.pdf` | Cinco diapositivas para el business pitch |

## Resultados técnicos (único test, 1.409 muestras)

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|--:|--:|--:|--:|--:|
| 0,7388 | 0,5052 | 0,7834 | 0,6143 | 0,8414 |

**Calibración:** el score medio en test es **41,6%** y el abandono observado **26,5%** (Brier=**0,1688**). Por tanto las probabilidades están insuficientemente calibradas para pronósticos económicos; la simulación del panel es solo una sensibilidad hipotética y no debe usarse como presupuesto real. Para un despliegue se debe calibrar usando exclusivamente train/CV, volver a comprobar en test y validar el impacto con un grupo de control.

**Advertencia:** la matriz de confusión refleja 287 falsos positivos y 81 falsos negativos (a umbral 0,5). El precio de USD 249 y los supuestos de eficacia, costo y margen son **hipotéticos** y deben validarse con un piloto. La relación entre variables, SHAP, LIME y abandono es predictiva, **no causal**. Las tasas estimadas dependen de la población del conjunto Telco y pueden no generalizar a otros operadores.

## Pregunta comercial central

> ¿Qué aporta RetenIA? Ordena los clientes según probabilidad estimada de abandono y ayuda a justificar a quién contactar primero, antes de invertir en ofertas indiscriminadas.


## Versión final

Este paquete incluye el contraste corregido en las tarjetas de métricas, tema claro, dependencias de la app separadas de las de entrenamiento y una prueba de artefactos.

- Para despliegue: `pip install -r requirements.txt`.
- Para repetir notebook: `pip install -r requirements-training.txt`.
- Para validar modelos exportados: `python scripts/check_artifacts.py`.
- La versión de `scikit-learn` está fijada a 1.8.0 para compatibilidad con los modelos guardados.
- Los enlaces públicos deben completarse con las URLs reales luego de publicar.
