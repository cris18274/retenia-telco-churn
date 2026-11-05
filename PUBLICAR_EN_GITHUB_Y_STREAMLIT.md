# Publicar RetenIA en GitHub y Streamlit Community Cloud

## Entrega que exige la tarea

1. Notebook `.ipynb` reproducible.
2. Repositorio GitHub con `app.py`, `explain_utils.py`, modelos en `artifacts/`, CSV de muestra, `requirements.txt` y documentación.
3. Enlace público a la aplicación Streamlit funcional.
4. Presentación comercial **de máximo 5 diapositivas** (usar `Presentacion_Comercial_RetenIA.pptx`).

> La presentación doctoral de 22 diapositivas es material de estudio, **no** la entrega del business pitch.

## Antes de publicar en Windows

Desde PowerShell, en la carpeta que contiene `app.py`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/check_artifacts.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

La app debe abrir en `http://localhost:8501`. Probar los 4 paneles y, sobre todo, el formulario y las explicaciones. Para rehacer el notebook: instalar adicionalmente `requirements-training.txt` y ejecutarlo con JupyterLab. El notebook recrea los artefactos y puede modificar ligeramente los resultados.

## Paso 1: Crear el repositorio remoto

1. Ir a https://github.com/new e iniciar sesión.
2. Nombre sugerido: `retenia-telco-churn`.
3. Elegir **Public** si se compartirá con el docente; el dataset es el ejemplo académico de Telco Customer Churn, no datos de clientes propios.
4. Crear sin marcar README, .gitignore ni licencia: el proyecto ya los tiene.
5. Copiar la URL HTTPS real del nuevo repositorio.

## Paso 2: Subir archivos con Git en PowerShell

Abrir la carpeta **extraída**, no subir el ZIP como archivo único. Si Git ya está instalado:

```powershell
git init -b main
git add .
git status
git commit -m "Entrega RetenIA: churn, XAI y Streamlit"
git remote add origin https://github.com/TU_USUARIO/retenia-telco-churn.git
git push -u origin main
```

Reemplazar `TU_USUARIO` con el identificador real de GitHub. Si Git solicita autenticación, seguir el inicio de sesión del navegador o usar Git Credential Manager. No subir contraseñas, tokens ni datos confidenciales. Si se prefiere, GitHub ofrece **Add file → Upload files** para subir los archivos desde el navegador (manteniendo la estructura de carpetas).

## Paso 3: Desplegar en Streamlit

1. Abrir https://share.streamlit.io/ e iniciar sesión con GitHub.
2. `Create app` → `Yup, I have an app`.
3. Seleccionar el repositorio creado, la rama `main` y `app.py` como archivo principal.
4. En `Advanced settings` seleccionar **Python 3.12**, si está disponible.
5. Pulsar `Deploy` y revisar los logs si falla una dependencia.
6. Abrir la URL final de `*.streamlit.app`, probar todas las pestañas y copiar el enlace real.
7. Incluir enlaces del **repositorio** y de **Streamlit** en la entrega del EVA.

**Atención:** este paquete no ha sido subido a tu cuenta ni desplegado. El enlace solo existe tras realizar los pasos anteriores. No confundir `localhost:8501` con un enlace público.

## Paso 4: Entrega en EVA

- Adjuntar `RetenIA_Notebook_Completo.ipynb` (o adjuntar ZIP si el aula lo pide).
- Enlaces: `https://github.com/TU_USUARIO/retenia-telco-churn` y la URL real de Streamlit.
- Adjuntar `Presentacion_Comercial_RetenIA.pdf` o el `.pptx` de cinco diapositivas según el formato permitido.
- Verificar que el repositorio permita acceso del docente y que la app abra desde una ventana privada.

## Precauciones científicas

- Regresión logística optimizada con TPE; Random Forest usado como referencia.
- En test: Accuracy 0.7388, Precision 0.5052, Recall 0.7834, F1 0.6143, ROC-AUC 0.8414.
- El modelo no está calibrado para estimación económica: score medio 41.6% frente a abandono observado 26.5%.
- El ROI, el precio de la licencia y la eficacia de campañas son supuestos **hipotéticos**.
- SHAP y LIME explican patrones predictivos, no causas.
- Las imágenes LIME precalculadas originalmente usaron la alternativa metodológica local; al instalar `lime`, vuelve a ejecutar el notebook para generar evidencia con el paquete oficial y comprueba el método mostrado.

## Si Streamlit Cloud falla

1. Revisar `Manage app` → logs.
2. Confirmar que existen `artifacts/model.joblib` y `artifacts/rf_reference.joblib` en GitHub.
3. Confirmar `scikit-learn==1.8.0` y Python 3.12.
4. Confirmar que el CSV y el directorio `artifacts/` mantienen los mismos nombres y mayúsculas (Linux distingue mayúsculas).
5. Comprobar en local con `python scripts/check_artifacts.py`.
6. Si el paquete `lime` o `shap` falla al instalar, compartir el error de logs antes de cambiar versiones aleatoriamente.

Fuentes: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy y https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
