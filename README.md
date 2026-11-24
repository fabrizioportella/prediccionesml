# Nexo — Predicción KNN

Aplicación web en español con dos modelos de clasificación K-nearest neighbors (KNN):

- **Satisfacción del cliente:** estima si un cliente está satisfecho con el servicio.
- **Decisión de inversión:** clasifica un caso de forma educativa según su perfil e indicadores.

Incluye datos sintéticos para probarla sin preparar archivos. También permite cargar datos propios
en formato CSV. La aplicación procesa variables numéricas y categóricas antes de entrenar el modelo.

## Requisitos

- Python 3.9 o posterior.
- Acceso a Internet durante la instalación de las dependencias.

## Ejecutar en macOS

1. Instala Python 3 desde [python.org](https://www.python.org/downloads/) si aún no lo tienes.
   Abre Terminal y comprueba la instalación:

   ```bash
   python3 --version
   ```

2. Abre la carpeta del proyecto en VS Code y selecciona **Terminal → New Terminal**. Así la
   terminal quedará ubicada en la carpeta del proyecto.

3. Crea y activa un entorno virtual:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Instala las dependencias y ejecuta la aplicación:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

## Ejecutar en Windows

1. Instala Python 3 desde [python.org](https://www.python.org/downloads/). Durante la instalación,
   selecciona **Add Python to PATH**. Abre PowerShell y comprueba la instalación:

   ```powershell
   py --version
   ```

2. Abre la carpeta del proyecto en VS Code y selecciona **Terminal → New Terminal**.

3. Crea y activa un entorno virtual. En PowerShell:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   Si usas el Símbolo del sistema (CMD), activa el entorno con:

   ```bat
   .venv\Scripts\activate.bat
   ```

4. Instala las dependencias y ejecuta la aplicación:

   ```powershell
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

Al iniciar, Streamlit mostrará una dirección local, normalmente
`http://localhost:8501`. Ábrela en un navegador. Para detener la aplicación, vuelve a la terminal y
presiona **Ctrl+C**.

## Cómo probar los modelos

1. Elige **Satisfacción del cliente** o **Decisión de inversión**.
2. Deja los datos sintéticos de demostración para una prueba rápida.
3. Completa los campos del caso y presiona **Ver predicción**.
4. Si quieres, abre **¿Qué tan bien funciona el modelo?** para consultar las métricas de evaluación.

El número de vecinos y la carga de un CSV propio son opciones adicionales; no son necesarios para
probar la aplicación por primera vez.

## Usar un archivo CSV propio

Abre **¿Quieres probar con tu propio archivo CSV?** y carga un archivo UTF-8 separado por comas, con
encabezados en la primera fila y registros en las filas siguientes. Elige la columna que se desea
predecir; debe tener exactamente dos resultados posibles, por ejemplo `Sí` y `No`. Las demás
columnas se utilizan como variables. Las filas sin resultado se descartan y los valores vacíos de
las variables se imputan durante el entrenamiento.

## Nota sobre los datos

Los ejemplos incorporados son **sintéticos**; no representan clientes, mercados ni decisiones reales.
El clasificador de inversión es solo educativo y no constituye asesoría financiera. Las métricas se
calculan sobre una partición de prueba y no garantizan el rendimiento con datos futuros.
