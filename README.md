# FINLAB — Intelligence Lab

Aplicación educativa de análisis cuantitativo.

## Archivos
- `streamlit_app.py`: aplicación.
- `requirements.txt`: dependencias.

## Ejecución
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Datos y privacidad
La aplicación descarga datos históricos públicos mediante yfinance. No importa automáticamente las variables de una sesión temporal de Google Colab; vuelve a descargar los datos cuando se abre. Los CSV se pueden exportar desde la pestaña Datos.

Para una publicación privada, usa un repositorio privado y configura los permisos de visualización de la aplicación. Nunca guardes contraseñas, claves API ni datos personales en el código.

FINLAB es educativo, no ejecuta operaciones ni ofrece recomendaciones financieras.
