"""
PIPELINE MAESTRO — Sistema de Prediccion LSTM
==============================================
Este archivo coordina y ejecuta SOLO los pasos de preparacion
de datos, en el orden correcto, con un solo comando.

IMPORTANTE: Este pipeline NO entrena modelos.
    El entrenamiento se hace por separado ejecutando los scripts
    en la carpeta agents/ cuando los datos ya esten listos.

    Por que separado? Porque entrenar consume mucha memoria y
    tiempo. Primero preparas los datos, luego entrenas cuando
    estes listo.

Uso desde la terminal:
    python pipeline.py          -> Prepara datos de sismos Y energia
    python pipeline.py sismos   -> Solo prepara datos sismicos
    python pipeline.py energia  -> Solo prepara datos de energia

Fases del proyecto completo:
    FASE 1 -> Preparacion de datos     (este archivo, pipeline.py)
    FASE 2 -> Entrenamiento de modelos (scripts en agents/)
    FASE 3 -> API de predicciones      (api.py)
    FASE 4 -> Dashboard interactivo    (frontend/src/App.jsx)
"""

import subprocess
import sys
import time
from pathlib import Path

# ── Rutas principales ─────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
CARPETA_SRC   = BASE_DIR / "src"              # scripts de preprocesamiento
CARPETA_DATOS = BASE_DIR / "data" / "processed"

# ── Colores para mensajes en consola ──────────────────────────────────────────
VERDE    = "\033[92m"
ROJO     = "\033[91m"
AZUL     = "\033[94m"
AMARILLO = "\033[93m"
NEGRITA  = "\033[1m"
RESET    = "\033[0m"


def separador(titulo):
    """Imprime una linea divisoria con titulo para separar secciones."""
    print(f"\n{NEGRITA}{'─'*55}")
    print(f"  {titulo}")
    print(f"{'─'*55}{RESET}")


def ejecutar_script(nombre_script, descripcion):
    """
    Ejecuta un script de la carpeta src/ y reporta si salio bien o mal.

    Todos los scripts de preparacion de datos viven en src/.
    Los de entrenamiento estan en agents/ y NO se tocan desde aqui.

    Parametros:
        nombre_script -> nombre del archivo .py a ejecutar
        descripcion   -> mensaje que se muestra en la terminal

    Retorna:
        True si el script termino sin errores, False si fallo.
    """
    ruta_script = CARPETA_SRC / nombre_script
    print(f"\n{AZUL}>> {descripcion}...{RESET}")

    tiempo_inicio = time.time()
    resultado = subprocess.run(
        [sys.executable, str(ruta_script)],
        cwd=str(BASE_DIR),
        capture_output=False
    )
    tiempo_total = time.time() - tiempo_inicio

    if resultado.returncode == 0:
        print(f"{VERDE}[LISTO]{RESET} {nombre_script} — {tiempo_total:.1f}s")
        return True
    else:
        print(f"{ROJO}[FALLO]{RESET} {nombre_script} — algo salio mal, revisa el error arriba")
        return False


def limpiar_datos_procesados():
    """
    Vacia la carpeta data/processed/ antes de empezar.
    Asi nos aseguramos de que no queden archivos viejos mezclados con los nuevos.
    """
    separador("Limpiando data/processed/ — borrando lo viejo")

    if CARPETA_DATOS.exists():
        archivos = list(CARPETA_DATOS.iterdir())
        for archivo in archivos:
            archivo.unlink()
        if archivos:
            print(f"  Se fueron {len(archivos)} archivo(s) al reino de las sombras. Carpeta lista.")
        else:
            print("  La carpeta ya estaba vacia, no habia nada que limpiar.")
    else:
        CARPETA_DATOS.mkdir(parents=True)
        print("  Carpeta data/processed/ creada por si las moscas.")


# ── Definicion de los pipelines ───────────────────────────────────────────────

def pipeline_sismos():
    """
    Prepara el dataset sismico en 5 pasos:
        1. Limpia el dataset crudo (quita errores y columnas inservibles)
        2. Construye la serie temporal diaria (un registro por dia)
        3. Normaliza los valores entre 0 y 1
        4. Genera las ventanas deslizantes para el LSTM
        5. Divide en train / validacion / test
    """
    separador("PREPARANDO DATOS SISMICOS")

    pasos = [
        ("limpiar_sismos.py",     "Paso 1/5 — Limpiando dataset temblorsitos"),
        ("preprocesar_sismos.py", "Paso 2/5 — Construyendo serie diaria"),
        ("normalizar_sismos.py",  "Paso 3/5 — Normalizando valores"),
        ("ventanas_sismos.py",    "Paso 4/5 — Generando ventanas deslizantes"),
        ("split_sismos.py",       "Paso 5/5 — Dividiendo en train / val / test"),
    ]

    for script, descripcion in pasos:
        exito = ejecutar_script(script, descripcion)
        if not exito:
            print(f"\n{ROJO}Pipeline sismico se cayo en: {script}{RESET}")
            print( "  Corrige el error y volvelo a correr.")
            return False

    print(f"\n{VERDE}Datos sismicos listos. Ya puedes entrenar con los scripts de agents/.{RESET}")
    return True


def pipeline_energia():
    """
    Prepara el dataset de energia electrica en 4 pasos:
        1. Limpia el dataset crudo (rellena huecos, unifica fecha/hora)
        2. Normaliza los valores entre 0 y 1
        3. Genera las ventanas deslizantes para el LSTM
        4. Divide en train / validacion / test

    Nota: este pipeline tarda mas porque el dataset tiene ~2 millones
    de registros. Es normal, no te asustes si demora unos minutos.
    """
    separador("PREPARANDO DATOS DE ENERGIA ELECTRICA")

    pasos = [
        ("limpiar_energia.py",    "Paso 1/4 — Limpiando dataset de energia"),
        ("normalizar_energia.py", "Paso 2/4 — Normalizando valores"),
        ("ventanas_energia.py",   "Paso 3/4 — Generando ventanas deslizantes"),
        ("split_energia.py",      "Paso 4/4 — Dividiendo en train / val / test"),
    ]

    for script, descripcion in pasos:
        exito = ejecutar_script(script, descripcion)
        if not exito:
            print(f"\n{ROJO}Pipeline de energia se cayo en: {script}{RESET}")
            print( "  Corrige el error y volvelo a correr.")
            return False

    print(f"\n{VERDE}Datos de energia listos. Ya puedes entrenar con los scripts de agents/.{RESET}")
    return True


# ── MAIN — Punto de entrada ───────────────────────────────────────────────────
if __name__ == "__main__":

    argumentos = [a.lower() for a in sys.argv[1:]]

    print(f"""
{NEGRITA}╔══════════════════════════════════════════════════════╗
║          PIPELINE MAESTRO — Sistema LSTM             ║
║          Preparacion de Datos  (FASE 1 de 4)         ║
╚══════════════════════════════════════════════════════╝{RESET}

  Uso:
    python pipeline.py          -> Prepara sismos y energia
    python pipeline.py sismos   -> Solo sismos
    python pipeline.py energia  -> Solo energia

  Recordatorio: los modelos se entrenan aparte desde agents/
""")

    tiempo_inicio = time.time()

    # ── Ejecutar segun los argumentos que reciba ──────────────────────────────
    if "sismos" in argumentos:
        limpiar_datos_procesados()
        pipeline_sismos()

    elif "energia" in argumentos:
        limpiar_datos_procesados()
        pipeline_energia()

    else:
        # Sin argumentos -> prepara ambos datasets de una vez
        limpiar_datos_procesados()
        resultado_sismos  = pipeline_sismos()
        resultado_energia = pipeline_energia()

        separador("RESUMEN FINAL")
        estado_s = f"{VERDE}Todo bien{RESET}"  if resultado_sismos  else f"{ROJO}Hubo un error{RESET}"
        estado_e = f"{VERDE}Todo bien{RESET}"  if resultado_energia else f"{ROJO}Hubo un error{RESET}"
        print(f"  Datos sismicos  : {estado_s}")
        print(f"  Datos de energia: {estado_e}")
        print(f"\n  Siguiente paso: ir a agents/ y correr los scripts de entrenamiento.")

    tiempo_total = time.time() - tiempo_inicio
    print(f"\n{NEGRITA}{'─'*55}")
    print(f"  Tiempo total: {tiempo_total:.1f} segundos")
    print(f"{'─'*55}{RESET}\n")