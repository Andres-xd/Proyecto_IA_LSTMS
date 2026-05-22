import subprocess
import sys
import time
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).resolve().parent
CARPETA_SRC = BASE_DIR / "src"
CARPETA_DATOS = BASE_DIR / "data" / "processed"

# Colores
VERDE = "\033[92m"
ROJO = "\033[91m"
AZUL = "\033[94m"
NEGRITA = "\033[1m"
RESET = "\033[0m"


def separador(titulo):
    print(f"\n{NEGRITA}{'-' * 55}")
    print(f"  {titulo}")
    print(f"{'-' * 55}{RESET}")


def ejecutar_script(nombre_script, descripcion):
    ruta_script = CARPETA_SRC / nombre_script
    print(f"\n{AZUL}>> {descripcion}...{RESET}")

    tiempo_inicio = time.time()
    resultado = subprocess.run(
        [sys.executable, str(ruta_script)],
        cwd=str(BASE_DIR),
        capture_output=False,
    )
    tiempo_total = time.time() - tiempo_inicio

    if resultado.returncode == 0:
        print(f"{VERDE}[LISTO]{RESET} {nombre_script} - {tiempo_total:.1f}s")
        return True

    print(f"{ROJO}[FALLO]{RESET} {nombre_script} - algo salio mal, mira arriba")
    return False


def limpiar_datos_procesados():
    separador("Limpiando data/processed")

    if CARPETA_DATOS.exists():
        archivos = list(CARPETA_DATOS.iterdir())
        for archivo in archivos:
            archivo.unlink()
        if archivos:
            print(f"  Ya quedo limpia. Se borraron {len(archivos)} archivo(s).")
        else:
            print("  Ya estaba vacia, asi que todo bien.")
    else:
        CARPETA_DATOS.mkdir(parents=True)
        print("  La carpeta no estaba, asi que la cree.")


# Pipelines
def pipeline_sismos():
    separador("Preparando datos sismicos")

    pasos = [
        ("limpiar_sismos.py", "Paso 1/5 - dejando limpio el dataset"),
        ("preprocesar_sismos.py", "Paso 2/5 - armando la serie diaria"),
        ("normalizar_sismos.py", "Paso 3/5 - acomodando los valores"),
        ("ventanas_sismos.py", "Paso 4/5 - creando las ventanas"),
        ("split_sismos.py", "Paso 5/5 - separando train, val y test"),
    ]

    for script, descripcion in pasos:
        exito = ejecutar_script(script, descripcion)
        if not exito:
            print(f"\n{ROJO}El pipeline de sismos se paro en: {script}{RESET}")
            print("  Arreglalo y lo corremos otra vez.")
            return False

    print(f"\n{VERDE}Sismos listos. Ya puedes seguir con agents/.{RESET}")
    return True


def pipeline_energia():
    separador("Preparando datos de energia")

    pasos = [
        ("limpiar_energia.py", "Paso 1/4 - dejando limpio el dataset"),
        ("normalizar_energia.py", "Paso 2/4 - acomodando los valores"),
        ("ventanas_energia.py", "Paso 3/4 - creando las ventanas"),
        ("split_energia.py", "Paso 4/4 - separando train, val y test"),
    ]

    for script, descripcion in pasos:
        exito = ejecutar_script(script, descripcion)
        if not exito:
            print(f"\n{ROJO}El pipeline de energia se paro en: {script}{RESET}")
            print("  Arreglalo y lo corremos otra vez.")
            return False

    print(f"\n{VERDE}Energia lista. Ya puedes seguir con agents/.{RESET}")
    return True


# Inicio
if __name__ == "__main__":
    argumentos = [a.lower() for a in sys.argv[1:]]

    print(
        f"""
{NEGRITA}=======================================================
  PIPELINE MAESTRO
  Preparacion de datos
======================================================={RESET}

  Uso:
    python pipeline.py
    python pipeline.py sismos
    python pipeline.py energia

  Nota:
    El entrenamiento va aparte, desde agents/.
"""
    )

    tiempo_inicio = time.time()

    if "sismos" in argumentos:
        limpiar_datos_procesados()
        pipeline_sismos()

    elif "energia" in argumentos:
        limpiar_datos_procesados()
        pipeline_energia()

    else:
        limpiar_datos_procesados()
        resultado_sismos = pipeline_sismos()
        resultado_energia = pipeline_energia()

        separador("Resumen")
        estado_s = f"{VERDE}Todo bien{RESET}" if resultado_sismos else f"{ROJO}Algo fallo{RESET}"
        estado_e = f"{VERDE}Todo bien{RESET}" if resultado_energia else f"{ROJO}Algo fallo{RESET}"
        print(f"  Sismos : {estado_s}")
        print(f"  Energia: {estado_e}")
        print("\n  Lo que sigue es correr el entrenamiento en agents/.")

    tiempo_total = time.time() - tiempo_inicio
    print(f"\n{NEGRITA}{'-' * 55}")
    print(f"  Tiempo total: {tiempo_total:.1f} segundos")
    print(f"{'-' * 55}{RESET}\n")
