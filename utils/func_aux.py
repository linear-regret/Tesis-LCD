from pymoo.visualization.scatter import Scatter
from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import shutil
import time
import os
import re


def go_to_PCUI_Proyect():
    """Para regresar al folder del proyecto PCUI_Proyect"""
    while os.path.basename(os.getcwd()) != "Tesis-LCD":
        os.chdir("..")
    os.chdir("./PFI-EMOA/PCUI_Project/")


def go_to_Assesment():
    """Para regresar al folder del proyecto Assessment"""
    while os.path.basename(os.getcwd()) != "Tesis-LCD":
        os.chdir("..")
    os.chdir("./Assessment")


def get_w_espaciados(n_puntos):
    """Se obtienen n_puntos espaciados en 2D que tienen cota inferior y superior porque las que excluían a una de las variables daban lo mismo"""
    prop = 1 / n_puntos
    tuplas = []
    for i in range(n_puntos + 1):
        if i == 0:
            tuplas.append((0.001, 0.999))
        elif i == n_puntos:
            tuplas.append((0.999, 0.001))
        else:
            tuplas.append((i * prop, 1 - (i * prop)))
    return tuplas


def create_param_file(w_comb, label, feval=50000, n_obj=3, n_var=26, wfg_npos=4):
    """Crea archivo de configuración para ejecutar el algoritmo especificando
    w_comb: combinación de pesos para la escalarización
    label: nombre del archivo
    feval: número de evaluaciones de la función de aptitud
    n_obj: variables objetivo
    n_var: variables de decisión
    wfg_npos: número de position related parameters

    Regresa el nombre del arhivo para encontrarlo con los siguientes programas
    """

    go_to_PCUI_Proyect()

    # Se lee y modifica un template para construir el archivo de configuración indicado
    src = r"./demo/input/Param_03D_template.cfg"
    dst = rf"./demo/input/Param_{str(n_obj).zfill(2)}D-{label}.cfg"

    # Se elimina el archivo si ya existe
    if os.path.exists(dst):
        os.remove(dst)

    # Se lee el template y se construye el contenido del nuevo archivo
    with open(src, "r") as f:
        s = f.read()
        s = re.sub(
            r"feval = \d+", f"feval = {feval}", s
        )  # cambiar el número de evaluaciones
        s = re.sub(
            r"nobj = \d+", f"nobj = {n_obj}", s
        )  # cambiar el número de objetivos
        s = re.sub(
            r"^nvar = \d+", f"nvar = {n_var}", s, flags=re.MULTILINE
        )  # cambiar el número de variables
        s = re.sub(
            r"^wfg_npos = \d+", f"wfg_npos = {wfg_npos}", s, flags=re.MULTILINE
        )  # cambiar el numero de pos related parameters para los wfg
        s = re.sub(
            r"^wfile_r2 = input/weight/weight_03D_14.sld",
            f"wfile_r2 = input/weight/weight_{str(n_obj).zfill(2)}D_120.senergy",
            s,
            flags=re.MULTILINE,
        )

    # Crear y escribir el nuevo archivo
    s += f"\n\nw_comb = {w_comb[0]}, {w_comb[1]}\np_Deltap = 2"

    with open(dst, "w") as f:
        f.write(s)

    return dst[dst.find("Param") :]


def read_output(path):
    """Lee un archivo .pof .pos y lo convierte a un dataframe para poder visualizarlo mejor"""
    df = pd.read_csv(path, skiprows=1, header=None, sep=" ").iloc[:, :-1]
    return df.rename_axis(index="punto").rename(
        columns={i: f"f{i}" for i in range(len(df.columns))}
    )


def ejecutar_algoritmo(
    algoritmo, param_input, problema_prueba, num_ejecuciones, func_escalarizacion
):
    """Se ejecuta el algoritmo de acuerdo a los parámetros indicados"""
    go_to_PCUI_Proyect()
    os.chdir("./demo")

    if algoritmo == "pcuiemoa":
        s = f"./{algoritmo} input/{param_input} {problema_prueba} {num_ejecuciones} {func_escalarizacion} IGD+ ES"
        params = re.findall(r"_(.+)\-", param_input)[0]
        path_out_lista = [
            f"./demo/output/PCUI-EMOA_ATCH_IGD+_ES_{problema_prueba}_{params}_R{str(i+1).zfill(2)}.pof"
            for i in range(num_ejecuciones)
        ]
    else:
        s = f"./emo_moea {algoritmo} input/{param_input} {problema_prueba} {num_ejecuciones}"

    print(f"Ejecutando comando\n{s}")
    print("-" * 100)
    os.system(s)
    print("-" * 100)
    print("Comando terminado")

    go_to_PCUI_Proyect()

    return path_out_lista


def ejecutar_algoritmo_R2(
    algoritmo, param_input, problema_prueba, num_ejecuciones, func_escalarizacion
):
    """Se ejecuta el algoritmo de acuerdo a los parámetros indicados"""
    go_to_PCUI_Proyect()
    os.chdir("./demo")

    if algoritmo == "pcuiemoa":
        s = f"./{algoritmo} input/{param_input} {problema_prueba} {num_ejecuciones} {func_escalarizacion} IGD+ ES"
        params = re.findall(r"_(.+)\-", param_input)[0]
        path_out_lista = [
            f"./demo/output/PCUI-EMOA_ATCH_IGD+_ES_{problema_prueba}_{params}_R{str(i+1).zfill(2)}.pof"
            for i in range(num_ejecuciones)
        ]
    else:
        s = f"./emo_moea {algoritmo} input/{param_input} {problema_prueba} {num_ejecuciones}"

    print(f"Ejecutando comando\n{s}")
    print("-" * 100)
    os.system(s)
    print("-" * 100)
    print("Comando terminado")

    go_to_PCUI_Proyect()

    return path_out_lista


def ejecutar_algoritmo_con_input(
    w_comb,
    label,
    n_obj,
    n_var,
    problema_prueba,
    ind_conv="IGD+",
    ind_diversity="ES",
    algoritmo="pcuiemoa",
    feval=50_000,
    num_ejecuciones=10,
    func_escalarizacion="augmented_chebyshev_pcui",
    wfg_npos=4,
):
    """Se ejecuta el algoritmo de acuerdo a los parámetros indicados, se guarda en output y se le cambia el nombre para que tenga el w_0 usado"""
    go_to_PCUI_Proyect()

    param_input = create_param_file(
        w_comb=w_comb,
        label=label,
        n_obj=n_obj,
        n_var=n_var,
        feval=feval,
        wfg_npos=wfg_npos,
    )

    os.chdir("./demo")

    if algoritmo == "pcuiemoa":
        s = f"./{algoritmo} input/{param_input} {problema_prueba} {num_ejecuciones} {func_escalarizacion} {ind_conv} {ind_diversity}"
        params = re.findall(r"_(.+)\-", param_input)[0]
        path_out_lista_pof = [
            f"./demo/output/PCUI-EMOA_ATCH_{ind_conv}_{ind_diversity}_{problema_prueba}_{params}_R{str(i+1).zfill(2)}.pof"
            for i in range(num_ejecuciones)
        ]
        path_out_lista_pos = [
            f"./demo/output/PCUI-EMOA_ATCH_{ind_conv}_{ind_diversity}_{problema_prueba}_{params}_R{str(i+1).zfill(2)}.pos"
            for i in range(num_ejecuciones)
        ]
    else:
        s = f"./emo_moea {algoritmo} input/{param_input} {problema_prueba} {num_ejecuciones}"

    print(f"Ejecutando comando\n{s}")
    print("-" * 100)
    os.system(s)
    print("-" * 100)
    print("Comando terminado")

    # Renombrar archivo
    go_to_PCUI_Proyect()

    for i in range(len(path_out_lista_pof)):
        file_pof = path_out_lista_pof[i]
        file_pos = path_out_lista_pos[i]
        idx_d = file_pof.find("D_")

        try:
            os.rename(
                file_pof,
                f"{file_pof[:idx_d+2]}{str(label).zfill(2)}W_{file_pof[idx_d+2:]}",
            )
        except:
            print("Error de copia en ", file_pof)
        try:
            os.rename(
                file_pos,
                f"{file_pos[:idx_d+2]}{str(label).zfill(2)}W_{file_pos[idx_d+2:]}",
            )
        except:
            print("Error de copia en ", file_pos)

    return path_out_lista_pof


def obtener_sols_no_dominadas(path_exec):
    """
    Se le pasa como argumento la ruta del ejecución que es la salida de la función ejecutar_algoritmo
    Regresa la ruta del archivo de las soluciones no dominadas.
    """
    os.chdir("./demo/")
    cadena = path_exec[path_exec.find("output") :]
    s = rf"./emo_ndset {cadena}"
    print(f"Ejecutando comando:\n{s}")
    print("-" * 100)
    os.system(s)
    ruta_no_dominadas = cadena + ".nd"

    if os.path.exists(f"{ruta_no_dominadas}"):
        print("Comando terminado")
        go_to_PCUI_Proyect()
        return f"./demo/{cadena}.nd"

    go_to_PCUI_Proyect()
    return "Error"


def calcular_HV(path_nd, n_exec=1, ref_punto=[4, 3, 2]):
    go_to_PCUI_Proyect()
    os.chdir("./demo")
    path_demo = path_nd[path_nd.find("output") :]
    comando = f'./emo_indicator HV {path_demo} {n_exec} {" ".join([str(s) for s in ref_punto])}'
    print(f"Ejecutando\n{comando}", "-" * 100, sep="\n")
    # output = run(comando, capture_output=True).stdout
    os.system(comando)
    go_to_PCUI_Proyect()
    return comando


def ejecucion_evaluacion(
    w,
    label,
    feval,
    problema_prueba,
    num_ejecuciones,
    func_escalarizacion,
    ref_punto,
    n_obj,
    n_var,
    algoritmo="pcuiemoa",
):
    """
    Función que crea el archivo de configuración para pesos w, corre el algoritmo, obtiene las soluciones no dominadas, y calcula el HV
    """
    go_to_PCUI_Proyect()
    param_input = create_param_file(w, label, feval, n_obj, n_var)
    path_sd = ejecutar_algoritmo(
        algoritmo, param_input, problema_prueba, num_ejecuciones, func_escalarizacion
    )
    path_nd = obtener_sols_no_dominadas(path_sd)
    calcular_HV(path_nd, ref_punto=ref_punto)
    go_to_PCUI_Proyect()
    return read_output(path_sd), read_output(path_nd)


def ejecucion_evaluacion_pymoo(
    w,
    label,
    feval,
    problema_prueba,
    num_ejecuciones,
    func_escalarizacion,
    ref_punto,
    n_obj,
    n_var,
    calcular_HV=True,
    algoritmo="pcuiemoa",
    wfg_npos=4,
):
    """
    Función que crea el archivo de configuración para pesos w, corre el algoritmo
    Regresa las soluciones dominadas, las no dominadas y el valor del hipervolumen para las no dominadas
    """
    go_to_PCUI_Proyect()
    param_input = create_param_file(w, label, feval, n_obj, n_var, wfg_npos=wfg_npos)
    path_sd_lista = ejecutar_algoritmo(
        algoritmo, param_input, problema_prueba, num_ejecuciones, func_escalarizacion
    )
    path_nd_lista = [obtener_sols_no_dominadas(path_sd) for path_sd in path_sd_lista]
    if ref_punto is not None:
        ind = HV(ref_point=ref_punto)

    df_sd_lista = [read_output(path_sd) for path_sd in path_sd_lista]
    df_nd_lista = [read_output(path_nd) for path_nd in path_nd_lista]

    go_to_PCUI_Proyect()
    if calcular_HV:
        return df_sd_lista, df_nd_lista, [ind(df_ndi.values) for df_ndi in df_nd_lista]
    else:
        return df_sd_lista, df_nd_lista, 0


def plot_vs_PF(problema, n_var, n_obj, punto_ref, valores):
    """Función auxiliar para graficar"""

    F = get_problem(problema, n_var=n_var, n_obj=n_obj).pareto_front()

    plot = Scatter()
    plot.add(F, label="Pareto")
    plot.add(punto_ref, s=70, color="k", marker="*", label="punto_ref")
    plot.add(valores, label=f"$w_0$", alpha=0.5)
    plot.show()
    plt.legend()
    return plot


def write_results(
    prob,
    df_nd_lista,
    df_sd_lista,
    HV_lista=None,
    ws=[0, 1],
    num_ejecuciones=1,
    n_obj=3,
    folder="../datos_algoritmos",
):
    N_segmentacion = len(ws)

    go_to_PCUI_Proyect()

    df_nd_lista_join = []
    df_sd_lista_join = []
    df_HV_lista_join = []

    for i in range(N_segmentacion):
        if HV_lista is not None:
            df_HV_i = pd.DataFrame({"HV": HV_lista[i], "run": range(num_ejecuciones)})
            df_HV_i["w_0"] = ws[i][0]
            df_HV_lista_join.append(df_HV_i)

        for n in range(num_ejecuciones):
            df_nd_i = df_nd_lista[i][n]
            df_sd_i = df_sd_lista[i][n]
            df_nd_i["w_0"] = ws[i][0]
            df_sd_i["w_0"] = ws[i][0]
            df_nd_i["run"] = n
            df_sd_i["run"] = n

            df_nd_lista_join.append(df_nd_i)
            df_sd_lista_join.append(df_sd_i)

    df_sd_export = pd.concat(df_sd_lista_join)
    df_nd_export = pd.concat(df_nd_lista_join)

    if HV_lista is not None:
        df_HV_export = pd.concat(df_HV_lista_join)

    if not os.path.exists(folder):
        os.mkdir(folder)
        os.mkdir(folder + "/nd")
        os.mkdir(folder + "/sd")

        if HV_lista is not None:
            os.mkdir(folder + "/HV")

    df_nd_export.to_csv(f"{folder}/nd/nd_{prob}_{n_obj}.csv", index=False)
    df_sd_export.to_csv(f"{folder}/sd/sd_{prob}_{n_obj}.csv", index=False)

    if HV_lista is not None:
        df_HV_export.to_csv(f"{folder}/HV/HV_{prob}_{n_obj}.csv", index=False)
    return


def get_df_CD(problema, n_objetivos, indicador, hiperparam_ind_conv, df_PI):
    '''Para obtener el df en la notación del programa para los CDs'''
    df_reducido = df_PI.query(
        f'(problema=="{problema}") & (n_objetivos=={n_objetivos}) & (indicador=="{indicador}") & (hiperparam_ind_conv=="{hiperparam_ind_conv}")'
    )

    return (
        pd.DataFrame(
            {
                "classifier_name": df_reducido["w_0"]
                .astype(float)
                .round(3)
                .astype(str)
                .astype(str),
                "dataset_name": df_reducido["run"].astype(str),
                "accuracy": df_reducido["valor_indicador"],
            }
        )
        .reset_index()
        .iloc[:, 1:]
    )
