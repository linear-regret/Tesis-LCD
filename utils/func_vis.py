# %%
from tqdm.notebook import tqdm as tqm
import matplotlib.pyplot as plt
import plotly.graph_objs as go

import plotly.express as px
import scipy.stats as st
import seaborn as sns
import pandas as pd
import numpy as np
import os

from pymoo.problems import get_problem
from pymoo.util.plotting import plot


def get_w_espaciados(n_puntos):
    """Se obtienen n_puntos espaciados en 2D que tienen cota inferior y superior porque las que excluían a una de las variables daban lo mismo"""
    prop = 1 / n_puntos
    tuplas = []
    for i in range(n_puntos + 1):
        if i == 0:
            limite_izquierdo, limite_derecho = (0.001, 0.999)
        elif i == n_puntos:
            limite_izquierdo, limite_derecho = (0.999, 0.001)
        else:
            limite_izquierdo, limite_derecho = (
                round(i * prop, 3),
                round(1 - (i * prop), 3),
            )

        tuplas.append((limite_izquierdo, limite_derecho))
    return tuplas


def conjunto_pareto(problema, n_objetivos):
    go_to_Assesment()
    try:
        df = (
            pd.read_csv(
                f"./demo/refsets/{problema}_{str(n_objetivos).zfill(2)}D.pof", sep=" "
            )
            .reset_index()
            .iloc[:, :n_objetivos]
        )
        print("Se encontró el conjunto de referencia en Assessment")
    except:
        try:
            problem = get_problem(f"{problema.lower()}", n_objetivos)
            df = pd.DataFrame(problem.pareto_front())
            print("Se encontró el conjunto de referencia en pymoo")
        except:
            print("No se encontró el frente de pareto ni en Assessment ni en pymoo")
            return None
    df.columns = [f"f{i}" for i in range(len(df.columns))]

    return df


def get_sol(problema, n_objetivos, w0, run, ind_conv="IGDp", nd=True):
    go_to_Assesment()

    cadena = f"../archivos_w_{ind_conv}/w_0{w0}/PCUIEMOA_{problema.upper()}_{str(n_objetivos).zfill(2)}D_R{str(run).zfill(2)}.pof"

    if nd:
        cadena += ".nd"

    df = pd.read_csv(cadena, sep=" ").reset_index().iloc[:, :n_objetivos]
    df.columns = [f"f{i}" for i in range(len(df.columns))]
    return df


def get_sol_path(path, n_objetivos):
    """Obtener df de la solución"""
    go_to_PCUI_Proyect()

    df = pd.read_csv(path, sep=" ").reset_index().iloc[:, :n_objetivos]
    df.columns = [f"f{i}" for i in range(len(df.columns))]
    return df


def get_sol_path_Assesment(path, n_objetivos):
    """Obtener df de la solución partiendo de Assesment"""
    go_to_Assesment()

    df = pd.read_csv(path, sep=" ").reset_index().iloc[:, :n_objetivos]
    df.columns = [f"f{i}" for i in range(len(df.columns))]
    return df


def PCP(df_sol, problema, n_objetivos, df_pareto=None):
    plt.title(f"{problema}, numero de objetivos: {n_objetivos}")
    if df_pareto is not None:
        for i, row in df_pareto.iterrows():
            if i == 1:
                plt.plot(row, c="b", alpha=0.4, linewidth=0.5, label="Referencia PF")
            else:
                plt.plot(row, c="b", alpha=0.4, linewidth=0.5)

    for i, row in df_sol.iterrows():
        if i == 1:
            plt.plot(row, c="r", alpha=0.4, linewidth=0.5, label="Aproximación PF")
        else:
            plt.plot(row, c="r", alpha=0.4, linewidth=0.5)

    plt.legend()


def plot_pareto_sol(df_pareto, df_sol, problema):
    n_objetivos = len(df_pareto.columns)

    if n_objetivos == 2:
        trace_1 = go.Scatter(
            x=df_pareto["f0"],
            y=df_pareto["f1"],
            mode="markers",
            marker=dict(size=5, color="blue", opacity=0.7),
            name="Referencia PF",
        )
        trace_2 = go.Scatter(
            x=df_sol["f0"],
            y=df_sol["f1"],
            mode="markers",
            marker=dict(size=5, color="red", opacity=0.7),
            name="solución",
        )
        fig = go.Figure(data=[trace_1, trace_2])
        fig.update_layout(
            title=f"Aproximación PF \n{problema} n_obj {n_objetivos}",
            height=500,
            width=500,
        )

        fig.show()

    elif n_objetivos == 3:
        trace_set1 = go.Scatter3d(
            x=df_pareto["f0"],
            y=df_pareto["f1"],
            z=df_pareto["f2"],
            mode="markers",
            marker=dict(
                size=5,
                color="blue",
                opacity=0.7,
            ),
            name="Referencia PF",
        )

        trace_set2 = go.Scatter3d(
            x=df_sol["f0"],
            y=df_sol["f1"],
            z=df_sol["f2"],
            mode="markers",
            marker=dict(
                size=5,
                color="red",
                opacity=0.7,
            ),
            name="Aproximación PF",
        )

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title="f0"), yaxis=dict(title="f1"), zaxis=dict(title="f2")
            ),
            margin=dict(l=0, r=0, b=0, t=0),
        )
        fig = go.Figure(data=[trace_set1, trace_set2], layout=layout)
        fig.update_layout(
            title=f"Aproximación al frente de Pareto\n{problema} n_obj {n_objetivos} run",
            height=500,
            width=500,
        )

        fig.show()

        PCP(df_pareto, df_sol, problema, n_objetivos)

    else:
        # Hay d(d-1)/2 plots, de dos por renglón
        n_plots = n_objetivos * (n_objetivos - 1) / 2
        ncols = 3
        nrows = int(np.ceil(n_plots / ncols))
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

        k = 0
        for i in range(n_objetivos):
            for j in range(i + 1, n_objetivos):
                row = k // ncols
                col = k % ncols
                axi = ax[row, col]
                axi.set_xlabel(f"$f_{i}$", size=15)
                axi.set_ylabel(f"$f_{j}$", size=15)
                axi.scatter(
                    df_pareto.iloc[:, i],
                    df_pareto.iloc[:, j],
                    label="Referencia de PF",
                    s=4,
                )

                axi.scatter(
                    df_sol.iloc[:, i], df_sol.iloc[:, j], label="Aproximación al PF"
                )
                axi.legend()
                k += 1
        plt.suptitle(f"{problema}, numero de objetivos: {n_objetivos}")
        plt.show()

        PCP(df_pareto, df_sol, problema, n_objetivos)

    return


def plot_prob(
    problema, n_objetivos, nom_prob="Problema", w0=0, run=1, ind_conv="IGDp", nd=False
):
    df_pareto = conjunto_pareto(problema=problema, n_objetivos=n_objetivos)
    df_sol = get_sol(
        problema=problema,
        n_objetivos=n_objetivos,
        w0=w0,
        run=run,
        ind_conv=ind_conv,
        nd=nd,
    )

    plot_pareto_sol(df_pareto, df_sol, problema=nom_prob)


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


# %%
go_to_Assesment()

w_0 = get_w_espaciados(n_puntos=10)


df_todos_PI = pd.read_csv("../tablas_generadas/todos_QI.csv")

df_todos_PI_R2 = df_todos_PI.query("hiperparam_ind_conv=='R2'")

df_todos_PI_IGDp = df_todos_PI.query("hiperparam_ind_conv=='IGD+'")

df_WC_todos = pd.read_csv("../tablas_generadas/WC_QI_todos.csv").set_index(
    ["n_objetivos", "problema", "indicador", "hiperparam_ind_conv"]
)

df_WC_IGDp = df_WC_todos.loc[:, :, :, "IGD+"].copy()
df_WC_R2 = df_WC_todos.loc[:, :, :, "R2"].copy()


# %%
def get_boxplot(
    problema, n_objetivos, indicador, df_PI=df_todos_PI_IGDp, save_img_path=""
):
    fig = px.box(
        df_PI[
            (df_PI["n_objetivos"] == n_objetivos)
            & (df_PI["problema"] == problema)
            & (df_PI["indicador"] == indicador.lower())
        ],
        x="w_0",
        y="valor_indicador",
        hover_data=["run"],
    )
    fig.update_layout(
        title=f"{problema}\nNúmero de objetivos {n_objetivos}\nindicador {indicador}",
        xaxis_title="w_0",
        yaxis_title=f"{indicador}",
        height=500,
        width=900,
    )
    fig.update_traces(boxmean=True)
    if len(save_img_path) > 0:
        fig.write_image(save_img_path)
    else:
        fig.show()


def meanan(x):
    """Media que no toma en cuenta valores nulos"""
    if not np.isnan(x).all():
        return np.mean(x)
    return 0


def get_tabla(indicador, df_PI, aggfunc="mean"):
    df_hv = df_PI[(df_PI["indicador"] == indicador.lower())]
    df = df_hv.pivot_table(
        index=["problema", "n_objetivos"],
        columns="w_0",
        values="valor_indicador",
        aggfunc=aggfunc,
    )
    return df


def format_scientific_6_digits(x):
    """Para formatear tabla a 6 dígitos significativos"""
    return "{:.6e}".format(x)


def highlight_max_min(s):
    """para colorear el máximo (rojo) y el mínimo (azul) de cada fila"""
    is_max = s == np.nanmax(s)
    is_min = s == np.nanmin(s)
    max_color = "background-color: red"
    min_color = "background-color: blue"
    return [max_color if v else min_color if w else "" for v, w in zip(is_max, is_min)]


def get_heatmap(df_WC, df_todos_PI, problema, n_objetivos, indicador):
    """Regresa el heatmap de wilcoxon, el boxplot para ver que efectivamente uno le está ganando al otro y un dataframe con la media para ver si coinciden los datos"""
    plt.figure(figsize=(6, 4))
    plt.title(
        f"$p$-value Wilcoxon. {problema} {n_objetivos} objetivos  {indicador.upper()}\n Color $\\rightarrow$ renglón mejor que columna"
    )
    mask = df_WC.loc[(n_objetivos, problema, indicador.lower())].astype(float) < 0.05
    sns.heatmap(
        df_WC.loc[(n_objetivos, problema, indicador.lower())].astype(float),
        vmin=0,
        vmax=0.05,
        mask=~mask,
    )
    plt.xlabel("$w_0$")
    plt.ylabel("$w_0$")
    plt.xticks(ticks=np.arange(11) + 0.5, labels=[round(w0i[0], 2) for w0i in w_0])
    plt.grid()
    plt.yticks(ticks=np.arange(11) + 0.5, labels=[round(w0i[0], 2) for w0i in w_0])
    plt.show()
    get_boxplot(problema, n_objetivos, indicador, df_PI=df_todos_PI)
    return (
        df_todos_PI[
            (df_todos_PI["problema"] == problema)
            & (df_todos_PI["n_objetivos"] == n_objetivos)
            & (df_todos_PI["indicador"] == indicador.lower())
        ]
        .groupby(["w_0"])
        .valor_indicador.mean()
        .reset_index()
    )


def conteo_borda_problema(df_borda, n_objetivos, problema, indicador):
    plt.title(f"Conteo de borda {problema} {n_objetivos} objetivos {indicador.upper()}")
    plt.ylabel("Victorias")
    plt.xlabel("$w_0$")
    plt.bar(
        x=[str(round(wi[0], 3)) for wi in w_0],
        height=df_borda.loc[n_objetivos, problema, indicador.lower()],
        color="purple",
    )
    plt.show()

    return
