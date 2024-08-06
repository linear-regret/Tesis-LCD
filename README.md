Este proyecto tiene la siguiente estructura de archivos:

# PFI-EMOA

- En el Directorio de `PFI-EMOA` se encuentra la carpeta de `PCUI-Project` dentro de la que se tiene todo el código usado para ejecutar el algoritmo de PFI-EMOA.

En el directorio se incluye el Makefile para compilar el proyecto y tres directorios: `demo`, `include` y `src`. Estos directorios cuentan con la lógica necesaria para correr el algoritmo. Dado que este proyecto sólo se enfoca en correr el código ya hecho, nos concentraremos explicando las partes relevantes para el mismo.

Para ejecutar una corrida del algoritmo se tomará como ejemplo el siguiente comando en la terminal que se llama desde el directorio de `PCUI_Project`

```bash
./pcuiemoa input/Param_03D-0.cfg WFG1 10 augmented_chebyshev_pcui IGD+ ES
```

- `./pcuiemoa` es la ruta del ejecutable.
- `input/Param_03D-0.cfg` es la ruta del archivo de configuración
  - En este archivo se encuentra la información relevante de cada corrida y es el que se modifica ejecución por ejecución. Los parámetros a modificar en este archivo incluyen: el número de objetivos, de dimensiones del espacio factible, ruta al archivo de semilla que se usa para la inicialización de la población y otros más dependiendo del indicador involucrado que se detallan en la sección [Archivos de configuración](#archivos-de-configuración).
- `WFG1` es el problema prueba donde se evaluará el algoritmo
  - En este proyecto se corrieron los problemas de WFG1-WFG9 y DTLZ1-DTLZ7.
- `10` el número de ejecuciones a realizar.
  - Estas ejecuciones se realizan con poblaciones inicializadas de acuerdo al archivo de semillas que se especifica en el archivo de configuración.
- `augmented_chebyshev_pcui` es la función de escalarización a usar. En este proyecto se deja fija a la función aumentada de chebyshev.
- `IGD+` es el indicador de convergencia a usar en la escalarización de Chebyshev. En este trabajo se cambia entre el IGD+ (el usado en PFI-EMOA) y el R2 (que necesita el conjunto de pesos para las funciones de utilidad y que se especifica en el archivo de configuración)
- `ES` es el indicador de diversidad a usar en la escalarización de Chebyshev. En este trabajo se deja fijo el indicador de la Energía-S de Riesz.

Al finalizar la ejecución el archivo se escriben dos archivos por cada ejecución en el directorio de `demo/output` que difieren sólo en la extensión del archivo. El nombre depende del archivo de configuración que especificamos al llamar el ejecutable:

- `PCUI-EMOA_ATCH_IGD+_ES_WFG1_03D_00W_R07.pof`: La aproximación al frente de Pareto resultante después de finalizar la ejecución del algoritmo.
- `PCUI-EMOA_ATCH_R2_ES_WFG1_03D_00W_R07.pos`: La aproximación al conjunto de Pareto al finalizar la ejecución del algoritmo; es decir en el espacio factible.

Donde la primer parte del nombre del archivo `PCUI-EMOA_ATCH_R2_ES_WFG1_03D_` especifica los parámetros que se establecieron al ejecutar el algoritmo.

La parte de `00W` se refiere al conjunto de pesos usado para esta ejecución, explicado en la sección de [Archivos de configuración](#archivos-de-configuración).

Mientras que la parte de `R07` se refiere al número de corrida del algoritmo de las ejecucones especificadas, comenzando en 1. Es decir, con la línea de ejecución antes escrita se tendrían archivos (tanto .pof como .pos) desde `R01` hasta `R10`.

## Archivos de configuración

En el directorio `demo/input` se encuentran archivos de configuración que detallan cómo se va a hacer cada ejecución del algoritmo. Sólo se mencionarán las partes relevantes y aquellas que cambian ejecución a ejecución dentro de este archivo.

La escalarización de los indicadores usados para la estimación de densidad de PFI-EMOA está fija en la de Chebyshev. Así, lo que comparamos es el peso que cada una de estas componentes tiene. Se consideraron 11 combinaciones de pesos dados por

$$\vec{w} \in \left[ (0.001,0.999), (0.1,0.9),(0.2,0.8),(0.3,0.7),(0.4,0.6), (0.5,0.5),(0.6,0.4),(0.7,0.3),(0.8,0.2),(0.9,0.1),(0.999,0.001) \right] $$

Tomamos un archivo de configuración como ejemplo: `Param_05D-0.cfg`. Donde el `05D` se refiere al número de objetivos y el `-0` al final significa que se modificó el archivo para tener la combinación de peso de índice 0, es decir $\vec{w}=(0.001,0.999)$

```cfg
seed = ./input/seed.dat
```

Es la ruta de las semillas usadas para la inicialización de la población

```cfg
# Number of decision variables (0 default value)
nvar = 30
# Number of objective functions
nobj = 5
```

son las variables del espacio factible y del espacio de objetivos. La especificación para cada problema y número de objetivos se encuentra en la ruta de `tablas_generadas/problemas_prueba.csv`

```cfg
wfile_r2 = input/weight/weight_05D_120.senergy
utl_r2 =  achievement_scalarizing_function
```

Donde `wfile_r2` es la ruta al archivo de pesos usado cuando se establece R2 como el indicador de convergencia del estimador de densidad. El nombre del archivo `weight_05D_120.senergy` indica la dimensión del espacio de objetivos $5$ y el número de vectores de peso, que se fija a 120 para todos los problemas.

```cfg
w_comb = 0.001, 0.999
```

Finalmente `w_comb` nos indica los pesos que se le daran a cada uno de los indicadores en el estimador de densidad. Siento el primero el correspondiente al indicador de convergencia (IGD+ o R2) y el segundo al de diversidad (S-energy).

# Assessment

Proyecto con código para calcular indicadores de calidad de poblaciones así como para generar archivos .tex con tablas de comparaciones usando Wilcoxon.

Cambiando el script `evaluate.sh` se especifica la ruta de los archivos con las poblaciones a calcular, el número de objetivos y el número de indicadores.

Devuelve archivos que tienen extensión de acuerdo al indicador que se calculó. Por ejemplo, el archivo `PCUIEMOA_DTLZ1_03D.eps+` tiene la lista de valores de epsilon+ que se calcularon para tres objetivos, donde cada renglón corresponde a una semilla diferente de acuerdo a lo que se le pasó como entrada al archivo de `evaluate.sh`. Estos archivos se almacenan dentro de directorios correspondientes a cada combinación de pesos en `archivos_w_IGD+/w00`, por ejemplo.

# tablas_generadas

En este directorio están todas las tablas que se obtuvieron. Se listan las principales y su contenido:

- `QI_carac.csv`
  - Tiene columnas de indicador, categoría y meta. Donde categoría está en Diversidad y Convergencia y Meta significa si es un indicador que se quiere maximizar o minimizar. Esto último es importante al realizar pruebas de Wilcoxon ya que no son simétricas.
- `problemas_prueba.csv`
  - Contiene para cada problema en el número de objetivos considerado, las va
- `todos_QI.csv`
  - Tiene el valor de los diferentes indicadores de calidad para cada indicador de convergencia del estimador de densidad (IGD+, R2), cada problema (WFG1-WG9 y DTLZ1-DTL7), cada número de objetivos (de 2 a 7), cada combinación de pesos (11 combinaciones, desde $w_0=0.001$ hasta $w_0=0.999$), cada cada corrida (0 a 9), cada indicador (de los 7 calculados).
- `Friedman_todos.csv`
  - p-value de la prueba de Friedman para cada problema y número de objetivos. Si es menor a 0.05 se rechaza la hipótesis nula de
- `WC_QI_todos.csv`
  - p-value de las pruebas de wicloxon para determinar cuál combinación de pesos es mejor. La prueba está adaptada para mostrar cuál es superior dependiendo de si el objetivo del indicador es maximizar o minimizarse
- `conteo_borda_todos.csv`: Se obtienen los conteos de victorias por cada indicador sumando todas las veces en las que su p-value fue menor a 0.05 en la prueba de wilcoxon uno a uno.

# Notebooks

En la raíz del proyecto se encuentran 5 notebooks que se explicarán en esta sección

## 1_Generando_aproximaciones_al_frente.ipynb

El objetivo de este trabajo es comparar ejecuciones de PFI-EMOA con distintos hiperparámetros. En este cuaderno se revisa cómo cambiar estos hiperparámetros para proporcionar una entrada al algoritmo que corra con estas especificaciones.

Se define la función `create_param_file` que toma el archivo `./demo/input/Param_03D_template.cfg` y lo modifica de la siguiente forma de acuerdo a los siguientes argumentos:

- w_comb: combinación de pesos para la escalarización.
- label: nombre del archivo
  - Se pone de acuerdo a la posición de la combinación de w_comb dentro de la lista de pesos..
- feval: número de evaluaciones de la función de aptitud, se fija a 50,000
- n_obj: variables objetivo.
  - Esta variable cambia también el archivo que usa el algoritmo para obtener los vectores de pesos en caso de usar R2 como indicador de convergencia en el estimador de densidad.
- n_var: variables de decisión
- wfg_npos: parámetros de posición para los problemas de WFG

En este notebook se calculan los pesos usados para el indicador de R2 generados usando la función `get_reference_directions` de la librería [pymoo](https://pymoo.org/misc/reference_directions.html) con el método de S-energy y se encuentran dentro del directorio `PFI-EMOA/PCUI_Project/demo/weights` como se explicó en la sección de [Archivos de configuración](#archivos-de-configuración)

Después la función `ejecutar_algoritmo_con_input` crea el archivo de configuración, hace la ejecución del algoritmo y renombra el archivo para que su nombre tenga toda la información de los diferentes hiperparámetros. Así, se ejecutan los problemas. Este procedimiento puede tardar mucho tiempo dado que las combinaciones de algoritmos a ejecutar con diferentes entradas puede ser muy grande.

Después se mueven los archivos a diferentes carpetas:

- La ejecución de los archivos de IGD+ se pasaron a la carpeta `archivos_w_IGDp`
- La ejecución de los archivos de R2 se pasó a la carpeta de `archivos_w_R2`

## 2_Calculando_Indicadores_de_Calidad.ipynb

En este cuaderno se calculan los indicadores de calidad usando el código del folder [Assessment](#assessment). Se reescribe el archivo de `evaluate.sh` según los indicadores a calcular y el número de objetivos.

Después, se almacenan todos los indicadores en el archivo `tablas_generadas/todos_QI.csv` que tiene como columnas

- hiperparam_ind_conv: Especifica el indicador de convergencia usado en el estimador de densidad (R2 o IGD+)
- problema: el nombre del problema prueba
- n_objetivos: el número de objetivos
- w_0: el valor de la primera entrada del vector de pesos (la otra entrada está definida por la condición $\sum w_i =1$)
- run: Va de cero a diez y es el índice de la semilla usada para inicializar la población
- indicador: El indicador calculado
- valor_indicador: El valor del indicador calculado

## 3_Visualizaciones.ipynb

Tiene las funciones necesarias para mostrar visualizaciones de scatterplots y PCP para un problema en un número de objetivos. Para cada $w_0$ se escoje la corrida que tuvo la mediana del hipervolumen en esa corrida como representante del desempeño de cada combinación de pesos. Se grafican las soluciones obtenidas tanto como la referencia del problema particular y se compara el desempeño del algoritmo obtenido con R2 y con IGD+ como indicador de convergencia en el estimador de densidad.

Además se calculan boxplots de los indicadores por cada problema para así poder diferenciar cuáles pesos tuvieron mayor éxito maximizando o minimizando algún indicador.

## 4_Pruebas_Estadísticas.ipynb

En este cuaderno se calcula la prueba de Friedman para cada población dentro de un mismo problema y número de objetivos. Se obtienen gráficas descriptivas de cuando se tiene una diferencia significativa entre las diferentes poblaciones (corridas) de indicadores en cada combinación de pesos diferente.

Se calcula la prueba de Wilcoxon uno a uno donde se compara un algoritmo en un problema para un indicador y dos configuraciones de pesos. Se obtiene `./tablas_generadas/WC_QI_todos.csv` que tiene la información de todas las pruebas de Wilcoxon para cada comparación realizada.

Se calculan los conteos de borda, es decir, el número de veces que una combinación de pesos fue significativamente mejor que otra (p-value por abajo de 0.05).

Se generan gráficas de agregaciones que nos permiten ver cómo se comportan los conteos con respecto a diferentes aspectos de los problemas como el número de objetivos y el indicador medido.

## 5_CD_plots_Borda_Tablas.ipynb

Se usa la tabla `todos_QI_maximizados.csv` que tiene los indicadores con signo negaivo si se minimizan y se obtienen los diagramas de diferencia crítica para cada problema y para diferentes agregaciones.

Se obtienen gráficas de conteos de borda para números distintos de objetivos tomando en cuenta ambos indicadores de convergencia del estimador de densidad. Además se comparan directamente los resultados de ambos algoritmos.

Por último se comprueba que el algoritmo da el mismo resultado si se usa la combinación `w_comb = 0,1` o `w_comb = 1,0` por lo que se optó por dar los extremos por `w_comb = 0.001, 0.999` y `w_comb = 0.999, 0.001`

# utils

En este folder se encuentran funciones auxiliares para la lectura, visualización y manipulación de algunas funciones.
