Base de datos compuesta por grabaciones en formato .wav de dígitos en inglés, esta se divide en los siguientes conjuntos:

- test: 484 muestras (~10%)
- validation: 990 muestras (~20%)
- train: 3476 muestras (~70%)

Cada conjunto posee las clases balanceadas y se compone de hablantes distintos (cada hablante tiene dos grabaciones por cada dígito).

El archivo labels.csv contiene el detalle de cada grabación, indicando la dirección donde se almacena, un identificador del hablante (irrelevante para esta tarea), la partición a la que pertenece y su clase.

Las grabaciones se separan en 11 clases, donde la relación entre el índice de cada clase y el contenido del audio se muestra a continuación:

- 0 : 'zero'
- 1 : 'one
- 2 : 'two'
- 3 : 'three'
- 4 : 'four'
- 5 : 'five'
- 6 : 'six'
- 7 : 'seven'
- 8 : 'eight'
- 9 : 'nine'
- 10 : 'oh' (forma alternativa de decir cero en inglés)

De forma adicional, se incluye el archivo read_example.ipynb con un corto código ejemplificando la forma de acceder a los audios de la base de datos.