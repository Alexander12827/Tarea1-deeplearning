# Tarea 1 – Reconocimiento de Dígitos con MLP

Repositorio para la Tarea 1 de **EL7060-1 Deep Learning para Procesamiento de Señales**, Universidad de Chile.  
El objetivo es implementar un **reconocedor de dígitos aislados en inglés** usando un **Multilayer Perceptron (MLP)** en PyTorch.

---

## 📂 Estructura del repositorio

Tarea1-DeepLearning/
│── data/ # scripts o instrucciones para descargar TiDigits
│── src/
│ │── preprocessing.py # extracción de características (Mel Filter Bank)
│ │── model.py # definición del MLP
│ │── train.py # entrenamiento + validación
│ │── evaluate.py # evaluación (accuracy + matriz de confusión)
│── notebooks/ # experimentos en Jupyter/Colab
│── results/ # métricas, gráficos y matriz de confusión
│── report/ # informe (máx. 6 páginas)
│── slides/ # presentaciones avance/final
│── README.md # este archivo
│── requirements.txt # dependencias

yaml
Copy code

---

## 🚀 Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/usuario/Tarea1-DeepLearning.git
   cd Tarea1-DeepLearning
Crear un entorno virtual (opcional, recomendado):

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # en Linux/Mac
venv\Scripts\activate     # en Windows
Instalar dependencias:

bash
Copy code
pip install -r requirements.txt
📊 Dataset
Se utiliza la base de datos TiDigits (descargable desde el link entregado en el enunciado).

Guardar los audios en data/.

Las particiones ya están dadas en train, valid y test.

🛠️ Uso
Preprocesamiento (ejemplo con Mel Filter Bank):

bash
Copy code
python src/preprocessing.py
Entrenamiento:

bash
Copy code
python src/train.py --epochs 50 --batch_size 32 --lr 0.001
Evaluación:

bash
Copy code
python src/evaluate.py
Los resultados (accuracy, matriz de confusión) se guardarán en la carpeta results/.

📈 Resultados esperados
Accuracy en test set

Matriz de confusión entre las 11 clases (0–9 + dígito "oh").

📑 Entregables
Código funcionando (este repo).

Informe (máx. 6 páginas).

Presentaciones (avance y final).

📌 Consideraciones
Justificar hiperparámetros: tasa de aprendizaje, dropout, batch size, early stopping, etc.

Comparar resultados para distintos valores.

Discutir ventajas y limitaciones del MLP en esta tarea.

Analizar efecto de usar features dinámicos.

👥 Autores

Alejandro
Marcelo