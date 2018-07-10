# chatbot-seq2seq

Este artículo se centrará en cómo construir un modelo de secuencia a secuencia utilizando Python y Tensorflow y el dataset de conversaciones Movie Dialogue Corpus para construir un chatbot simple.

Para este proyecto, construiremos un chatbot usando conversaciones del [Movie Dialogue Corpus] de Cornell University. Las principales características de nuestro modelo son celdas LSTM, un RNN dinámico bidireccional y decodificadores con atención.

Las conversaciones se limpiarán extensamente para ayudar al modelo a producir mejores respuestas. Como parte del proceso de limpieza, se eliminará la puntuación, las palabras raras se reemplazarán por "UNK" (nuestro token "desconocido"), no se usarán oraciones más largas y todas las letras estarán en minúsculas.

Con una mayor cantidad de datos, sería más práctico mantener las características, como la puntuación. Sin embargo, para este proyecto no quiero dejarme llevar por demasiado tiempo.

Una de las cosas realmente buenas de usar un modelo de secuencia a secuencia es la diversidad de sus aplicaciones. Aunque lo usaremos para construir un chatbot, también se puede aplicar a la traducción de idiomas, resumen de texto, generación de texto, etc.

Para más información puedes acceder a este artículo

https://pixelitymarketing.com/@pixelitystudios.ml/crea-tu-primer-chatbot-con-python
