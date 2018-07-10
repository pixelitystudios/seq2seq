
# coding: utf-8

# ## Importación de librerías
# 
# Primero necesitaremos importar las librerías que utilizarémos para la creación del chatbot las cuales son:
# * Pandas
# * Numpy
# * Tensorflow
# * Re
# * Time

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time


# ### Verificaremos nuestra versión de tensorflow

# In[2]:


tf.__version__


# # Visualización y modelado de nuesto dataset

# #### Carga del dataset
# 
# Deberémos cargar nuestro dataset de los archivos proporcionados por Movie Dialogue Corpus:
# * movie_lines.txt
# * movie_conversations.txt

# In[3]:


lines = open('./movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('./movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')


# Visualizaremos las primeras 5 líneas que utilizaremos para entrenar nuestro modelo

# In[4]:


lines[:5]


# Visualizamos los primeros 5 IDS de las sentencias que serán nuestro input y datos objetivo.

# In[5]:


conv_lines[:5]


# Creamos un diccionario para mapear la identificación de cada línea con su texto

# In[6]:


id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]


# Creamos una lista de todos los id de las "líneas" de conversaciones.

# In[7]:


convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))


# Así deberá quedar nuestra lista convs:

# In[8]:


convs[:5]


# Clasificamos las oraciones en preguntas (entradas) y respuestas (objetivos)

# In[9]:


questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])


# Comprobamos si hemos cargado los datos correctamente

# In[10]:


limit = 0
for i  in range(limit, limit+5):
    print(questions[i])
    print(answers[i])
    print()


# Comparamos longitudes de preguntas y respuestas

# In[11]:


print(len(questions))
print(len(answers))


# Generamos nuestra función para limpiar nuestros datos eliminando caracteres innecesarios y alterando el formato de las palabras.

# In[12]:


def clean_text(text):

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text


# Limipiamos nuestros datos.

# In[13]:


clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))


# In[14]:


clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))


# Verificamos algunos de los datos para asegurarnos de que se hayan limpiado bien.

# In[15]:


limit = 0
for i in range(limit, limit+5):
    print(clean_questions[i])
    print(clean_answers[i])
    print()


# Ahora encontraremos la duración de las oraciones

# In[16]:


lengths = []
for question in clean_questions:
    lengths.append(len(question.split()))
for answer in clean_answers:
    lengths.append(len(answer.split()))


# Creamos un marco de datos para que los valores puedan ser inspeccionados

# In[17]:


lengths = pd.DataFrame(lengths, columns=['counts'])


# In[18]:


lengths.describe()


# In[19]:


print(np.percentile(lengths, 80))
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))
print(np.percentile(lengths, 95))
print(np.percentile(lengths, 99))


# Eliminamos las preguntas y respuestas que tengan menos de 2 palabras y más de 20 palabras.

# In[20]:


min_line_length = 2
max_line_length = 20

short_questions_temp = []
short_answers_temp = []

i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1


# In[21]:


short_questions = []
short_answers = []

i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1


# Comparamos el número de líneas que usaremos con el número total de líneas.

# In[22]:


print("# de preguntas:", len(short_questions))
print("# de respuestas:", len(short_answers))
print("% del dataset usado: {}%".format(round(len(short_questions)/len(questions),4)*100))


# Creamos un diccionario para la frecuencia del vocabulario

# In[23]:


vocab = {}
for question in short_questions:
    for word in question.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
            
for answer in short_answers:
    for word in answer.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1


# * Eliminamos palabras raras del vocabulario.
# * Vamos a tratar de sustituir a menos del 5% de las palabras con < UNK >
# * Veremos esta proporción pronto.

# In[24]:


threshold = 10
count = 0
for k,v in vocab.items():
    if v >= threshold:
        count += 1


# In[25]:


print("Tamaño total de nuestro vocabulario:", len(vocab))
print("Tamao del vocabulario que usaremos:", count)


# En caso de que deseemos usar un vocabulario diferente para el texto de origen y el de destino, podemos establecer diferentes valores de umbral. No obstante, crearemos diccionarios para proporcionar un número entero único para cada palabra.

# In[26]:


questions_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        questions_vocab_to_int[word] = word_num
        word_num += 1
        
answers_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        answers_vocab_to_int[word] = word_num
        word_num += 1


# Agregamos los tokens únicos a los diccionarios de vocabulario.

# In[27]:


codes = ['<PAD>','<EOS>','<UNK>','<GO>']

for code in codes:
    questions_vocab_to_int[code] = len(questions_vocab_to_int)+1
    
for code in codes:
    answers_vocab_to_int[code] = len(answers_vocab_to_int)+1


# Creamos diccionarios para asignar los enteros únicos a sus respectivas palabras. es decir, un diccionario inverso para vocab_to_int.

# In[28]:


questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}
answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}


# Verificamos la longitud de los diccionarios.

# In[29]:


print(len(questions_vocab_to_int))
print(len(questions_int_to_vocab))
print(len(answers_vocab_to_int))
print(len(answers_int_to_vocab))


# Agregamos el token de final de frase al final de cada respuesta.

# In[30]:


for i in range(len(short_answers)):
    short_answers[i] += ' <EOS>'


# Convertimos el texto a enteros. Volvemos a colocar las palabras que no están en el vocabulario correspondiente a < UNK >

# In[31]:


questions_int = []
for question in short_questions:
    ints = []
    for word in question.split():
        if word not in questions_vocab_to_int:
            ints.append(questions_vocab_to_int['<UNK>'])
        else:
            ints.append(questions_vocab_to_int[word])
    questions_int.append(ints)
    
answers_int = []
for answer in short_answers:
    ints = []
    for word in answer.split():
        if word not in answers_vocab_to_int:
            ints.append(answers_vocab_to_int['<UNK>'])
        else:
            ints.append(answers_vocab_to_int[word])
    answers_int.append(ints)


# Verificamos las longitudes

# In[32]:


print(len(questions_int))
print(len(answers_int))


# Calculamos el porcentaje de todas las palabras se han sustituido por < UNK >

# In[33]:


word_count = 0
unk_count = 0

for question in questions_int:
    for word in question:
        if word == questions_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1
    
for answer in answers_int:
    for word in answer:
        if word == answers_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1
    
unk_ratio = round(unk_count/word_count,4)*100
    
print("Número total de palabras:", word_count)
print("Número de veces que se uso <UNK>:", unk_count)
print("Porcentaje de palabras que son <UNK>: {}%".format(round(unk_ratio,3)))


# Ordeamos las preguntas y respuestas por la duración de las preguntas. Esto reducirá la cantidad de relleno durante el entrenamiento, lo que debería acelerar el entrenamiento y ayudar a reducir la pérdida

# In[34]:


sorted_questions = []
sorted_answers = []

for length in range(1, max_line_length+1):
    for i in enumerate(questions_int):
        if len(i[1]) == length:
            sorted_questions.append(questions_int[i[0]])
            sorted_answers.append(answers_int[i[0]])

print(len(sorted_questions))
print(len(sorted_answers))
print()
for i in range(3):
    print(sorted_questions[i])
    print(sorted_answers[i])
    print()


# # Entrenamiento

# ### Crear marcadores de posición para las entradas al modelo

# En este paso crearemos marcadores de prosición para las entradas de nuestro modelo. Nuestro learning_rate y keep_prob no deberán tener parámetro de forma. Esto se debe a que la forma predeterminada es None que es lo que necesitamos, así que podemos dejarla en blanco para mantener nuestor código conciso.

# In[35]:


def model_inputs():
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return input_data, targets, lr, keep_prob


# ### Elimine la última identificación de palabra de cada lote y concatenamos < GO > al comienzo de cada lote
# tf.strided_slice() eliminará la palabra final de cada lote. Anexado al inicio de cada lote estará el token < GO > . Este formato es necesario para crear las incrustaciones para nuestra capa de decodificación.

# In[36]:


def process_encoding_input(target_data, vocab_to_int, batch_size):
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


# ### Creamos la capa de codificación
# Esto codificará nuestros datos de entrada.
# 
# * Según lo que he leído, las celdas LSTM generalmente superan a las celdas GRU en tareas seq2seq, como esta.
# * Hacer que el codificador sea bidireccional demostró ser mucho más efectivo que una simple red de retroalimentación.
# * Devolvemos solo el estado del codificador porque es la entrada para nuestra capa de decodificación. En pocas palabras, los pesos de las celdas de codificación son lo que nos interesa.

# In[37]:


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    fw_cell_array = []
    for _ in range(num_layers):
        fw_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        #fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob = keep_prob)
        fw_cell_array.append(fw_cell)
        
    enc_cell = tf.contrib.rnn.MultiRNNCell(fw_cell_array)
    _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = enc_cell,
                                                   cell_bw = enc_cell,
                                                   sequence_length = sequence_length,
                                                   inputs = rnn_inputs, 
                                                   dtype=tf.float32)
    return enc_state


# ### Decodificamos los datos de entrenamiento
# El uso de la atención en nuestras capas de decodificación reduce la pérdida de nuestro modelo en aproximadamente un 20% y aumenta el tiempo de entrenamiento en un 20%. Yo diría que es una compensación justa. 
# 
# Algunas notas para hacer:
# 
# * El modelo funciona mejor cuando los estados de atención se establecen con ceros.
# * Las dos opciones de atención son bahdanau y luong. Bahdanau es menos costoso desde el punto de vista computacional y se lograron mejores resultados con él.

# In[38]:


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob, batch_size):

    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])
    
    att_keys, att_vals, att_score_fn, att_construct_fn =             tf.contrib.seq2seq.prepare_attention(attention_states,
                                                 attention_option="bahdanau",
                                                 num_units=dec_cell.output_size)
    
    train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                     att_keys,
                                                                     att_vals,
                                                                     att_score_fn,
                                                                     att_construct_fn,
                                                                     name = "attn_dec_train")
    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, 
                                                              train_decoder_fn, 
                                                              dec_embed_input, 
                                                              sequence_length, 
                                                              scope=decoding_scope)
    train_pred_drop = tf.nn.dropout(train_pred, keep_prob)
    return output_fn(train_pred_drop)


# ### Decodificamos los datos de predicción
# decoding_layer_infer() es muy similar a decoding_layer_train(). La principal diferencia son los parámetros extra. Estos parámetros adicionales son necesarios para ayudar al modelo a crear respuestas precisas para sus oraciones de entrada.
# 
# Tampoco hay abandonos en esta función. Esto se debe a que lo estamos usando para crear nuestras respuestas durante las pruebas (también conocido como hacer predicciones), y queremos utilizar nuestra red completa para eso.

# In[39]:


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])
    
    att_keys, att_vals, att_score_fn, att_construct_fn =             tf.contrib.seq2seq.prepare_attention(attention_states,
                                                 attention_option="bahdanau",
                                                 num_units=dec_cell.output_size)
    
    infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn, 
                                                                         encoder_state[0], 
                                                                         att_keys, 
                                                                         att_vals, 
                                                                         att_score_fn, 
                                                                         att_construct_fn, 
                                                                         dec_embeddings,
                                                                         start_of_sequence_id, 
                                                                         end_of_sequence_id, 
                                                                         maximum_length, 
                                                                         vocab_size, 
                                                                         name = "attn_dec_inf")
    infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, 
                                                                infer_decoder_fn, 
                                                                scope=decoding_scope)
    
    return infer_logits


# ### Creamos la celda de descodificación e ingresamos los parámetros para las capas de descodificación de entrenamiento e inferencia
# Aquí estamos usando las dos funciones anteriores, una celda de decodificación y una capa completamente conectada para crear nuestros logits de entrenamiento y de inferencia. Estamos utilizando tf.variable_scope () para reutilizar las variables del entrenamiento para hacer predicciones.
# 
# Les recomiendo encarecidamente que inicialicen sus weights y biases. Inicializando sus weights con una distribución normal truncada y una pequeña desviación estándar, esto realmente puede ayudar a mejorar el rendimiento de su modelo.

# In[40]:


def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, vocab_to_int, keep_prob, batch_size): 
    with tf.variable_scope("decoding") as decoding_scope:
        fw_cell_array = []
        for _ in range(num_layers):
            fw_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            #fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, input_keep_prob = keep_prob)
            fw_cell_array.append(fw_cell)
        dec_cell = tf.contrib.rnn.MultiRNNCell(fw_cell_array)
        
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, 
                                                                vocab_size, 
                                                                None, 
                                                                scope=decoding_scope,
                                                                weights_initializer = weights,
                                                                biases_initializer = biases)

        train_logits = decoding_layer_train(encoder_state, 
                                            dec_cell, 
                                            dec_embed_input, 
                                            sequence_length, 
                                            decoding_scope, 
                                            output_fn, 
                                            keep_prob, 
                                            batch_size)
        decoding_scope.reuse_variables()
        infer_logits = decoding_layer_infer(encoder_state, 
                                            dec_cell, 
                                            dec_embeddings, 
                                            vocab_to_int['<GO>'],
                                            vocab_to_int['<EOS>'], 
                                            sequence_length - 1, 
                                            vocab_size,
                                            decoding_scope, 
                                            output_fn, keep_prob, 
                                            batch_size)

    return train_logits, infer_logits


# ### Usamos las funciones anteriores para crear el entrenamiento y los logits de inferencia
# Aquí es donde unimos todo y generamos los resultados para nuestro modelo.
# 
# * De manera similar a la inicialización de weights y biases, creo que lo mejor es inicializar mis incrustaciones también. En lugar de utilizar una distribución normal truncada, una distribución uniforme al azar es más apropiada. Si lo desea, puede leer más sobre las incrustaciones del tutorial de TensorFlow .
# * Como no tenemos que procesar las entradas de nuestra codificación, podemos tf.contrib.layers.embed_sequence() para simplificar el código un poco.
# * Si quieren acortar un poco su código, pueden devolver decoding_layer() en lugar de crear train_logits & infer_logits y devolverlos. Lo escribí de esta manera para ser más explícito.

# In[41]:


def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, answers_vocab_size, 
                  questions_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers, 
                  questions_vocab_to_int):

    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, 
                                                       answers_vocab_size+1, 
                                                       enc_embedding_size,
                                                       initializer = tf.random_uniform_initializer(0,1))
    enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length)

    dec_input = process_encoding_input(target_data, questions_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([questions_vocab_size+1, dec_embedding_size], 0, 1))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    train_logits, infer_logits = decoding_layer(dec_embed_input, 
                                                dec_embeddings, 
                                                enc_state, 
                                                questions_vocab_size, 
                                                sequence_length, 
                                                rnn_size, 
                                                num_layers, 
                                                questions_vocab_to_int, 
                                                keep_prob, 
                                                batch_size)
    return train_logits, infer_logits


# ### Establecemos los hiperparámetros

# In[59]:


#epochs = 100
epochs = 1 # Se declara una epoca por propósito de prueba
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75


# Restablecemos el gráfico para asegurarnos de que esté listo para el entrenamiento

# In[60]:


tf.reset_default_graph()


# Inicializamos la sesión

# In[61]:


sess = tf.InteractiveSession()


# Esto configura la estructura de nuestro gráfico.
# 
# * Elegí usar una sesión interactiva para proporcionar un poco más de flexibilidad al construir este modelo, pero puede usar cualquier tipo de sesión que desee.
# * La longitud de la secuencia será la longitud de línea máxima para cada lote. Clasifiqué mis entradas por longitud para reducir la cantidad de relleno al crear los lotes. Esto ayudó a acelerar el entrenamiento.
# * Si no está familiarizado con los modelos seq2seq, la entrada a menudo se revierte. Esto ayuda a que un modelo produzca mejores resultados porque cuando los datos de entrada se introducen en el modelo, el inicio de la secuencia se acercará más al inicio de la secuencia de salida.
# * Aunque he recortado mis degradados a ± 5, no noté mucha diferencia con ± 1.

# Cargamos las entradas del modelo

# In[62]:


input_data, targets, lr, keep_prob = model_inputs()


# La longitud de la secuencia será la longitud de línea máxima para cada lote

# In[63]:


sequence_length = tf.placeholder_with_default(max_line_length, None, name='sequence_length')


# Encontramos la forma de los datos de entrada para sequence_loss

# In[64]:


input_shape = tf.shape(input_data)


# Creamos el entrenamiento y los logits de inferencia

# In[65]:


train_logits, inference_logits = seq2seq_model(
    tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(answers_vocab_to_int), 
    len(questions_vocab_to_int), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, 
    questions_vocab_to_int)


# Creamos un tensor para los logits de inferencia, necesarios si carga una versión de punto de control del modelo

# In[66]:


tf.identity(inference_logits, 'logits')

with tf.name_scope("optimization"):
    # Loss function
    cost = tf.contrib.seq2seq.sequence_loss(
        train_logits,
        targets,
        tf.ones([input_shape[0], sequence_length]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


# Rellenamos las oraciones con < PAD > para que cada oración de un lote tenga la misma longitud

# In[67]:


def pad_sentence_batch(sentence_batch, vocab_to_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# Lote preguntas y respuestas juntas

# In[68]:


def batch_data(questions, answers, batch_size):
    for batch_i in range(0, len(questions)//batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        pad_questions_batch = np.array(pad_sentence_batch(questions_batch, questions_vocab_to_int))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, answers_vocab_to_int))
        yield pad_questions_batch, pad_answers_batch


# Validamos la capacitación con el 10% de los datos

# In[69]:


train_valid_split = int(len(sorted_questions)*0.15)


# Dividimos las preguntas y respuestas en entrenamiento y validación de datos

# In[70]:


train_questions = sorted_questions[train_valid_split:]
train_answers = sorted_answers[train_valid_split:]

valid_questions = sorted_questions[:train_valid_split]
valid_answers = sorted_answers[:train_valid_split]

print(len(train_questions))
print(len(valid_questions))


# In[71]:


display_step = 100 # Controlamos la pérdida de entrenamiento después de cada 100 lotes
stop_early = 0 
stop = 5 # Si la pérdida de validación disminuye en 5 controles consecutivos, detenemos el entrenamiento
validation_check = ((len(train_questions))//batch_size//2)-1 # Módulo para verificar la pérdida de validación
total_train_loss = 0 # Registramos la pérdida de entrenamiento para cada paso de visualización
summary_valid_loss = [] # Registramos la pérdida de validación para guardar mejoras en el modelo


# Nombramos nuestro modelo e iniciamos nuestro entrenamiento

# In[72]:


checkpoint = "best_model.ckpt" 

sess.run(tf.global_variables_initializer())

for epoch_i in range(1, epochs+1):
    for batch_i, (questions_batch, answers_batch) in enumerate(
            batch_data(train_questions, train_answers, batch_size)):
        start_time = time.time()
        _, loss = sess.run(
            [train_op, cost],
            {input_data: questions_batch,
             targets: answers_batch,
             lr: learning_rate,
             sequence_length: answers_batch.shape[1],
             keep_prob: keep_probability})

        total_train_loss += loss
        end_time = time.time()
        batch_time = end_time - start_time

        if batch_i % display_step == 0:
            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                  .format(epoch_i,
                          epochs, 
                          batch_i, 
                          len(train_questions) // batch_size, 
                          total_train_loss / display_step, 
                          batch_time*display_step))
            total_train_loss = 0

        if batch_i % validation_check == 0 and batch_i > 0:
            total_valid_loss = 0
            start_time = time.time()
            for batch_ii, (questions_batch, answers_batch) in                     enumerate(batch_data(valid_questions, valid_answers, batch_size)):
                valid_loss = sess.run(
                cost, {input_data: questions_batch,
                       targets: answers_batch,
                       lr: learning_rate,
                       sequence_length: answers_batch.shape[1],
                       keep_prob: 1})
                total_valid_loss += valid_loss
            end_time = time.time()
            batch_time = end_time - start_time
            avg_valid_loss = total_valid_loss / (len(valid_questions) / batch_size)
            print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))
            
            # Reducimos la tasa de aprendizaje, pero no por debajo de su valor mínimo
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate

            summary_valid_loss.append(avg_valid_loss)
            if avg_valid_loss <= min(summary_valid_loss):
                print('New Record!') 
                stop_early = 0
                saver = tf.train.Saver() 
                saver.save(sess, checkpoint)

            else:
                print("No Improvement.")
                stop_early += 1
                if stop_early == stop:
                    break
    
    if stop_early == stop:
        print("Stopping Training.")
        break


# ### Preparamos la pregunta para el modelo

# In[74]:


def question_to_seq(question, vocab_to_int):
    
    question = clean_text(question)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]


# Usamos una pregunta de los datos como su entrada

# In[75]:


random = np.random.choice(len(short_questions))
input_question = short_questions[random]
input_question = question_to_seq(input_question, questions_vocab_to_int)


# Rellenamos las preguntas hasta que sea igual a max_line_length

# In[76]:


input_question = input_question + [questions_vocab_to_int["<PAD>"]] * (max_line_length - len(input_question))


# Agregamos preguntas vacías para que input_data tenga la forma correcta

# In[77]:


batch_shell = np.zeros((batch_size, max_line_length))


# Establecemos la primera pregunta para que sea una pregunta de entrada

# In[78]:


batch_shell[0] = input_question


# Ejecutamos el modelo con la pregunta de entrada

# In[79]:


answer_logits = sess.run(inference_logits, {input_data: batch_shell, 
                                            keep_prob: 1.0})[0]


# Eliminamos el relleno de la Pregunta y Respuesta

# In[80]:


pad_q = questions_vocab_to_int["<PAD>"]
pad_a = answers_vocab_to_int["<PAD>"]


# Obtenemos las respuestas

# In[82]:


print('Question')
print('  Word Ids:      {}'.format([i for i in input_question if i != pad_q]))
print('  Input Words: {}'.format([questions_int_to_vocab[i] for i in input_question if i != pad_q]))

print('\nAnswer')
print('  Word Ids:      {}'.format([i for i in np.argmax(answer_logits, 1) if i != pad_a]))
print('  Response Words: {}'.format([answers_int_to_vocab[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))

