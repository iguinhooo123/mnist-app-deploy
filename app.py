import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Título e Subtítulo
st.title('Classificador de Números (MNIST)')
st.write('Envie uma imagem de um número desenhado à mão (0-9).')

# 2. Carregar o modelo (Cache para não recarregar a cada clique)
@st.cache_resource
def load_my_model():
    # Certifique-se que o nome do arquivo é o mesmo que você baixou
    return tf.keras.models.load_model('final_CNN_model.h5')

model = load_my_model()

# 3. Widget de Upload de Arquivo
file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

if file is not None:
    # Mostrar a imagem enviada
    image = Image.open(file).convert('L') # Converte para escala de cinza
    st.image(image, caption='Imagem enviada', use_column_width=True)
    
    # 4. Pré-processamento (Igual ao que você fez no treino)
    img_array = np.array(image.resize((28, 28))) # Redimensionar para 28x28
    img_array = img_array / 255.0            # Normalizar
    img_array = img_array.reshape(1, 28, 28, 1) # Formato que a CNN espera (batch, height, width, channel)

    # 5. Botão de Previsão
    if st.button('Classificar'):
        prediction = model.predict(img_array)
        label = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        st.success(f'O modelo prevê que este número é: *{label}*')
        st.info(f'Certeza: {confidence:.2f}%')