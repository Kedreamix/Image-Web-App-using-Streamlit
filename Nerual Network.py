import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Nadam  # Stochastic Gradient Descend
from tensorflow.keras.utils import to_categorical, plot_model
from PIL import Image

st.write("""
# Neural Network with 3 layer 
""")

np.random.seed(1671)  # for reproducibility

@st.cache
def get_data():
    # data: shuffled and split between train and test sets
    #
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
    # RESHAPED = 784
    #
    X_train = X_train.reshape(60000, RESHAPED)
    X_test = X_test.reshape(10000, RESHAPED)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize 
    #
    X_train /= 255
    X_test /= 255


    # convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, NB_CLASSES)
    Y_test = to_categorical(y_test, NB_CLASSES)

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    st.write("The original data: Minst Data")
    st.write("Shape of train dataset:",X_train.shape)
    st.write("Shape of test dataset:",X_test.shape)
    return X_train,Y_train,X_test,Y_test

# network and training
epochs = st.slider('Fill in number of epoch: ', min_value=1, max_value=100, value=10, step=1)
batch_sz = st.slider('Specify batch size: ', min_value=16, max_value=1024,value=64,step=16)
amount_of_output_classes = st.number_input('Specify amount of output classes', min_value=10, format=None, step=1)
NB_EPOCH = epochs
BATCH_SIZE = batch_sz
VERBOSE = 1
NB_CLASSES = amount_of_output_classes  # number of outputs = number of digits
amount_of_hidden_nodes = st.number_input('Specify amount of hidden nodes', min_value=50, format=None, step=10)
N_HIDDEN = amount_of_hidden_nodes
# st.write("""
# #### Optimizer Stochastic Gradient Descent
# """)
# OPTIMIZER = Nadam()  # SGD optimizer, explained later in this chapter

st.sidebar.markdown("See the model performances or pick oneto have more details and play with it:")
optimizer_func_name = st.sidebar.selectbox("Optimizer function name：",
                               ('sgd',
                                'adam',
                                'nadam',
                                'adamax',
                                'rmsprop',
                                'adagrad',
                                'adadelta'))

VALIDATION_SPLIT = 0.2  # how much TRAIN is reserved for VALIDATION

RESHAPED = 784

# 10 outputs
# final stage is softmax

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))

# select activation
activation_func_name = st.sidebar.selectbox("Select Activation function name: ", ('softmax', 'relu', 'sigmoid', 'tanh'))
model.add(Activation(activation_func_name))

model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

# select loss function
loss_func_name = st.sidebar.selectbox("Select Loss function name: ",
                          ('categorical_crossentropy',
                           'kullback_leibler_divergence', 
                           'poisson'))

if st.sidebar.checkbox('Summary of model'):
    model.summary()
    model_img = 'model_img.png'
    plot_model(model, to_file=model_img, show_shapes=True)
    image = Image.open(model_img)        
    st.sidebar.image(image, caption='Summary of Model',use_column_width=True)

@st.cache
def compile_model(model,loss_func_name,optimizer_func_name):
    model.compile(loss=loss_func_name,
                optimizer=optimizer_func_name,
                metrics=['accuracy'])

@st.cache
def train_model(model,X_train, Y_train, BATCH_SIZE, NB_EPOCH):
    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=NB_EPOCH,
                        verbose=VERBOSE,
                        validation_split=VALIDATION_SPLIT)
    return history
 

if st.button('Process'):
    
    
    # compile model
    # compile_model(model,loss_func_name,optimizer_func_name)
    model.compile(loss=loss_func_name,
                optimizer=optimizer_func_name,
                metrics=['accuracy'])
        
    # X_train,Y_train,X_test,Y_test = get_data()
    # data: shuffled and split between train and test sets
    #
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
    # RESHAPED = 784
    #
    X_train = X_train.reshape(60000, RESHAPED)
    X_test = X_test.reshape(10000, RESHAPED)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize 
    #
    X_train /= 255
    X_test /= 255


    # convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, NB_CLASSES)
    Y_test = to_categorical(y_test, NB_CLASSES)

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    st.write("The original data: Minst Data")
    st.write("Shape of train dataset:",X_train.shape)
    st.write("Shape of test dataset:",X_test.shape) 
        
    # 开始训练
    # history = train_model(model,X_train, Y_train, BATCH_SIZE, NB_EPOCH)
    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=NB_EPOCH,
                        verbose=VERBOSE,
                        validation_split=VALIDATION_SPLIT)
    
    
    # list all data in history
    print(history.history.keys())

    # plot for accuracy train and test
    fig, ax = plt.subplots()
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    line1 = ax.plot(range(1, epochs + 1), history.history['accuracy'], label='train')
    line2 = ax.plot(range(1, epochs + 1), history.history['val_accuracy'], label='test')
    ax.legend(['train', 'test'])
    st.pyplot(fig)

    # plot for loss train and test
    fig, ax = plt.subplots()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    line1 = ax.plot(range(1, epochs + 1), history.history['loss'], label='train')
    line2 = ax.plot(range(1, epochs + 1), history.history['val_loss'], label='test')
    ax.legend(['train', 'test'])
    st.pyplot(fig)


    score = model.evaluate(X_test, Y_test, verbose=VERBOSE)

    md_results = f"Test loss is **{score[0]}** \n" \
                f" Test accuracy: **{score[1]}**."
    st.markdown(md_results)

    print("\nTest loss value :", score[0])
    print('Test accuracy :', score[1])