from keras.models import Model
from keras.layers import Input, Conv1D, LeakyReLU, MaxPool1D, LSTM, Bidirectional, TimeDistributed, Dense, Reshape, Softmax,Flatten
from keras.layers import UpSampling2D, Conv2DTranspose


def temporal_autoencoder(n_clusters,input_dim,timesteps, n_filters=50, kernel_size=10, strides=1, pool_size=10, n_units=[50, 1, 10]):
    # 创建聚类网络
    #(None,720,1)
    x = Input(shape=(timesteps,input_dim), name='input_seq')
    #(None,720,50)
    tae = Conv1D(n_filters, kernel_size, strides=strides, padding='same', activation='linear')(x)
    tae = LeakyReLU()(tae)
    #(None,72,50)
    tae = MaxPool1D(pool_size)(tae)
    #(None,72,10)
    tae = Bidirectional(LSTM(n_units[2],return_sequences=True), merge_mode='sum')(tae)
    tae = LeakyReLU()(tae)
    #(None,72,1)
    tae = Bidirectional(LSTM(n_units[1],return_sequences=True), merge_mode='sum')(tae)
    tae = LeakyReLU()(tae)

    # 聚类网络的前半部分，没有softmax层
    before = Model(inputs=x,outputs=tae,name='before_softmax')

    # 聚类网络的后半部分，softmax层
    #(None,72)
    sml = Flatten(name='flatten')(tae)
    sml = Dense(n_clusters,name='dense')(sml)
    #(None,k)
    output = Softmax(name='sm')(sml)

    # 整个聚类网络
    cModel = Model(inputs=x, outputs=output, name='ClusteringModel')

    # 预训练阶段，要把聚类网络的前半部分当encoder用
    decoded = Reshape((-1, 1, n_units[1]))(tae)
    decoded = UpSampling2D((pool_size, 1))(decoded)
    decoded = Conv2DTranspose(input_dim, (kernel_size, 1), padding='same')(decoded)
    output = Reshape((-1, input_dim))(decoded)
    pretrain_ae = Model(inputs=x,outputs=output,name='Pretrain_AE')

    # 把聚类网络中的softmax层单独取出，因为后续要单独训练
    softMax_input=Input(shape=(timesteps//pool_size,n_units[1]),name='softmax_input')
    softMax = cModel.get_layer('flatten')(softMax_input)
    softMax = cModel.get_layer('dense')(softMax)
    softMax_output= cModel.get_layer('sm')(softMax)
    softMax_model = Model(inputs=softMax_input, outputs=softMax_output,name='softmax_model')
    

    # 创建一组自编码器
    autoencoders=[]
    for index in range(n_clusters):
        # Input
        x = Input(shape=(timesteps,input_dim), name='input_seq'+str(index))

        # Encoder
        encoded = Conv1D(n_filters, kernel_size, strides=strides, padding='same', activation='linear')(x)
        encoded = LeakyReLU()(encoded)
        encoded = MaxPool1D(pool_size)(encoded)
        encoded = Bidirectional(LSTM(n_units[0], return_sequences=True), merge_mode='sum')(encoded)
        encoded = LeakyReLU()(encoded)
        encoded = Bidirectional(LSTM(n_units[1], return_sequences=True), merge_mode='sum')(encoded)
        encoded = LeakyReLU(name='latent')(encoded)

        # Decoder
        decoded = Reshape((-1, 1, n_units[1]), name='reshape')(encoded)
        decoded = UpSampling2D((pool_size, 1), name='upsampling')(decoded)  #decoded = UpSampling1D(pool_size, name='upsampling')(decoded)
        decoded = Conv2DTranspose(input_dim, (kernel_size, 1), padding='same', name='conv2dtranspose')(decoded)
        output = Reshape((-1, input_dim), name='output_seq')(decoded)  #output = Conv1D(1, kernel_size, strides=strides, padding='same', activation='linear', name='output_seq')(decoded)

        # AE model
        autoencoder = Model(inputs=x, outputs=output, name='AE'+str(index))

        autoencoders.append(autoencoder)


    return autoencoders, cModel, softMax_model, before, pretrain_ae