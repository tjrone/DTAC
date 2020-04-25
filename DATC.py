from TAE import temporal_autoencoder
from tslearn.clustering import KShape
from time import time
import numpy as np
import csv

# Keras
from keras.models import Model
from keras.layers import Dense, Reshape, UpSampling2D, Conv2DTranspose, GlobalAveragePooling1D, Softmax
from keras.losses import kullback_leibler_divergence
import keras.backend as K
from keras.utils.np_utils import to_categorical

# local 
from loss import myloss
from metrics import *

class DATC:
    """
    Deep Auto-encoders Temporal Clustering (DATC) model

    # Arguments
        n_clusters: number of clusters
        input_dim: input dimensionality
        n_filters: number of filters in convolutional layer
        timesteps: length of time series.
        kernel_size: size of kernel in convolutional layer
        strides: strides in convolutional layer
        pool_size: pooling size in max pooling layer, must divide the time series length
        n_units: numbers of units in the two BiLSTM layers
        dist_metric: distance metric used in cluster_init
        cluster_init: cluster initialization method
    """

    def __init__(self, n_clusters, input_dim, timesteps=None,
                 n_filters=50, kernel_size=10, strides=1, pool_size=10, n_units=[50, 1, 10],
                cluster_init='kshape'):
        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.timesteps = timesteps
        self.strides = strides
        self.pool_size = pool_size
        self.n_units = n_units
        self.softMax= self.autoencoders = self.cModel= self.TAE = self.preTrainAE = None

    def initialize(self):
        
        # 创建一组自编码器和聚类网络
        self.autoencoders, self.cModel, self.softMax, self.TAE, self.preTrainAE = temporal_autoencoder(
            n_clusters=self.n_clusters,
            input_dim=self.input_dim,
            timesteps=self.timesteps,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            pool_size=self.pool_size,
            n_units=self.n_units
        )


    def compile(self, optimizer):
        """
        Compile DTC model

        # Arguments
            optimizer: optimization algorithm
        """
        for i in range (len(self.autoencoders)):
            self.autoencoders[i].compile(loss='mse',optimizer=optimizer)
        self.softMax.compile(loss=myloss,optimizer=optimizer)

    def pretrain(self, X,
                num_clusters,
                optimizer='adam',
                epochs=10,
                batch_size=64,
                save_dir='results/tmp',
                verbose=1):
        # 用整个数据集训练一个自编码器，其中的encoder就是我们的聚类网络的前半部分
        print("start to pretrain an AE for entire dataset.")
        self.preTrainAE.compile(optimizer=optimizer,loss='mse')
        self.preTrainAE.fit(X,X,batch_size=batch_size,epochs=epochs)
        embeded = self.TAE.predict(X)
        print("shape of embeded vector:")
        print(embeded.shape)
        # 通过k-shape得到标签
        seed = 0
        np.random.seed(seed)
        ks = KShape(n_clusters=num_clusters, verbose=True, random_state=seed)
        y_pred = ks.fit_predict(embeded)
        # 对标签进行独热编码
        categorical_labels = to_categorical(y_pred, num_classes=num_clusters)
        print('Begin to pretrain cModel')
        # 编译模型
        self.cModel.compile(optimizer=optimizer,loss='categorical_crossentropy')
        # 训练 cModel
        self.cModel.fit(X,categorical_labels,batch_size=batch_size,epochs=epochs)

        # train k Autoencoders
        for i in range(num_clusters):
            # Find the time-series that belong to #i cluster
            print('Begin to pretrain AE {}'.format(i))
            x=X[y_pred==i]
            self.autoencoders[i].compile(optimizer=optimizer,loss='mse')
            self.autoencoders[i].fit(x,x,batch_size=batch_size,epochs=epochs)
        print('-----------Pretrain End.-----------')



    def fit(self, X_train, y_train=None,
            n_clusters=3,
            epochs=100,
            eval_epochs=10,
            save_epochs=10,
            batch_size=64,
            tol=0.001,
            patience=5,
            save_dir='results/tmp'):
        """
        Training procedure

        # Arguments
           X_train: training set
           y_train: (optional) training labels
           epochs: number of training epochs
           eval_epochs: evaluate metrics on train/val set every eval_epochs epochs
           save_epochs: save model weights every save_epochs epochs
           batch_size: training batch size
           tol: tolerance for stopping criterion
           patience: patience for stopping criterion
           save_dir: path to existing directory where weights and logs are saved
        """
        # Logging file
        logfile = open(save_dir + '/dtc_log.csv', 'w')
        fieldnames = ['epoch']
        if y_train is not None:
            fieldnames += ['acc', 'pur', 'nmi', 'ari']
        logwriter = csv.DictWriter(logfile, fieldnames)
        logwriter.writeheader()

        y_pred_last = None
        patience_cnt = 0

        print('Training for {} epochs.\nEvaluating every {} and saving model every {} epochs.'.format(epochs, eval_epochs, save_epochs))

        hidden=self.TAE.predict(X_train)

        for epoch in range(epochs):
            print('epoch {}'.format(epoch))
            # Evaluate losses and metrics on training set
            if epoch % eval_epochs == 0:

                # Initialize log dictionary
                logdict = dict(epoch=epoch)

                # 得到目前模型的预测标签
                p=self.cModel.predict(X_train)
                one_hots=props_to_onehot(p)
                y_pred = np.array([np.argmax(one_hot) for one_hot in one_hots])
                
                # Evaluate the clustering performance using labels
                if y_train is not None:
                    logdict['acc'] = cluster_acc(y_train, y_pred)
                    logdict['pur'] = cluster_purity(y_train, y_pred)
                    logdict['nmi'] = metrics.normalized_mutual_info_score(y_train, y_pred)
                    logdict['ari'] = metrics.adjusted_rand_score(y_train, y_pred)
                    print('[Train] - Acc={:f}, Pur={:f}, NMI={:f}, ARI={:f}'.format(logdict['acc'], logdict['pur'],
                                                                                    logdict['nmi'], logdict['ari']))

                logwriter.writerow(logdict)

                # 检查是否能停止训练
                if y_pred_last is not None:
                    assignment_changes = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if epoch > 0 and assignment_changes < tol:
                    patience_cnt += 1
                    print('Assignment changes {} < {} tolerance threshold. Patience: {}/{}.'.format(assignment_changes, tol, patience_cnt, patience))
                    if patience_cnt >= patience:
                        print('Reached max patience. Stopping training.')
                        logfile.close()
                        break
                else:
                    patience_cnt = 0
                
            # Save intermediate model and plots
            if epoch % save_epochs == 0:
                self.cModel.save_weights(save_dir + '/DATC_model_' + str(epoch) + '.h5')
                print('Saved model to:', save_dir + '/DATC_model_' + str(epoch) + '.h5')

            # 得到目前auto-encoders的重构误差
            matrixD=np.empty([X_train.shape[0],0])
            for i in range (n_clusters):
                reconX=self.autoencoders[i].predict(X_train)
                d=(0.5*(np.square(np.abs(X_train-reconX)))).sum(axis=1)
                matrixD=np.append(matrixD,d,axis=1)
            # 训练聚类网络
            self.softMax.fit(hidden, matrixD, epochs=1, batch_size=batch_size,verbose=1)

            # 训练k个自编码器
            p=self.softMax.predict(hidden)
            one_hots=props_to_onehot(p)
            labels = [np.argmax(one_hot)for one_hot in one_hots]
            labels = np.array(labels)
            print(labels)
            for i in range (n_clusters):
                x=X_train[labels==i]
                print(len(x))
                self.autoencoders[i].compile(optimizer='adam',loss='mse')
                self.autoencoders[i].fit(x,x,epochs=1,batch_size=batch_size)

        # Save the final model
        logfile.close()
        print('Saving model to:', save_dir + '/DTC_model_final.h5')
        self.cModel.save_weights(save_dir + '/DTC_model_final.h5')


# sortmax 结果转 onehot
def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b