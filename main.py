# Utilities
import warnings
import os
import numpy as np
import argparse
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from time import time

# Keras
from keras.models import Model
from keras.layers import Dense, Reshape, UpSampling2D, Conv2DTranspose, GlobalAveragePooling1D, Softmax
from keras.losses import kullback_leibler_divergence
import keras.backend as K

# local
from dataloader import load_data
from DATC import DATC
from metrics import *


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='Computers', help='UCR/UEA univariate or multivariate dataset')
    #parser.add_argument('--validation', default=False, type=bool, help='use train/validation split')
    parser.add_argument('--n_clusters', default=None, type=int, help='number of clusters')
    parser.add_argument('--n_filters', default=50, type=int, help='number of filters in convolutional layer')
    parser.add_argument('--kernel_size', default=10, type=int, help='size of kernel in convolutional layer')
    parser.add_argument('--strides', default=1, type=int, help='strides in convolutional layer')
    parser.add_argument('--pool_size', default=10, type=int, help='pooling size in max pooling layer')
    parser.add_argument('--n_units', default=[50,1,10], type=int, help='numbers of units in the BiLSTM layers')
    parser.add_argument('--cluster_init', default='kshape', type=str, choices=['kmeans', 'kshape'], help='cluster initialization method')
    parser.add_argument('--pretrain_epochs', default=20, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--eval_epochs', default=1, type=int)
    parser.add_argument('--save_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--tol', default=0.001, type=float, help='tolerance for stopping criterion')
    parser.add_argument('--patience', default=5, type=int, help='patience for stopping criterion')
    parser.add_argument('--save_dir', default='results/tmp')
    args = parser.parse_args()
    print(args)

    # 创建保存模型和日志的目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 读取数据（使用不同的数据，即改造此部分）
    (X_train, y_train)= load_data(args.dataset)

    # 数据预处理
    # TODO

    # 定义好cluster的个数，如果没有找到cluster的个数，就去看看标签的个数（实际数据是有监督的情况）
    if args.n_clusters is None:
        args.n_clusters = len(np.unique(y_train))

    # 预训练时使用的优化器
    pretrain_optimizer = 'adam'

    # 定义DATC模型
    datc= DATC(n_clusters=args.n_clusters,          # cluster个数
              input_dim=X_train.shape[-1],          # 特征数，一般是单变量时序聚类，此处一般是1
              timesteps=X_train.shape[1],
              n_filters=args.n_filters,             # 1D CNN中filter的个数
              kernel_size=args.kernel_size,         # kernel的大小
              strides=args.strides,                 # 步长
              pool_size=args.pool_size,             # pooling的大小
              n_units=args.n_units,                 # LSTM中神经元个数
              )
    # 初始化DATC模型
    datc.initialize()

    print("-------------cModel Preview-------------")
    datc.cModel.summary()
    print("----------Auto-encoder Preview----------")
    datc.autoencoders[0].summary()
    print("------------Softmax Preview-------------")
    datc.softMax.summary()
    print("----------TAE Preview--------------------")
    datc.TAE.summary()
    print("----------Pretrain_AE Preview--------------------")
    datc.preTrainAE.summary()
    print("-------------Preview end----------------")  

    print("-------------Begin to Pretain-------------------")
    t0=time()
    datc.pretrain(X_train,args.n_clusters,pretrain_optimizer,
                    args.pretrain_epochs,args.batch_size,args.save_dir)
    print('Pretrain time: ', (time() - t0))

    datc.compile('adam')
    # 开始训练模型
    t0 = time()
    datc.fit(X_train,y_train,args.n_clusters,args.epochs,args.eval_epochs,args.save_epochs,
            args.batch_size,args.tol, args.patience,args.save_dir)
    print('Training time: ', (time() - t0))

    # Evaluate
    print('Performance (TRAIN)')
    results = {}
    p = datc.cModel.predict(X_train)
    y_pred = p.argmax(axis=1)
    results['acc'] = cluster_acc(y_train, y_pred)
    results['pur'] = cluster_purity(y_train, y_pred)
    results['nmi'] = metrics.normalized_mutual_info_score(y_train, y_pred)
    results['ari'] = metrics.adjusted_rand_score(y_train, y_pred)
    print(results)

