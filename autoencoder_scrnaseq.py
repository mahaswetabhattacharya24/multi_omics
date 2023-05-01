import os
import sys
import time
import scprep
import magic
import anndata
import scanpy
import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

# Global scanpy settings
scanpy.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
scanpy.logging.print_header()
scanpy.settings.set_figure_params(dpi=80, facecolor='white')

def evaluateModel(el1, dr1, el2, dr2,bottleneck, lr, power, bs,epoch):
    w2 = 10**(power)
    input_dim = X_train.shape[1]
    input_data = tf.keras.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(int(el1), activation='relu')(input_data)
    encoded = tf.keras.layers.Dropout(dr1)(encoded)
    encoded = tf.keras.layers.Dense(int(el2), activation='relu')(encoded)
    encoded = tf.keras.layers.Dropout(dr2)(encoded)
    encoded = tf.keras.layers.Dense(int(bottleneck), activation='relu')(encoded)
    #Classifier
    classifier = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(encoded)
    # Decoder
    decoder = tf.keras.layers.Dense(int(el2), activation='relu')(encoded)
    decoder = tf.keras.layers.Dense(int(el1), activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(input_dim, activation='relu', name='autoencoder')(decoder)
    # Ensemble model
    model = tf.keras.Model(input_data, [decoder,classifier, encoded])
    opt = tf.optimizers.RMSprop(learning_rate=lr)
    model.compile(loss=['mean_squared_error','binary_crossentropy'],loss_weights = [1,w2],optimizer=opt)
    model.fit(X_train,{'autoencoder': X_train, 'classifier': Y_train},
          epochs=int(epoch),shuffle=True, batch_size = int(bs),
          validation_data= (X_test, {'autoencoder': X_test,'classifier': Y_test}),
          verbose=0)
    pred_test = model.predict(X_test)
    R2_test = sklearn.metrics.r2_score(X_test.flatten().reshape(X_test.flatten().shape[0],1),
                           pred_test[0].flatten().reshape(pred_test[0].flatten().shape[0],1))
    classified_test = pred_test[1]; classified_test[classified_test >= 0.5] = 1; classified_test[classified_test < 0.5] = 0;
    accuracy_test = sklearn.metrics.accuracy_score(Y_test,classified_test)
    print('R2: {:.2f}, Accuracy: {:.2f}'.format(R2_test,accuracy_test))
    tf.keras.backend.clear_session()
    del model
    return(R2_test+accuracy_test)


if __name__ == "__main__":
    USAGE = "python test_phantom.py <directory>"
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)
    data = pd.read_csv(''.join([sys.argv[1], 'Gene_Count_per_Cell.tsv']), sep='\t', index_col=0).transpose()
    sc_data = anndata.AnnData(X=data)

    ## Data preprocessing: Extracting highly variable genes
    # Data preprocessing
    scanpy.pp.filter_cells(sc_data, min_genes=200)
    scanpy.pp.filter_genes(sc_data, min_cells=1)
    sc_data.var['mt'] = sc_data.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
    scanpy.pp.calculate_qc_metrics(sc_data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    scanpy.pp.normalize_total(sc_data, target_sum=1e4)
    scanpy.pp.log1p(sc_data)
    scanpy.pp.highly_variable_genes(sc_data, n_top_genes=1000, subset=True)

    ## Create new dataframe from the highly variable genes
    gene_counts = pd.DataFrame(sc_data.X, index=sc_data.obs.index, columns=sc_data.var.index)
    diag_match = gene_counts.index.str.match('(^[^_]+_)').astype(int)
    gene_counts['Condition'] = np.where(diag_match, 'Diabetic', 'Control')

    ## Denoising scRNAseq data using MAGIC following square root transformation
    gene_counts.iloc[:, :-1] = np.exp((gene_counts.iloc[:, :-1]))
    gene_counts.iloc[:, :-1] = (gene_counts.iloc[:, :-1]) - 1
    gene_counts.iloc[:, :-1] = scprep.transform.sqrt(gene_counts.iloc[:, :-1])
    gene_counts = gene_counts.reset_index(drop=True)
    magic_op = magic.MAGIC()
    gene_counts.iloc[:, :-1] = magic_op.fit_transform(gene_counts.iloc[:, :-1])

    ## Count number of cells with each condition and balance if there is class imbalance
    data = gene_counts.copy()
    cellCount = data['Condition'].value_counts()
    if cellCount['Control'] > cellCount['Diabetic']:
        dataResampled = pd.concat([data.loc[data['Condition'] == 'Control'],
                                   sklearn.utils.resample(data.loc[data['Condition'] == 'Diabetic'], replace=True,
                                            n_samples=cellCount['Control'], random_state=42)])
    elif cellCount['Control'] < cellCount['Diabetic']:
        dataResampled = pd.concat([data.loc[data['Condition'] == 'Diabetic'],
                                   sklearn.utils.resample(data.loc[data['Condition'] == 'Control'], replace=True,
                                            n_samples=cellCount['Diabetic'], random_state=42)])
    else:
        print('Two classes are balanced in the dataset')

    ## Split Data into training and test set
    # %% Split Data
    splitSize = 0.1
    train, test = sklearn.model_selection.train_test_split(dataResampled, test_size=splitSize, shuffle=True)
    # %% Input output scaling
    X_train = train.iloc[:, :-1];
    X_train = sklearn.preprocessing.MinMaxScaler().fit_transform(X_train)
    X_test = test.iloc[:, :-1];
    X_test = sklearn.preprocessing.MinMaxScaler().fit_transform(X_test)
    Y_train = train.iloc[:, -1];
    Y_train = sklearn.preprocessing.LabelEncoder().fit_transform(Y_train)
    Y_test = test.iloc[:, -1];
    Y_test = sklearn.preprocessing.LabelEncoder().fit_transform(Y_test)

    pbounds = {'el1': (500, 1000),
               'dr1': (0, 0.5),
               'el2': (200, 500),
               'dr2': (0, 0.5),
               'bottleneck': (50, 150),
               'lr': (1e-8, 1e-2),
               'bs': (32, 512),
               'epoch': (50, 100),
               'power': (-10, 10)
               }

    optimizer = BayesianOptimization(
        f=evaluateModel,
        pbounds=pbounds,
        verbose=2)

    start_time = time.time()
    optimizer.maximize(init_points=10, n_iter=100)
    time_took = time.time() - start_time

    # %% Train
    input_dim = X_train.shape[1]
    bottleneck = optimizer.max['params']['bottleneck']
    el1 = optimizer.max['params']['el1']
    dr1 = optimizer.max['params']['dr1']
    el2 = optimizer.max['params']['el2']
    dr2 = optimizer.max['params']['dr2']
    lr = optimizer.max['params']['lr']
    bs = optimizer.max['params']['bs']
    w2 = 10 ** optimizer.max['params']['power']
    # epoch = 70#optimizer.max['params']['epoch']
    input_data = tf.keras.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(int(el1), activation='relu')(input_data)
    encoded = tf.keras.layers.Dropout(dr1)(encoded)
    encoded = tf.keras.layers.Dense(int(el2), activation='relu')(encoded)
    encoded = tf.keras.layers.Dropout(dr2)(encoded)
    encoded = tf.keras.layers.Dense(int(bottleneck), activation='relu')(encoded)
    # Classifier
    classifier = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(encoded)
    # Decoder
    decoder = tf.keras.layers.Dense(int(el2), activation='relu')(encoded)
    decoder = tf.keras.layers.Dense(int(el1), activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid', name='autoencoder')(decoder)
    # Ensemble model
    model = tf.keras.Model(input_data, [decoder, classifier, encoded])
    opt = tf.optimizers.RMSprop(learning_rate=lr)
    model.compile(loss=['mean_squared_error', 'binary_crossentropy'], loss_weights=[1, w2], optimizer=opt)
    # model.compile(loss=['mean_squared_error'],optimizer=opt)
    history = model.fit(X_train, {'autoencoder': X_train, 'classifier': Y_train},
                        epochs=1000, shuffle=True, batch_size=200,
                        validation_data=(X_test, {'autoencoder': X_test, 'classifier': Y_test}),
                        verbose=2)

    # %% Plot losses
    plt.figure(dpi=600)
    plt.subplot(2, 1, 1)
    plt.plot(history.history['autoencoder_loss'], label=' Training loss')
    plt.plot(history.history['val_autoencoder_loss'], label='Validation loss')
    plt.legend(prop={"size": 5}, loc='upper right')
    plt.title('Autoencoder')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['classifier_loss'], label=' Training loss')
    plt.plot(history.history['val_classifier_loss'], label='Validation loss')
    plt.legend(prop={"size": 5}, loc='upper right')
    plt.title('Classifier')
    # %% Prediction
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    classified_train = pred_train[1]
    classified_train[classified_train >= 0.5] = 1
    classified_train[classified_train < 0.5] = 0
    classified_test = pred_test[1]
    classified_test[classified_test >= 0.5] = 1
    classified_test[classified_test < 0.5] = 0

    ## Scoring
    R2_train = r2_score(X_train.flatten().reshape(X_train.flatten().shape[0], 1),
                        pred_train[0].flatten().reshape(pred_train[0].flatten().shape[0], 1))
    print('Train Data: R2: {:.2f}'.format(R2_train))

    accuracy_train = accuracy_score(Y_train, classified_train)
    print('Train Data: Accuracy: {:.2f}'.format(accuracy_train))

    R2_test = r2_score(X_test.flatten().reshape(X_test.flatten().shape[0], 1),
                       pred_test[0].flatten().reshape(pred_test[0].flatten().shape[0], 1))
    print('Test Data: R2: {:.2f}'.format(R2_test))

    accuracy_test = accuracy_score(Y_test, classified_test)
    print('Test Data: Accuracy: {:.2f}'.format(accuracy_test))









