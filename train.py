# -*- coding: utf-8 -*-
"""
Main model training for WyCryst Framwork
"""

from data import *
from featurizer import *
from utils import *
from PVAE import  PVAE
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau,CSVLogger,LearningRateScheduler,EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import joblib
import warnings
from tensorflow.python.framework.ops import disable_eager_execution


disable_eager_execution()
warnings.filterwarnings("ignore")
loo_val = True

def main():
    # read ternary compound data into dataframe
    print('---------Building Input Data---------------')
    sup_prop = ['formation_energy_per_atom']
    wyckoff_multiplicity_array, wyckoff_DoF_array = wyckoff_para_loader()
    df_clean = get_input_df()
    if loo_val:
        i = df_clean[((df_clean.pretty_formula == 'CaTiO3') & (df_clean.spacegroup_number == 62))].index
        df_clean = df_clean.drop(i)
        df_clean = df_clean.reset_index()

    Crystal, sg = wyckoff_represent(df_clean, 3, 20)

    a = np.stack(Crystal, axis=0)
    a.shape
    X = a
    X2 = np.stack(sg, axis=0)
    X2 = X2[:, :, 0]
    Y = df_clean[['formation_energy_per_atom'] + ['ind']].values
    scaler_y = MinMaxScaler()
    sup_dim = len(sup_prop)
    Y[:, :sup_dim] = scaler_y.fit_transform(Y[:, :sup_dim])
    X_train, X_test, X2_train, X2_test, y_train, y_test = train_test_split(X, X2, Y, test_size=0.2, random_state=21)

    # print PVAE input and output shape
    print('---------Printing Input Shape---------------')
    print('Wyckoff array size:',X.shape, '\nSpace Group array size:', X2.shape
          , '\nTarget Property array size:', Y.shape)

    # building and training PVAE
    print('---------Building PVAE---------------')
    VAE, encoder, decoder, regression, vae_loss, loss_recon, loss_sg, loss_KL\
        , loss_prop, loss_formula = PVAE(X_train, y_train)

    # Callbacks
    print('---------Training PVAE---------------')
    # CSV = CSVLogger('temp_files/Wyckoff_log.csv')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=6, min_lr=2e-5)

    VAE.add_loss(vae_loss)
    VAE.add_metric(loss_KL, name='kl_loss', aggregation='mean')
    VAE.add_metric(loss_prop, name='prop_loss', aggregation='mean')
    VAE.add_metric(loss_recon, name='recon_loss', aggregation='mean')
    VAE.add_metric(loss_sg, name='sg_loss', aggregation='mean')
    VAE.add_metric(loss_formula, name='wyckoff_formula_loss', aggregation='mean')
    VAE.add_metric(vae_loss, name='total_loss', aggregation='mean')

    VAE.compile(optimizer=RMSprop(learning_rate=2e-4))
    VAE.fit(x=[X_train, X2_train, y_train[:, :1]], shuffle=True,
            batch_size=256, epochs=48, callbacks=[reduce_lr],
            validation_data=([X_test, X2_test, y_test[:, :1]], None))

    # Print model performance
    print('---------Printing Result---------------')
    vae_x, vae_sg = VAE.predict([X_test, X2_test, y_test[:, :1]])
    # Property MAE
    y_result = regression.predict([X_test, X2_test])
    y_result = scaler_y.inverse_transform(y_result)
    y_test1 = scaler_y.inverse_transform(y_test[:, :1])
    MAE_result = MAE(y_test1, y_result)
    print('property-learning branch MAE', MAE_result,'eV/atom')
    # Element ACC
    Element = joblib.load('./data/element.pkl')
    E_v = np_utils.to_categorical(np.arange(0, len(Element), 1))
    accu = []
    vae_ele = []
    X_ele = []
    for i in range(num_ele):
        ele_v = np.argmax(vae_x[:, 0:len(E_v), i], axis=1)
        ele_t = np.argmax(X_test[:, 0:len(E_v), i], axis=1)
        vae_ele.append(ele_v)
        X_ele.append(ele_t)
        accu1 = 100 * round(metrics.accuracy_score(ele_v, ele_t), 3)
        accu.append(accu1)
    print('Element accuracy %', accu)
    # Wyckoff ACC
    X_test1 = X_test
    wyckoff_test = []
    for i in range(vae_x.shape[0]):
        wyckoff_test.append(np.all(np.around(vae_x[i][198:224], decimals=0) == X_test1[i][198:224]))
    wyckoff_test = np.array(wyckoff_test)
    print('Wyckoff Accuracy %',np.mean(wyckoff_test))
    # SG ACC
    sg_v1 = np.argmax(vae_sg, axis=1)
    sg_t = np.argmax(X2_test, axis=1)
    accu1 = 100 * round(metrics.accuracy_score(sg_v1, sg_t), 3)
    print('SG Accuracy %', accu1)

    # Save model
    print('---------Saving Model---------------')
    print('trained model weights saved to temp_files/temp_model')
    VAE.save_weights('temp_files/vae_models/temp_model/VAE_weights.h5')
    encoder.save_weights('temp_files/vae_models/temp_model/encoder.h5')
    decoder.save_weights('temp_files/vae_models/temp_model/decoder.h5')
    regression.save_weights('temp_files/vae_models/temp_model/regression.h5')

    print('---------End---------------')


if __name__ == '__main__':
    main()