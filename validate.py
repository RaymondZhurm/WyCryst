# -*- coding: utf-8 -*-
"""
Main model validation for WyCryst Framwork
"""

from data import *
from featurizer import *
from utils import *
from PVAE import *
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau,CSVLogger,LearningRateScheduler,EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import re
from os.path import abspath, dirname, join
from tqdm import tqdm
import itertools
import warnings
from tensorflow.python.framework.ops import disable_eager_execution


disable_eager_execution()
warnings.filterwarnings("ignore")
repro_result = True
loo_val = True


def main():
    # read ternary compound data into dataframe
    print('---------Building Input Data---------------')
    sup_prop = ['formation_energy_per_atom']
    wyckoff_multiplicity_array, wyckoff_DoF_array = wyckoff_para_loader()
    df_clean = get_input_df()

    if loo_val:
        df_loo = df_clean[((df_clean.pretty_formula == 'CaTiO3') & (df_clean.spacegroup_number == 62))]
        df_loo = df_loo.reset_index()
        i = df_clean[((df_clean.pretty_formula == 'CaTiO3') & (df_clean.spacegroup_number == 62))].index
        df_clean = df_clean.drop(i)
        df_clean = df_clean.reset_index()
    else:
        # define dataframes for sampling here
        df_clean = df_clean.reset_index()
        pass

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
    if loo_val:
        np.random.seed(124123)
        test_noise = np.random.normal(0, 1, (25000, 256))

    VAE, encoder, decoder, regression, vae_loss, loss_recon, loss_sg, loss_KL\
        , loss_prop, loss_formula = PVAE(X_train, y_train)

    # Callbacks
    print('---------Loading Trained Model---------------')
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

    if loo_val:
        VAE.load_weights('temp_files/vae_models/CaTiO3_sg62/VAE_weights.h5')
        encoder.load_weights('temp_files/vae_models/CaTiO3_sg62/encoder.h5')
        decoder.load_weights('temp_files/vae_models/CaTiO3_sg62/decoder.h5')
        regression.load_weights('temp_files/vae_models/CaTiO3_sg62/regression.h5')
    else:
        # define the model for sampling (trained model)
        VAE.load_weights('temp_files/vae_models/temp_model/VAE_weights.h5')
        encoder.load_weights('temp_files/vae_models/temp_model/encoder.h5')
        decoder.load_weights('temp_files/vae_models/temp_model/decoder.h5')
        regression.load_weights('temp_files/vae_models/temp_model/regression.h5')


    # sampling from latent space
    print('---------Sampling Result---------------')
    vae_x, vae_sg = VAE.predict([X_test, X2_test, y_test[:, :1]])
    df_selected = df_clean[df_clean['pretty_formula'] == 'CaTiO3']
    df_selected = df_selected.reset_index()

    target_C, target_sg = wyckoff_represent(df_selected, 3, 20)
    a = np.stack(target_C, axis=0)
    X = a
    X2 = np.stack(target_sg, axis=0)
    X2 = X2[:, :, 0]
    test_latent = encoder.predict([X_test, X2_test])
    sample_latent = encoder.predict([X, X2])

    # Set number of purturbing instances around each compound
    Nperturb = 5000
    # Set local purturbation (Lp) scale
    Lp_scale = 1.4
    # set random state
    if loo_val:
        np.random.seed(124123)

    # Sample (Lp)
    samples = sample_latent
    samples = np.tile(samples, (Nperturb, 1))
    gaussian_noise = np.random.normal(0, 1, samples.shape)
    samples = samples + gaussian_noise * Lp_scale
    wyckoff_designs = decoder.predict(samples, verbose=1)

    sample_x = wyckoff_designs[0]
    sample_x[sample_x < 0.1] = 0
    sample_sg = wyckoff_designs[1]
    sample_sg[sample_sg < 0.1] = 0

    # get sample elments
    Element = joblib.load('./data/element.pkl')
    E_v = np_utils.to_categorical(np.arange(0, len(Element), 1))
    sample_ele = []
    for i in range(num_ele):
        ele_v = np.argmax(sample_x[:, 0:len(E_v), i], axis=1)
        sample_ele.append(ele_v)
    sg_s = np.argmax(sample_sg, axis=1)

    # Load Parameter dic
    module_dir = 'data/wren_model_embedding/'
    with open(join(module_dir, "wyckoff-position-multiplicities.json")) as file:
        # dictionary mapping Wyckoff letters in a given space group to their multiplicity
        wyckoff_multiplicity_dict = json.load(file)
    with open(join(module_dir, "wyckoff-position-params.json")) as file:
        param_dict = json.load(file)

    # create dataframe for sampled data
    df_sample = pd.DataFrame(
        columns=['reconstructed_formula', 'reconstructed_ratio', 'reconstructed_wyckoff', 'reconstructed_sg',
                 'predicted_property', 'reconstructed_DoF', 'str_wyckoff'])

    for i in tqdm(range(sample_x.shape[0])):
        ### calculate reconstructed ratio and DoF
        recon_ratio = np.matmul(np.reshape(wyckoff_multiplicity_array[sg_s[i] + 1], (1, 26)),
                                np.round(sample_x[i][198:], decimals=0))
        recon_ratio = np.reshape(recon_ratio, (3,))

        recon_DoF = np.matmul(np.reshape(wyckoff_DoF_array[sg_s[i] + 1], (1, 26)),
                              np.round(sample_x[i][198:], decimals=0))
        recon_DoF = np.sum(np.reshape(recon_DoF, (3,)))

        ### reconstruct formula
        formula = ''
        elements = []
        for j in range(3):
            formula += Element[sample_ele[j][i]]
            elements.append(Element[sample_ele[j][i]])
            formula += str(int(recon_ratio[j]))
        recon_sg = sg_s[i] + 1

        ### check if compound is valid
        if np.any(recon_ratio == 0):
            continue
        recon_wyckoff_dic = get_reconstructed_wyckoff(np.round(sample_x[i, 198:, :]), elements, sg_s[i] + 1)
        all_wyckoff = list(itertools.chain(recon_wyckoff_dic[elements[0]], recon_wyckoff_dic[elements[1]],
                                           recon_wyckoff_dic[elements[2]]))
        common_wyckoff = [l for l in all_wyckoff if all_wyckoff.count(l) > 1]

        DoF_check = True
        for k in common_wyckoff:
            if param_dict[str(recon_sg)][re.sub(r'[^a-zA-Z]', '', k)] == 0:
                DoF_check = False
                break
        if DoF_check == False:
            continue

        ### add info to dataframe
        df_sample.loc[i, 'reconstructed_sg'] = recon_sg
        df_sample.loc[i, 'reconstructed_formula'] = formula
        df_sample.at[i, 'reconstructed_ratio'] = recon_ratio.tolist()
        df_sample['reconstructed_DoF'].at[i] = recon_DoF
        df_sample['reconstructed_wyckoff'].at[i] = recon_wyckoff_dic
        df_sample['str_wyckoff'].at[i] = str(df_sample.loc[i, 'reconstructed_wyckoff'])

    print('---------Saving Result---------------')
    df_sample['predicted_property'] = get_reconstructed_property(df_sample, regression, scaler_y)
    df_sample['predicted_SC'] = get_reconstructed_SC(df_sample)
    df_sample1 = df_sample[
        (df_sample['predicted_property'] < -3.36) & (df_sample['reconstructed_formula'] == 'Ca4Ti4O12')
        & (df_sample['reconstructed_DoF'] < 10)].groupby(by='str_wyckoff', group_keys=False).first()
    print('Saving total {} Wyckoff_Genes'.format(df_sample1.shape[0]))
    df_sample1.to_csv('temp_files/sampled_Wyckoff_Genes.csv')
    print('Generated Wyckoff_Genes saved to temp_files/sampled_Wyckoff_Genes.csv')
    print('---------End---------------')


if __name__ == '__main__':
    main()