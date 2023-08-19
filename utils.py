# -*- coding: utf-8 -*-
"""
Utility functions for WyCryst Framwork
"""

from featurizer import *
from data import *
import tensorflow as tf
import numpy as np
import pandas as pd
from os.path import abspath, dirname, join
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.preprocessing import MinMaxScaler
import joblib
import re
import json




#pad the  data along the second dimension
def pad (vae_x , p):
    dum_x = np.zeros((vae_x.shape[0],vae_x.shape[1]+p,vae_x.shape[2] ))
    dum_x[:,:-1*p,:] = vae_x
    return dum_x


#perform data normalizaiton
def minmax (X):
    
    scaler_x = MinMaxScaler()

    dim0 = X.shape[0]
    dim1 = X.shape[1]
    dim2 = X.shape[2]

    X1 = np.transpose(X,(1,0,2))
    X1 =X1.reshape(dim1,dim0*dim2)
    X1 = scaler_x.fit_transform(X1.T)
    X1 = X1.T
    X1 =X1.reshape(dim1,dim0,dim2)
    X1= np.transpose(X1,(1,0,2))
    return X1, scaler_x


def inv_minmax (X,scaler_x):

    dim0 = X.shape[0]
    dim1 = X.shape[1]
    dim2 = X.shape[2]

    X1 = np.transpose(X,(1,0,2))
    X1 =X1.reshape(dim1,dim0*dim2)
    X1 = scaler_x.inverse_transform(X1.T)
    X1 = X1.T
    X1 =X1.reshape(dim1,dim0,dim2)
    X1= np.transpose(X1,(1,0,2))
    return X1


def MAE(y_true, y_pred):
    y_true, y_pred = np.array(y_true+1e-12), np.array(y_pred+1e-12)
    return np.mean(np.abs((y_true - y_pred)))


def get_reconstructed_wyckoff(wyckoff_array, elements, sg):
    module_dir = './data/wren_model_embedding/'
    with open(join(module_dir, "wyckoff-position-multiplicities.json")) as file:
        wyckoff_multiplicity_dict = json.load(file)
    reconstructed_wyckoff_dic = {}
    for i in range(3):
        array = wyckoff_array[:, i]
        lst = []
        for j in range(26):
            if array[j] > 0:
                letter = chr(ord('a') + j)
                try:
                    lst.extend([wyckoff_multiplicity_dict[str(sg)][letter] + letter for i in range(int(array[j]))])
                except:
                    pass
        reconstructed_wyckoff_dic[elements[i]] = lst

    return reconstructed_wyckoff_dic



def get_reconstructed_SC(df_sample):
    # load CV_SC models
    models=[]
    for index in range(5):
        models.append(tf.keras.models.load_model('temp_files/forward_models/SC_final/cv_{}'.format(index+1)))

    # get crystal features and calculate CV_SC
    test_Crystal,test_sg = recon_wyckoff_represent(df_sample.reset_index(),3,20)
    test_Crystal = np.stack(test_Crystal,axis=0)
    test_sg = np.stack(test_sg,axis=0)[:,:,0]

    cv_prediction = []
    for i in models:
        cv_prediction.append(i.predict([test_Crystal,test_sg]))
        cv_mean = np.mean(cv_prediction)
#     print('prediction:',cv_prediction,', mean:',cv_mean)
    return np.mean(np.dstack(cv_prediction),axis=-1)


def get_wyckoff_dic(cif):
    #     crystal = Structure.from_str(cif,fmt="cif")
    crystal = cif
    data = SpacegroupAnalyzer(crystal)
    crystal_ = data.get_conventional_standard_structure()
    data = SpacegroupAnalyzer(crystal_)

    s_data = data.get_symmetrized_structure()
    wyckoff_dic = {}
    for i, sites in enumerate(s_data.equivalent_sites):
        species = sites[0].species_string
        if species in wyckoff_dic.keys():
            wyckoff_dic[species].append(s_data.wyckoff_symbols[i])
        else:
            wyckoff_dic[species] = [s_data.wyckoff_symbols[i]]

    return wyckoff_dic


def get_reconstructed_property(df_sample,regression,scaler_y):
    # get crystal representations and sg array for sample data
    Crystal, sg = recon_wyckoff_represent(df_sample.reset_index(), 3, 20)
    a = np.stack(Crystal, axis=0)
    X_recon = a

    X2_recon = np.stack(sg, axis=0)
    X2_recon = X2_recon[:, :, 0]
    #     print(X_recon.shape,X2_recon.shape)
    recon_pro = regression.predict([X_recon, X2_recon])
    recon_pro = scaler_y.inverse_transform(recon_pro)

    return recon_pro


def recon_wyckoff_represent(df, num_ele=3, num_sites=20):
    Element = joblib.load('./data/element.pkl')

    df1 = pd.read_csv('data/atomic_features.csv')

    E_v = np_utils.to_categorical(np.arange(0, len(Element), 1))

    #     print('number of element:',num_ele, 'number of sites:',num_sites)

    elem_embedding_file = './data/atom_init.json'
    with open(elem_embedding_file) as f:
        elem_embedding = json.load(f)
    elem_embedding = {int(key): value for key, value
                      in elem_embedding.items()}
    feat_cgcnn = []

    for key, value in elem_embedding.items():
        feat_cgcnn.append(value)

    feat_cgcnn = np.array(feat_cgcnn)

    # start featurization

    wyckoff = []
    sg = []
    test = []

    for x in range(len(df)):

        crystal = Composition(df['reconstructed_formula'][x])
        size = len(crystal.elements)

        # atomic featurizer
        z_u = np.array([crystal.elements[i].number for i in range(size)])
        if len(z_u) == 0:
            z_u = np.array([1, 1, 1])

        onehot = np.zeros((num_ele, len(E_v)))
        onehot[:len(z_u), :] = E_v[z_u - 1, :]

        coeffs_crsytal = np.zeros((num_ele, feat_cgcnn.shape[1]))
        for i in range(len(z_u)):
            try:
                coeffs_crsytal[i, :] = feat_cgcnn[z_u[i] - 1, :]
            except:
                coeffs_crsytal[i, :] = feat_cgcnn[0, :]

        dic = crystal.get_el_amt_dict()
        ratio_ = np.array([dic[crystal.elements[i].symbol] for i in range(size)])
        ratio_ /= crystal._natoms
        ratio = np.zeros((num_ele, 1))
        try:
            ratio[:len(z_u), 0] = ratio_
        except:
            #             print(df['reconstructed_formula'][x])
            pass
        ratio = ratio.reshape(1, num_ele)
        ratio1 = ratio * crystal._natoms

        # Ont-hot crystal system featurizer
        sg_cat = np.zeros((1, 230))
        sg_cat[0][int(df['reconstructed_sg'][x] - 1)] = 1
        sg_cat = sg_cat.T

        #         crystal_cat_list = np.concatenate((crystal_cat,crystal_cat,crystal_cat),axis=1)
        sg_cat_list = sg_cat

        # wyckoff featurizer

        wyckoff_ = np.zeros((num_ele, 52))
        wyckoff_dic = df['reconstructed_wyckoff'][x]
        #         print(wyckoff_dic)

        for i, element in enumerate(crystal.elements):

            for j in wyckoff_dic[str(element)]:
                #                 print(j)
                #                 print(j[0],j[:-1])
                site_num = ord(re.sub('[^a-zA-Z]+', '', j)) - 97
                wyckoff_[i, site_num] += 1
                wyckoff_[i, site_num + 26] = j[:-1]
                test.append(site_num)

        wyckoff_list = wyckoff_.T
        cell_ratio = np.sum(wyckoff_list[:26, :] * wyckoff_list[26:, :], axis=0)
        wyckoff_list = wyckoff_[:, :26].T

        #         print(wyckoff_)

        # atomic represeatnion
        atom_list = np.concatenate(
            ((onehot.T).T, cell_ratio.reshape(1, 3).T, ratio.T, np.zeros((num_ele, 1)), (coeffs_crsytal.T).T), axis=1)
        atom_list = atom_list.T

        #         print(atom_list.shape,crystal_cat_list.shape,wyckoff_list.shape)
        wyckoff.append(np.concatenate((atom_list, wyckoff_list), axis=0))
        sg.append(sg_cat_list)

    return wyckoff, sg