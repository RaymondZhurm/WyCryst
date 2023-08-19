# -*- coding: utf-8 -*-
"""
Featurizer for WyCryst Framwork
"""

from pymatgen.core import Composition
import pandas as pd
from keras.utils import np_utils
from utils import *
import numpy as np
import joblib
import re
import json



num_ele=3
num_sites = 20


def wyckoff_represent(df, num_ele=3, num_sites=20):
    Element = joblib.load('./data/element.pkl')

    df1 = pd.read_csv('./data/atomic_features.csv')

    E_v = np_utils.to_categorical(np.arange(0, len(Element), 1))

    # print('number of element:', num_ele, 'number of sites:', num_sites)

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

        crystal = Composition(df['pretty_formula'][x])
        size = len(crystal.elements)

        # atomic featurizer
        z_u = np.array([crystal.elements[i].number for i in range(size)])

        onehot = np.zeros((num_ele, len(E_v)))
        onehot[:len(z_u), :] = E_v[z_u - 1, :]

        coeffs_crsytal = np.zeros((num_ele, feat_cgcnn.shape[1]))
        for i in range(len(z_u)):
            coeffs_crsytal[i, :] = feat_cgcnn[z_u[i] - 1, :]

        dic = crystal.get_el_amt_dict()
        ratio_ = np.array([dic[crystal.elements[i].symbol] for i in range(size)])
        ratio_ /= crystal._natoms
        ratio = np.zeros((num_ele, 1))
        ratio[:len(z_u), 0] = ratio_
        ratio = ratio.reshape(1, num_ele)
        ratio1 = ratio * crystal._natoms

        # Ont-hot crystal system featurizer
        sg_cat = np.zeros((1, 230))
        sg_cat[0][df['spacegroup_number'][x] - 1] = 1
        sg_cat = sg_cat.T
        sg_cat_list = sg_cat

        # wyckoff featurizer

        wyckoff_ = np.zeros((num_ele, 52))
        wyckoff_dic = df['wyckoff_dic'][x]

        for i, element in enumerate(crystal.elements):

            for j in wyckoff_dic[str(element)]:

                site_num = ord(re.sub('[^a-zA-Z]+', '', j)) - 97
                wyckoff_[i, site_num] += 1
                wyckoff_[i, site_num + 26] = j[:-1]
                test.append(site_num)

        wyckoff_list = wyckoff_.T
        cell_ratio = np.sum(wyckoff_list[:26, :] * wyckoff_list[26:, :], axis=0)
        wyckoff_list = wyckoff_[:, :26].T


        # atomic represeatnion
        atom_list = np.concatenate(
            ((onehot.T).T, cell_ratio.reshape(1, 3).T, ratio.T, np.zeros((num_ele, 1)), (coeffs_crsytal.T).T), axis=1)
        atom_list = atom_list.T

        wyckoff.append(np.concatenate((atom_list, wyckoff_list), axis=0))
        sg.append(sg_cat_list)

    return wyckoff, sg



