# -*- coding: utf-8 -*-
"""
PVAE model for WyCryst Framwork
"""
from data import *
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense, Lambda,Conv1D,Conv1DTranspose,Conv2DTranspose, LeakyReLU,Activation,Flatten,Reshape, BatchNormalization, concatenate


def PVAE(X_train, y_train, sup_dim=1):

    # Define dimensions of the neural network
    input_dim = X_train.shape[1]
    channel_dim = X_train.shape[2]
    regression_dim = y_train.shape[1]-1
    sg_dim = 230
    latent_dim = 256
    max_filter = 128
    strides = [2,2,1]
    kernel = [5,3,3]
    regularization_coefficient = 0.01
    coeff_recon = 1/64
    coeff_KL = 4
    coeff_prop = 32
    coeff_sg = 1/6
    coeff_formula=1/256

    # Load Wyckoff Data
    wyckoff_multiplicity_array, wyckoff_DoF_array = wyckoff_para_loader()

    K.clear_session()
    x = Input(shape=(input_dim, channel_dim,))
    x2 = Input(shape=(sg_dim,), name="sg_number")
    regression_inputs = Input(shape=(regression_dim,))

    # Encoder crystal information into latent space
    en0 = Conv1D(max_filter // 4, kernel[0], strides=strides[0], padding='SAME')(x)
    en0 = BatchNormalization()(en0)
    en0 = LeakyReLU(0.2)(en0)
    en1 = Conv1D(max_filter // 2, kernel[1], strides=strides[1], padding='SAME')(en0)
    en1 = BatchNormalization()(en1)
    en1 = LeakyReLU(0.2)(en1)
    en2 = Conv1D(max_filter, kernel[2], strides=strides[2], padding='SAME')(en1)
    en2 = BatchNormalization()(en2)
    en2 = LeakyReLU(0.2)(en2)
    en3 = Flatten()(en2)
    en4 = Dense(1024, activation='relu')(en3)

    sg_latent = Dense(256, activation='relu')(x2)
    sg_latent = Dense(128, activation='relu')(sg_latent)
    latent = concatenate([en4, sg_latent])

    z_mean = Dense(latent_dim, activation='linear')(latent)
    z_log_var = Dense(latent_dim, activation='linear')(latent)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # Reparameterization
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder = Model([x, x2], z)

    # Linear model from latent space to desired property
    re0 = Activation('relu')(z_mean)
    re1 = Dense(128, activation="relu", kernel_regularizer='l2')(re0)
    re1 = Dense(32, activation="relu", kernel_regularizer='l2')(re1)
    y_predict_sup = Dense(sup_dim, activation='sigmoid', kernel_regularizer='l2')(re1)

    regression = Model([x, x2], y_predict_sup, name='forward_model')

    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    map_size = K.int_shape(encoder.layers[-11].output)[1]

    # Decoded sg
    decoded_sg = Dense(230, activation='relu')(latent_inputs)
    decoded_sg = Activation('softmax')(decoded_sg)


    # Reconstructed output
    de1 = Dense(max_filter * map_size, activation='relu')(latent_inputs)
    de1 = Reshape((map_size, max_filter))(de1)
    de1 = BatchNormalization()(de1)
    de2 = Conv1DTranspose(max_filter // 2, (kernel[2]), strides=(strides[2]),
                          padding='SAME')(de1)
    de2 = BatchNormalization()(de2)
    de2 = Activation('relu')(de2)
    de3 = Conv1DTranspose(max_filter // 4, (kernel[1]), strides=(strides[1]),
                          padding='SAME')(de2)
    de3 = BatchNormalization()(de3)
    de3 = Activation('relu')(de3)
    de4 = Conv1DTranspose(channel_dim, (kernel[0]), strides=(strides[0]),
                          padding='SAME')(de3)
    decoder_outputs = Activation('linear')(de4)
    decoder = Model(latent_inputs, [decoder_outputs, decoded_sg])

    [reconstructed_outputs, reconstructed_sg] = decoder(z)

    VAE = Model(inputs=[x, x2, regression_inputs], outputs=[reconstructed_outputs, reconstructed_sg])

    # VAE loss for fitting

    # formula check loss
    wp_mul_tensor = tf.constant(wyckoff_multiplicity_array)
    indices = tf.argmax(reconstructed_sg[:, :], axis=1)
    max_indices = tf.expand_dims(indices, axis=1)

    gathered_row = tf.gather(wp_mul_tensor, max_indices)
    gathered_row = tf.cast(gathered_row, tf.float32)

    reconstructed_wyckoff = reconstructed_outputs[:, 198:]
    reconstructed_wyckoff = tf.round(reconstructed_wyckoff + 0.0001)
    wyckoff_formula = tf.linalg.matmul(gathered_row, reconstructed_wyckoff)

    loss_recon = coeff_recon * K.sum(K.square(x[:, :, :] - reconstructed_outputs[:, :, :]))
    loss_sg = coeff_sg * K.sum(tf.keras.losses.categorical_crossentropy(x2, reconstructed_sg))
    loss_KL = coeff_KL * K.mean(
        - 0.5 * 1 / latent_dim * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
    loss_formula = coeff_formula * K.sum(K.square(x[:, 103:104, :] - wyckoff_formula[:, :, :]))
    loss_prop = coeff_prop * K.sum(K.square(regression_inputs[:, :sup_dim] - y_predict_sup[:, :sup_dim]))

    vae_loss = loss_recon + loss_KL + loss_prop + loss_sg + loss_formula

    VAE.summary()
    return VAE, encoder, decoder, regression, vae_loss, loss_recon, loss_sg, loss_KL, loss_prop, loss_formula