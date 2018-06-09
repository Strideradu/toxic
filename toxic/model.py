from keras.layers import Dense, Embedding, Input, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, \
    GlobalMaxPooling1D, Convolution1D, Conv2D, MaxPool2D, GlobalMaxPool2D, GlobalAvgPool2D
from keras.layers import Bidirectional, Dropout, CuDNNGRU, Flatten, SpatialDropout1D, BatchNormalization, Reshape, CuDNNLSTM
from keras.models import Model
from keras.optimizers import RMSprop, Nadam, Adagrad
from keras.layers.merge import Concatenate, Add

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras import regularizers

def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim


def get_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(x)
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['accuracy'])

    return model


def get_GRU_pool_con(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x1 = MaxPooling1D(4)(x)
    x2 = AveragePooling1D(4)(x)
    x = Concatenate(axis=1)([x1, x2])
    x = Flatten()(x)
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['accuracy'])

    return model


def get_GRU_GlobalMaxAve(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    # x = SpatialDropout1D(rate = dropout_rate)(x)
    # x = Dropout(0.2)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    # x = SpatialDropout1D(rate=dropout_rate)(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x = Concatenate(axis=1)([x1, x2])
    x = Dense(dense_size, activation="relu")(x)
    # x = Dropout(dropout_rate)(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=0.001),
                  metrics=['accuracy'])

    return model


def get_GRU_GlobalMax_Ave(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(rate=0.3)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    # x = Dropout(dropout_rate)(x)
    x1 = MaxPooling1D(128)(x)
    x2 = AveragePooling1D()(x)
    x = Concatenate(axis=1)([x1, x2])
    x = Flatten()(x)
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['accuracy'])

    return model


def get_GRU_GlobalMaxAveEach(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    # x = Dropout(dropout_rate)(x)
    x3 = GlobalMaxPooling1D()(x)
    x4 = GlobalAveragePooling1D()(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    # x = Dropout(dropout_rate)(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x = Concatenate(axis=1)([x1, x2, x3, x4])
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['accuracy'])

    return model


def get_GRU_GlobalMaxAveNoDense(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(rate = dropout_rate)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    # x = SpatialDropout1D(rate = dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    # x = SpatialDropout1D(rate=dropout_rate)(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x = Concatenate(axis=1)([x1, x2])
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=0.001),
                  metrics=['accuracy'])

    return model


def get_SingleGRU_GlobalMaxAve(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.0001),
                               recurrent_regularizer=regularizers.l2(0.0001),
                               bias_regularizer=regularizers.l2(0.0001)
                               ))(x)
    x = SpatialDropout1D(rate=dropout_rate)(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x = Concatenate(axis=1)([x1, x2])
    x = Dense(dense_size, activation="relu",
              kernel_regularizer=regularizers.l2(0.0001),
              bias_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=0.001),
                  metrics=['accuracy'])

    return model


def get_TextCNN(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size, filter_sizes=[2, 3, 4, 5, 7, 14],
                num_filters=128):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    z = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        z = Dropout(dropout_rate)(z)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = Dense(dense_size, activation="relu")(z)
    z = Dropout(dropout_rate)(z)
    output_layer = Dense(6, activation="sigmoid")(z)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(),
                  metrics=['accuracy'])

    return model


def get_TextCNN_GlobalPooling(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size,
                              filter_sizes=[2, 3, 4], num_filters=64):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    z = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv1 = GlobalMaxPooling1D()(conv)
        conv2 = GlobalAveragePooling1D()(conv)
        conv_blocks.append(conv1)
        conv_blocks.append(conv2)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = Dropout(dropout_rate)(z)
    z = Dense(dense_size, activation="relu")(z)
    output_layer = Dense(6, activation="sigmoid")(z)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=0.001),
                  metrics=['accuracy'])

    return model


def get_DPCNN(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size,
              filter_sizes=3, repeat=4):
    input_layer = Input(shape=(sequence_length,))
    num_filters = embedding_matrix.shape[1]
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    z = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    conv = Convolution1D(filters=num_filters,
                         kernel_size=filter_sizes,
                         padding="same",
                         activation="relu",
                         strides=1,
                         kernel_regularizer=regularizers.l2(0.0001),
                         activity_regularizer=regularizers.l2(0.0001),
                         bias_regularizer=regularizers.l2(0.0001)
                         )(z)

    conv = Convolution1D(filters=num_filters,
                         kernel_size=filter_sizes,
                         padding="same",
                         activation="relu",
                         strides=1,
                         kernel_regularizer=regularizers.l2(0.0001),
                         activity_regularizer=regularizers.l2(0.0001),
                         bias_regularizer=regularizers.l2(0.0001)
                         )(conv)
    # conv = Dropout(dropout_rate)(conv)
    conv = Add()([conv, z])
    for i in range(repeat):
        pool = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)
        # conv = SpatialDropout1D(rate=dropout_rate)(pool)
        conv = Convolution1D(filters=num_filters,
                             kernel_size=filter_sizes,
                             padding="same",
                             activation="relu",
                             strides=1,
                             kernel_regularizer=regularizers.l2(0.0001),
                             activity_regularizer=regularizers.l2(0.0001),
                             bias_regularizer=regularizers.l2(0.0001)
                             )(pool)

        conv = Convolution1D(filters=num_filters,
                             kernel_size=filter_sizes,
                             padding="same",
                             activation="relu",
                             strides=1,
                             kernel_regularizer=regularizers.l2(0.0001),
                             activity_regularizer=regularizers.l2(0.0001),
                             bias_regularizer=regularizers.l2(0.0001)
                             )(conv)
        # conv = Dropout(dropout_rate)(conv)
        conv = Add()([conv, pool])

    #z1 = GlobalMaxPooling1D()(conv)
    #z2 = GlobalAveragePooling1D()(conv)
    #z = Concatenate()([z1, z2])
    z = MaxPooling1D(3)(conv)
    z = Flatten()(z)
    z = Dense(dense_size, activation="relu", kernel_regularizer=regularizers.l2(0.0001),
                         activity_regularizer=regularizers.l2(0.0001),
                         bias_regularizer=regularizers.l2(0.0001))(z)
    # z = Dropout(0.5)(z)
    output_layer = Dense(6, activation="sigmoid", kernel_regularizer=regularizers.l2(0.0001),
                         activity_regularizer=regularizers.l2(0.0001),
                         bias_regularizer=regularizers.l2(0.0001))(z)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(),
                  metrics=['accuracy'])

    return model


def get_DPCNN_BN(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size,
                 filter_sizes=3, repeat=4):
    input_layer = Input(shape=(sequence_length,))
    num_filters = embedding_matrix.shape[1]
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    z = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    z = BatchNormalization()(z)
    conv = Convolution1D(filters=num_filters,
                         kernel_size=filter_sizes,
                         padding="same",
                         activation="relu",
                         strides=1)(z)

    conv = Convolution1D(filters=num_filters,
                         kernel_size=filter_sizes,
                         padding="same",
                         activation="relu",
                         strides=1)(conv)
    conv = Dropout(dropout_rate)(conv)
    conv = Add()([conv, z])
    for i in range(repeat):
        pool = MaxPooling1D(pool_size=3, strides=2)(conv)
        conv = SpatialDropout1D(rate=dropout_rate)(pool)
        conv = BatchNormalization()(conv)
        conv = Convolution1D(filters=num_filters,
                             kernel_size=filter_sizes,
                             padding="same",
                             activation="relu",
                             strides=1)(conv)

        conv = Convolution1D(filters=num_filters,
                             kernel_size=filter_sizes,
                             padding="same",
                             activation="relu",
                             strides=1)(conv)
        # conv = Dropout(dropout_rate)(conv)
        conv = Add()([conv, pool])

    z1 = GlobalMaxPooling1D()(conv)
    z2 = GlobalAveragePooling1D()(conv)
    z = Concatenate()([z1, z2])
    z = Dense(dense_size, activation="linear")(z)
    z = Dropout(0.5)(z)
    output_layer = Dense(6, activation="sigmoid")(z)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=0.001),
                  metrics=['accuracy'])

    return model


def get_2DTextCNN(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size,
                  filter_sizes=[1, 2, 3, 4, 5], num_filters=32):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    z = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    z = Reshape((sequence_length, embedding_matrix.shape[1], 1))(z)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv2D(num_filters, kernel_size=(sz, embedding_matrix.shape[1]), kernel_initializer='normal',
                      activation='elu')(z)
        conv1 = GlobalMaxPool2D()(conv)
        conv2 = GlobalAvgPool2D()(conv)
        conv_blocks.append(conv1)
        conv_blocks.append(conv2)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    # z = Flatten()(z)
    z = Dropout(dropout_rate)(z)
    z = Dense(dense_size, activation="relu")(z)
    output_layer = Dense(6, activation="sigmoid")(z)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=0.001),
                  metrics=['accuracy'])

    return model


def get_smallDPCNN(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size,
              filter_sizes=3, repeat=3):
    input_layer = Input(shape=(sequence_length,))
    num_filters = 64
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    z = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    conv1 = Convolution1D(filters=num_filters,
                         kernel_size=filter_sizes,
                         padding="same",
                         activation="relu",
                         strides=1)(z)

    conv2 = Convolution1D(filters=num_filters,
                         kernel_size=filter_sizes,
                         padding="same",
                         activation="relu",
                         strides=1)(conv1)
    conv2 = Dropout(dropout_rate)(conv2)
    conv = Add()([conv1, conv2])
    for i in range(repeat):
        pool = MaxPooling1D(pool_size=3, strides=2)(conv)
        conv = SpatialDropout1D(rate=dropout_rate)(pool)
        conv = Convolution1D(filters=num_filters,
                             kernel_size=filter_sizes,
                             padding="same",
                             activation="relu",
                             strides=1)(conv)

        # conv = Dropout(dropout_rate)(conv)
        conv = Add()([conv, pool])

    z1 = GlobalMaxPooling1D()(conv)
    z2 = GlobalAveragePooling1D()(conv)
    z = Concatenate()([z1, z2])
    z = Dense(dense_size, activation="relu")(z)
    z = Dropout(0.5)(z)
    output_layer = Dense(6, activation="sigmoid")(z)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=0.001),
                  metrics=['accuracy'])

    return model

def get_GRU_Capsnet(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    Routings = 5
    Num_capsule = 10
    Dim_capsule = 16
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = SpatialDropout1D(rate = dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = SpatialDropout1D(rate=dropout_rate)(x)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # x = SpatialDropout1D(rate=dropout_rate)(x)
    x = Flatten()(capsule)
    # x = Dropout(0.5)(capsule)
    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(),
                  metrics=['accuracy'])

    return model

def get_GRU_CapsnetPooling(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    Routings = 5
    Num_capsule = 10
    Dim_capsule = 16
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = SpatialDropout1D(rate = dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = SpatialDropout1D(rate=dropout_rate)(x)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # x = SpatialDropout1D(rate=dropout_rate)(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x = Flatten()(capsule)
    x = Concatenate(axis=1)([x1, x2, x])

    # x = Dropout(0.5)(capsule)
    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(),
                  metrics=['accuracy'])

    return model

def get_GRU_CapsnetAttnPooling(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    Routings = 5
    Num_capsule = 10
    Dim_capsule = 16
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    # x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = SpatialDropout1D(rate = dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = SpatialDropout1D(rate=dropout_rate)(x)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # x = SpatialDropout1D(rate=dropout_rate)(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x3 = Attention(sequence_length)(x)
    x = Flatten()(capsule)
    x = Concatenate(axis=1)([x1, x2, x3, x])

    # x = Dropout(0.5)(capsule)
    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(),
                  metrics=['accuracy'])

    return model

def get_GRU_Attn(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Attention(sequence_length)(x)

    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(),
                  metrics=['accuracy'])

    return model

def get_GRU_AttnPooling(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    x = Attention(sequence_length)(x)
    x = Concatenate(axis=1)([x1, x2, x])
    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(),
                  metrics=['accuracy'])

    return model

def get_PareGRU_GlobalMaxAve(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    embedding = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)

    y = Bidirectional(CuDNNGRU(2 * recurrent_units, return_sequences=True))(embedding)
    y = Bidirectional(CuDNNGRU(2 * recurrent_units, return_sequences=True))(y)
    # x = SpatialDropout1D(rate=dropout_rate)(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)
    y1 = GlobalMaxPooling1D()(y)
    y2 = GlobalAveragePooling1D()(y)
    x = Concatenate(axis=1)([x1, x2, y1, y2])
    x = Dense(dense_size, activation="relu")(x)
    # x = Dropout(dropout_rate)(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(lr=0.001),
                  metrics=['accuracy'])

    return model


def get_LSTMGRU_GlobalMaxAve(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    embedding = SpatialDropout1D(rate=dropout_rate)(embedding_layer)
    x = Bidirectional(CuDNNLSTM(recurrent_units, return_sequences=True))(embedding)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)

    # x = SpatialDropout1D(rate=dropout_rate)(x)
    x1 = GlobalMaxPooling1D()(x)
    x2 = GlobalAveragePooling1D()(x)

    x = Concatenate(axis=1)([x1, x2])

    # x = Dropout(dropout_rate)(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=Nadam(),
                  metrics=['accuracy'])

    return model