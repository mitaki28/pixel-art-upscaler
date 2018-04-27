import keras
import keras.backend as K

def random_normal():
    return keras.initializers.RandomNormal(stddev=0.02)

def batchnorm(h):
    return keras.layers.normalization.BatchNormalization(
        momentum=0.9,
        epsilon=2e-05,
    )(h)

def leaky_relu(h):
    return keras.layers.advanced_activations.LeakyReLU(
        alpha=0.2,
    )(h)

def weight_decay():
    return keras.regularizers.l2(0.00001)

def down_cbr(h, filters):
    h = keras.layers.Conv2D(
        filters=filters,
        kernel_size=4,
        strides=2,
        padding='same',
        kernel_initializer=random_normal(),
        kernel_regularizer=weight_decay(),
    )(h)
    h = batchnorm(h)
    h = leaky_relu(h)
    return h

def up_cbr(h, filters, dropout=False, use_resize_conv=False):
    if use_resize_conv:
        h = keras.layers.UpSampling2D(
            size=(2, 2),
        )(h)
        h = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=random_normal(),
            kernel_regularizer=weight_decay(),               
        )(h)
    else:
        h = keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=4,
            strides=2,
            padding='same',
            kernel_initializer=random_normal(),
            kernel_regularizer=weight_decay(),
        )(h)
    h = batchnorm(h)
    if dropout:
        h = keras.layers.Dropout(0.5)(h)
    h = keras.layers.core.Activation('relu')(h)
    return h

def generator(w, in_ch, out_ch, base_ch, use_resize_conv=False):
    x = keras.layers.Input(shape=(w, w, in_ch))

    h = keras.layers.Conv2D(
        filters=base_ch,
        kernel_size=5,
        strides=1,
        padding='same',
        kernel_initializer=random_normal(),     
        kernel_regularizer=weight_decay(),
    )(x)
    
    h0 = leaky_relu(h)
    h = keras.layers.Conv2D(
        filters=base_ch * 2,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=random_normal(),
        kernel_regularizer=weight_decay(),           
    )(h0)
    h = batchnorm(h)
    h1 = leaky_relu(h)

    h2 = down_cbr(h1, base_ch * 4)
    h3 = down_cbr(h2, base_ch * 8)
    h4 = down_cbr(h3, base_ch * 8)
    h5 = down_cbr(h4, base_ch * 8)
    h6 = down_cbr(h5, base_ch * 8)
    h7 = down_cbr(h6, base_ch * 8)

    h = up_cbr(h7, base_ch * 8, dropout=True, use_resize_conv=use_resize_conv)

    h = keras.layers.concatenate([h, h6])
    h = up_cbr(h, base_ch * 8, dropout=True, use_resize_conv=use_resize_conv)

    h = keras.layers.concatenate([h, h5])
    h = up_cbr(h, base_ch * 8, dropout=True, use_resize_conv=use_resize_conv)

    h = keras.layers.concatenate([h, h4])
    h = up_cbr(h, base_ch * 8, dropout=False, use_resize_conv=use_resize_conv)

    h = keras.layers.concatenate([h, h3])
    h = up_cbr(h, base_ch * 4, dropout=False, use_resize_conv=use_resize_conv)

    h = keras.layers.concatenate([h, h2])
    h = up_cbr(h, base_ch * 2, dropout=False, use_resize_conv=use_resize_conv)

    h = keras.layers.concatenate([h, h1])
    h = keras.layers.Conv2D(
        filters=base_ch,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=random_normal(),
        kernel_regularizer=weight_decay(),        
    )(h)
    h = batchnorm(h)
    h = keras.layers.Activation('relu')(h)

    h = keras.layers.concatenate([h, h0])
    h = keras.layers.Conv2D(
        filters=out_ch,
        kernel_size=5,
        strides=1,
        padding='same',
        kernel_initializer=random_normal(),
        kernel_regularizer=weight_decay(),
        name='output_gen',               
    )(h)
    return x, h

def discriminator(w, ch0, ch1, base_ch):
    assert base_ch % 2 == 0

    x0 = keras.layers.Input(shape=(w, w, ch0))
    x1 = keras.layers.Input(shape=(w, w, ch1))
    h0 = keras.layers.Conv2D(
        filters=base_ch // 2,
        kernel_size=5,
        strides=1,
        padding='same',
        kernel_initializer=random_normal(),
        kernel_regularizer=weight_decay(),        
    )(x0)
    h0 = batchnorm(h0)
    h0 = leaky_relu(h0)

    h1 = keras.layers.Conv2D(
        filters=base_ch // 2,
        kernel_size=5,
        strides=1,
        padding='same',
        kernel_initializer=random_normal(),
        kernel_regularizer=weight_decay(),                
    )(x1)
    h1 = batchnorm(h1)
    h1 = leaky_relu(h1)

    h = down_cbr(keras.layers.concatenate([h0, h1]), base_ch * 2)
    h = down_cbr(h, base_ch * 4)
    h = down_cbr(h, base_ch * 8)
    h = keras.layers.Conv2D(
        filters=1,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=random_normal(),
        kernel_regularizer=weight_decay(),  
    )(h)
    return x0, x1, h

def pix2pix(w, in_ch, out_ch, base_ch, use_resize_conv=False):
    gen_in, gen_out = generator(w, in_ch, out_ch, base_ch, use_resize_conv)
    dis_in_0, dis_in_1, dis_out = discriminator(w, in_ch, out_ch, base_ch)

    gen = keras.models.Model(gen_in, gen_out, 'Generator')
    gen_frozen = keras.models.Model(gen_in, gen_out, 'Generator-Frozen')
    gen_frozen.trainable = False

    dis = keras.models.Model([dis_in_0, dis_in_1], dis_out, 'Discriminator')
    dis_frozen = keras.models.Model([dis_in_0, dis_in_1], dis_out, 'Discriminator-Frozen')
    dis_frozen.trainable = False

    x_in = keras.layers.Input(shape=(w, w, in_ch))
    x_real = keras.layers.Input(shape=(w, w, out_ch))

    x_fake_gen = gen(x_in)
    y_fake_gen = dis_frozen([x_in, x_fake_gen])

    x_fake = gen_frozen(x_in)
    y_real_dis = dis([x_in, x_real])
    y_fake_dis = dis([x_in, x_fake])

    gen_trainer = keras.models.Model([x_in, x_real], [x_fake_gen, y_fake_gen])
    dis_trainer = keras.models.Model([x_in, x_real], [y_real_dis, y_fake_dis])

    return gen, dis, gen_trainer, dis_trainer

lam1 = 100
lam2 = 1

def gen_loss_l1(y_true, y_pred):
    return lam1 * keras.losses.mean_absolute_error(y_true, y_pred)

def gen_loss_adv(_, y_pred):
    return lam2 * K.mean(K.softplus(y_pred), axis=-1)

def dis_loss_real(_, y_pred):
    return K.mean(K.softplus(y_pred), axis=-1)

def dis_loss_fake(_, y_pred):
    return K.mean(K.softplus(-y_pred), axis=-1)

