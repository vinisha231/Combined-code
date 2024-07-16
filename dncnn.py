def DnCNN(depth=17, filters=64, image_channels=1, use_bnorm=True):
    input_layer = Input(shape=(None, None, image_channels), name='input')
    x = layers.Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu')(input_layer)

    for i in range(depth - 2):
        x = layers.Conv2D(filters=filters, kernel_size=3, padding='same')(x)
        if use_bnorm:
            x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)

    output_layer = layers.Conv2D(filters=image_channels, kernel_size=3, padding='same')(x)
    output_layer = Subtract()([input_layer, output_layer])
    model = models.Model(inputs=input_layer, outputs=output_layer, name='DnCNN')
    return model

model = DnCNN(depth=17, filters=64, image_channels=1, use_bnorm=True)
