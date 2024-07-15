#BEGIN DATA IMPLEMENTATION
    def load_flt_file(file_path, shape=(512, 512), add_channel=True): #Defines the flt load function with image size
        with open(file_path, 'rb') as file: #Opens the directory containing the flts
            data = np.fromfile(file, dtype=np.float32) #Stores the files as floats
            img = data.reshape(shape) #Reshapes the data to be used
            if add_channel: #Add channel set true so we do
                img = img[:, :, np.newaxis] #Adds a channel of depth to flt for use
        return img

    # Load and preprocess images
    def load_images_from_directory(directory, target_size=(512, 512)): #Function to load the images
        images = []                                                             #Creates empty array to store images
        for filename in os.listdir(directory): #For each file in the directory do
            if filename.endswith(".flt"): #If the file is an flt
                file_path = os.path.join(directory, filename) #Stores the name of the directory and file path to load
                img = load_flt_file(file_path, shape=target_size) #Loads the flts using previous function and stores as 512x512
                img = img / 255.0 #Normalizes to values between [0,1]
                images.append(img) #Adds the grabbed image to back of array
        return np.array(images)

    # Directories
    clean_dir = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/900views' #Sets the data directories
    dirty_dir = '/mmfs1/gscratch/uwb/CT_images/RECONS2024/60views'

    clean_images = load_images_from_directory(clean_dir) #Performs the load image function on data directories
    dirty_images = load_images_from_directory(dirty_dir)

    X_train, X_test, y_train, y_test = train_test_split(dirty_images, clean_images, test_size=0.2, random_state=42)
#END DATA IMPLEMENTATION



#BEGIN TRANSFORMER MODEL CODE
def create_patch_embeddings(inputs, patch_size=2, projection_dim=64): #Defines patch embedding function
        # Assuming inputs have shape (batch_size, height, width, channels)
        batch_size, height, width, channels = inputs.shape
        # Calculate the number of patches
        num_patches = (height // patch_size) * (width // patch_size)
        patches = layers.Conv2D(filters=projection_dim, kernel_size=patch_size, strides=patch_size)(inputs) #Convolves a filter that generates patches of the image
        # Reshape the output to have (num_patches, projection_dim)
        patches = layers.Reshape((num_patches, projection_dim))(patches)
              return patches

def positional_encoding(num_patches, projection_dim): #Def function for positional encoding the patches generated
        position = np.arange(num_patches)[:, np.newaxis] #Creates an array of the range up to num_patches, then makes it a 2d column vector
        div_term = np.exp(np.arange(0, projection_dim, 2) * -(np.log(10000.0) / projection_dim)) #This is the coeffient in the arg of the trig funcs based on pos
        pos_enc = np.zeros((num_patches, projection_dim)) #Creates an array of zeroes the same size as patches array
        pos_enc[:, 0::2] = np.sin(position * div_term) #Even indices are unique sine frequencies
        pos_enc[:, 1::2] = np.cos(position * div_term) #Odd indices are unique cos frequencies
        pos_enc = pos_enc[np.newaxis, ...]  # Shape: [1, num_patches, projection_dim]
        return tf.cast(pos_enc, dtype=tf.float32) #Casts as float for use in transformer

class TransformerBlock(layers.Layer):  #Creates the transformer block class
      def __init__(self, projection_dim, num_heads): #constructor  method with params of self, the dim of embedding space, and number of transformer heads
            super(TransformerBlock, self).__init__() #calls layers.layer parent class and init.
            self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim) #Attention layer
            self.ffn = tf.keras.Sequential([ #Dense forward projection/multilayer perceptron layer
                layers.Dense(projection_dim, activation="relu"),
                layers.Dense(projection_dim),
            ])
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6) #Normalizes perceptron outputs
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

      def call(self, inputs): #Defines method to add attention values to initial input vectors
            attn_output = self.attention(inputs, inputs) #creates a variable of the attention output
            out1 = self.layernorm1(inputs + attn_output) #adds to original and normalizes and stores as out1
            ffn_output = self.ffn(out1) #Runs this through feed forward
            return self.layernorm2(out1 + ffn_output) #Adds that output to output of attention layer

def build_transformer_model(num_patches, projection_dim, num_heads, transformer_layers, num_classes): #Function to build the transformer model
    inputs = layers.Input(shape=(512, 512, 1))  #Defines image shape as 512x512x1 tensor
    patches = create_patch_embeddings(inputs, patch_size=8, projection_dim=projection_dim) #Defines patches as patch embedding function called on images
    encoded_patches = patches + positional_encoding(num_patches, projection_dim)

    x = encoded_patches  #Variable storage of encoded patches
    for _ in range(transformer_layers): #For ever transformer layer
    x = TransformerBlock(projection_dim, num_heads)(x) #Iterate the transformer block on the patches

    # Use Conv2D and UpSampling2D layers to reconstruct the image
    x = layers.Reshape((64, 64, 64))(x) #Reshape as 64x64x64 tensor

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x) #Filter that image with 128 3x3 filters
    x = layers.UpSampling2D((2, 2))(x)  #Upsamples image to 128x128
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x) #Second filter pass
    x = layers.UpSampling2D((4, 4))(x)  #Gets us back to 512x512 image
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output layer to reconstruct the image

    return tf.keras.Model(inputs, outputs)  # Return the model object

num_patches = 4096 #Defines the number of patches used
projection_dim = 64 #This is the dimension of the embedding space of the patches (Where the patch vectors exist)
num_heads = 4 #Number of heads in the transformer
transformer_layers = 6 #How many transformer layers
num_classes = 4 #Not sure if this line is useful

model = build_transformer_model(num_patches, projection_dim, num_heads, transformer_layers, num_classes) #Builds the model with our attributes
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']) #Uses mse loss


# Train model
model.fit(X_train, y_train, validation_split=0.1, epochs=250, batch_size=16, verbose=1)

# Save model
model.save('transformer_model.keras')
#END TRANSFORMER MODEL CODE





#UNET MODEL CODE BEGIN
def ssim_loss(y_true, y_pred): #Defining a function of ssim loss image to image
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)) #return the loss value

def mse_loss(y_true, y_pred): #Defining a mean squared error loss
    return tf.reduce_mean(tf.square(y_true - y_pred)) #Returning loss

def combined_loss(y_true, y_pred, alpha = 0.2, beta = 0.8): #Define a mixed loss with proportions alpha and beta
    return alpha * ssim_loss(y_true, y_pred) + beta * mse_loss(y_true, y_pred) #Return the sum of the weighted losses =1

def unet_model(input_size=(512, 512, 3)): #Defining the model
        inputs = tf.keras.Input(input_size)
            # Downsample
        ...
        c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs) #Initial convolutional layer
        c1 = layers.Dropout(0.1)(c1) #Drops 10% of neurons from layer 1
        c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1) #Second convolution
        p1 = layers.MaxPooling2D((2, 2))(c1) #Max pools 2x2 regions

        c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1) #Second Same as first section but filters x 2
        c2 = layers.Dropout(0.1)(c2)
        c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2) #Third same but filters x 2
        c3 = layers.Dropout(0.1)(c3)
        c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3) #Fourth same but filters x 2
        c4 = layers.Dropout(0.1)(c4)
        c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4) #Fourth has no maxpool and filters x 2
        c5 = layers.Dropout(0.1)(c5)
        c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Upsample
        u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5) #Undoes previous convolve
        u6 = layers.concatenate([u6, c4], axis=3) #First skip connection
        c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6) #Filtering convolutional layer
        c6 = layers.Dropout(0.1)(c6) #Drop 10% neurons from layer 6
        c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6) #Second convolution and filter

        u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6) #Same as first but half filters
        u7 = layers.concatenate([u7, c3], axis=3) #Second skip connection
        c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = layers.Dropout(0.1)(c7)
        c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7) #Same but half filters
        u8 = layers.concatenate([u8, c2], axis=3)
        c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = layers.Dropout(0.1)(c8)
        c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8) #Same but half filters
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = layers.Dropout(0.1)(c9)
        c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)


        outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c9) #Final 3 filter 1x1 kernel with sigmoid activation
        model = models.Model(inputs=inputs, outputs=[outputs]) #Compresses the function into a single variable "model"

        return model #Returns that variable

#defines model as the unet model
model = unet_model()
model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy']) #Compiles the model with adam optimizer and our mixed loss

# Train model
model.fit(X_train, y_train, validation_split=0.1, epochs=250, batch_size=16, verbose=1) #Trains the dirty images with clean images as target

# Save model
model.save('unet_model.h5') #Saves the model
#UNET MODEL CODE END




	
#BEGIN DNCNN MODEL
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
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=250, batch_size=16,  validation_split=0.1, verbose=1)
# Save the model architecture and weights
model.save('dncnn_model.h5')
model.save_weights('dncnn_model.weights.h5')
#END DNCNN MODEL




#BEGIN IMAGE SAVING
model1 = load_model('transformer_model.keras')
model2 = load_model('unet_model.h5')
model3 = load_model('dncnn_model.weights.h5')
# Predict on a test set
predicted_images1 = model1.predict(X_test)
predicted_images2 = model2.predict(X_test)
predicted_images3 = model3.predict(X_test)
# Save original and reconstructed images for comparison
output_dir_original = '/mmfs1/home/bkphill2/output/original_images' #Saves the originals and reconstructed images to directory (Commented in other git code)
output_dir_reconstructed = '/mmfs1/home/bkphill2/output/reconstructed_images'

def save_as_flt(data, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Open the file in write-binary (wb) mode
    with open(file_path, 'wb') as file:
    # Write the data as floating point data
        file.write(data.astype(np.float32).tobytes())

for i in range(len(X_test)):
    original_img = X_test[i] * 255  # Scale if necessary, but keep as float if saving as .flt
    reconstructed_img1 = predicted_images1[i] * 255  # Scale if necessary, but keep as float
    reconstructed_img2 = predicted_images2[i] * 255	
    reconstructed_img3 = predicted_images3[i] * 255
    save_as_flt(original_img, os.path.join(output_dir_original, f'original_{i}.flt'))
    save_as_flt(reconstructed_img1, os.path.join(output_dir_reconstructed, f'reconstructed1_{i}.flt'))
    save_as_flt(reconstructed_img2, os.path.join(output_dir_reconstructed, f'reconstructed2_{i}.flt'))
    save_as_flt(reconstructed_img3, os.path.join(output_dir_reconstructed, f'reconstructed3_{i}.flt'))

print(f"Saved original images to {output_dir_original}")
print(f"Saved reconstructed images to {output_dir_reconstructed}")
#END IMAGE SAVING
