import os  # Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Subtract
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from PIL import Image

# Import necessary functions and classes

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
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs) #Initial convolutional layer
    c1 = layers.Dropout(0.1)(c1) #Drops 10% of neurons from layer 1
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1) #Second convolution
    p1 = layers.MaxPooling2D((2, 2))(c1) #Does a 2x2 pooling
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1) #Next convolution, now 32 layers deep
    c2 = layers.Dropout(0.1)(c2) #Dropout for the 32 layers
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2) #Final convolution
    p2 = layers.MaxPooling2D((2, 2))(c2) #2x2 pooling of layer 2
    
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2) #3rd convolutional 64
    c3 = layers.Dropout(0.2)(c3) #20% dropout from layer 3
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3) #Final layer 3 convolution
    p3 = layers.MaxPooling2D((2, 2))(c3) #Pooling 2x2 of 64 layers
    
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3) #4th layer 128
    c4 = layers.Dropout(0.2)(c4) #20% drop from 128
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4) #128 final layer
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4) #Pooling 2x2 of 128 layer
    
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4) #5th layer 256
    c5 = layers.Dropout(0.3)(c5) #30% dropout of 256
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5) #Final 256 convolution
    # Upsample
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5) # 128 transposed 2x2
    u6 = layers.concatenate([u6, c4]) #Concatenates 128 and 4th convolution
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6) #128 final convolution
    c6 = layers.Dropout(0.2)(c6) #20% dropout of 128
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6) #2nd 128 convolution
    
    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6) #128 transposed 2x2
    u7 = layers.concatenate([u7, c3]) #Concatenates 64 and 3rd layer
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7) #Convolution layer 64
    c7 = layers.Dropout(0.2)(c7) #20% dropout of 64
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7) #Final 2nd 64 layer
    
    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7) #2x2 transposed 64
    u8 = layers.concatenate([u8, c2]) #Concatenates the 32 and 2 layers
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8) #Convolution 2nd 32
    c8 = layers.Dropout(0.1)(c8) #10% dropout of the 2nd 32
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8) #Final 2nd 32
    
    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8) #Transpose 2x2 32
    u9 = layers.concatenate([u9, c1], axis=3) #Concatenates the 16 and 1
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9) #Convolution 16
    c9 = layers.Dropout(0.1)(c9) #10% of 16
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9) #2nd final layer
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9) #1x1 filter of final layer
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs]) #Keras model of all inputs and outputs
    return model #Return to function
    model = unet_model(input_size=(512, 512, 1)) #Creates a model with 512x512x1 inputs
    model.compile(optimizer=‘adam’, loss=combined_loss, metrics=[‘accuracy’, ssim_loss, mse_loss]) #Optimizer of adam
    model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=8, verbose=1) #Fit function for 50 epochs
    model.save(‘unet_model.h5’) 
