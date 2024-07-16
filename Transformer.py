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

@register_keras_serializable(package='Custom', name='TransformerBlock')
class TransformerBlock(layers.Layer):  #Creates the transformer block class
      def __init__(self, projection_dim, num_heads, **kwargs): #constructor  method with params of self, the dim of embedding space, and number of transformer heads
            super(TransformerBlock, self).__init__(**kwargs) #calls layers.layer parent class and init.
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

model = build_transformer_model(num_patches, projection_dim, num_heads, transformer_layers, num_classes)
