import tensorflow as tf
from dncnn2 import DnCNN
from Transformer import TransformerBlock,build_transformer_model
from unet_model_library import unet_model
from Combined_Loss import combined_loss

#to quickly get a model summary for a model save file

MODEL_Transformer = "model_Transformer.keras"
MODEL_Unet = "model_Unet.keras"
MODEL_DnCNN = "model_DnCNN.keras"

model_choice = input("(1) Transformer (2) Unet (3) DnCNN: ")

if model_choice == "1":
    model_file = MODEL_Transformer
elif model_choice == "2":
    model_file = MODEL_Unet 
else:
    model_file = MODEL_DnCNN 

model = tf.keras.models.load_model(model_file,custom_objects={
                                        'combined_loss':combined_loss,
                                        'TransformerBlock':TransformerBlock})
tf.print(model.summary())
