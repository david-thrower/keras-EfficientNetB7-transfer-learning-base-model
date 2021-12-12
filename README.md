Take the pre-trained version of tf.keras.applications.efficientnet.EfficientNetB7 and make it 
amenable to transfer learning:

    1. Import the model with its pre-trained weights. These are very transferable.
    2. Set all layers except the last convolutional2d layer as trainable=False. The last convolutional layer, we leave as trainable...
        1. This is the convolutional layer that will look at the deepest interactions of all these layers (the most use case specific information). It may be worth burning the comutational jet fuel to train this layer.
    3. Drop the dense layer at the output end of the model. Also drop the dropout layer. This can clash with BatchNormalization layers if they are on the input end of the block of Dense layers you forward the output of this model to.

Usage:

1. Build the model by running the code from the notebook. (Sorry, GitHub doesn't host files as big as this model saved with lf.keras.Model,save()... I tried ...)
2. Build a customized model using the Keras functional API:
```{python

# Load the base model
base_model = tf.keras.models.load_model('efficient_net_b_7_transferable_base_model')

# Build your customized model:

NUMBER_OF_CLASSES = 10

inp = base_model.layers[0].input
out = tf.keras.layers.Dense(NUMBER_OF_CLASSES = 10,'softmax')(base_model.layers[-1].output)  

# Sometimes a multi layer perceptron performs better than a single dense layer 
# In other words, you might want to add some layers that will make the lines above look like this: 
#    inp = base_model.layers[0].input
#    x = tf.keras.layers.Dense([some positive integer],'relu')(base_model.layers[-1].output)
#    x = tf.keras.layers.Dense([some positive integer],'relu')(x)
#    x = tf.keras.layers.Dense([some positive integer],'relu')(x)
#    out = tf.keras.layers.Dense(NUMBER_OF_CLASSES = 10,'softmax')(x)


my_model = tf.keras.Model(inputs = inp,outputs = out) # 
my_model.compile(optimizer="Adam",loss='categorical-crossentropy',metrics=['precision','accuracy'])
}
```
# Shared under MIT License except these exclusions:

    1. Military use, law enforcement use intended to lead to or manage incarceration, use in committing crimes, use in marketing / distrubuting / managing of adult films, 
       alcoholic beverages, tobaco, or prescription drugs of abuse, and use in a manner intended to identify or discriminate against anyone on any ethnic, ideological, 
       religious, racial, demographic, or socioeconomic / credit status (including credit and HR screening other than screening for criminal history) are excluded from the 
       license and are prohibited uses of this code.
