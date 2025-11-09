"""
segmentation_train.py
---------------------
Trains the Attention SegNet model for lesion segmentation.
Model weights (Att_Segnet.keras) are saved after training.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def attention_gate(g, s, num_filters):
    Wg = L.Conv2D(num_filters, 1, padding="same")(g)
    Wg = L.BatchNormalization()(Wg)

    Ws = L.Conv2D(num_filters, 1, padding="same")(s)
    Ws = L.BatchNormalization()(Ws)

    out = L.Activation("relu")(Wg + Ws)
    out = L.Conv2D(num_filters, 1, padding="same")(out)
    out = L.Activation("sigmoid")(out)

    return out * s
img_input = Input(shape= (256, 256, 3))
x0 = Conv2D(64, (3, 3), padding='same', name='conv1',strides= (1,1))(img_input)
x0= BatchNormalization(name='bn1')(x0)
x0 = Activation('relu')(x0)
x0 = Conv2D(64, (3, 3), padding='same', name='conv2')(x0)
x0 = BatchNormalization(name='bn2')(x0)
x0 = Activation('relu')(x0)
p0 = MaxPooling2D()(x0)

x1 = Conv2D(128, (3, 3), padding='same', name='conv3')(p0)
x1 = BatchNormalization(name='bn3')(x1)
x1 = Activation('relu')(x1)
x1 = Conv2D(128, (3, 3), padding='same', name='conv4')(x1)
x1 = BatchNormalization(name='bn4')(x1)
x1 = Activation('relu')(x1)
p1 = MaxPooling2D()(x1)

x2 = Conv2D(256, (3, 3), padding='same', name='conv5')(p1)
x2 = BatchNormalization(name='bn5')(x2)
x2 = Activation('relu')(x2)
x2 = Conv2D(256, (3, 3), padding='same', name='conv6')(x2)
x2 = BatchNormalization(name='bn6')(x2)
x2 = Activation('relu')(x2)
x2 = Conv2D(256, (3, 3), padding='same', name='conv7')(x2)
x2 = BatchNormalization(name='bn7')(x2)
x2 = Activation('relu')(x2)
p2 = MaxPooling2D()(x2)

x3 = Conv2D(512, (3, 3), padding='same', name='conv8')(p2)
x3 = BatchNormalization(name='bn8')(x3)
x3 = Activation('relu')(x3)
x3 = Conv2D(512, (3, 3), padding='same', name='conv9')(x3)
x3 = BatchNormalization(name='bn9')(x3)
x3 = Activation('relu')(x3)
x3 = Conv2D(512, (3, 3), padding='same', name='conv10')(x3)
x3 = BatchNormalization(name='bn10')(x3)
x3 = Activation('relu')(x3)
p3 = MaxPooling2D()(x3)

x4 = Conv2D(512, (3, 3), padding='same', name='conv11')(p3)
x4 = BatchNormalization(name='bn11')(x4)
x4 = Activation('relu')(x4)
x4 = Conv2D(512, (3, 3), padding='same', name='conv12')(x4)
x4 = BatchNormalization(name='bn12')(x4)
x4 = Activation('relu')(x4)
x4 = Conv2D(512, (3, 3), padding='same', name='conv13')(x4)
x4 = BatchNormalization(name='bn13')(x4)
x4 = Activation('relu')(x4)
x4 = MaxPooling2D()(x4)
x4 = Dense(1024, activation = 'relu', name='fc1')(x4)
x4 = Dense(1024, activation = 'relu', name='fc2')(x4)
  # Decoding Layer
x = UpSampling2D()(x4)
x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv1')(x)
s = attention_gate(x, p3, 512)
x = L.Concatenate()([x, s])
#soft attention
x = BatchNormalization(name='bn14')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv2')(x)
x = BatchNormalization(name='bn15')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv3')(x)
x = BatchNormalization(name='bn16')(x)
x = Activation('relu')(x)

x = UpSampling2D()(x)
x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv4')(x)
x = BatchNormalization(name='bn17')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(512, (3, 3), padding='same', name='deconv5')(x)
x = BatchNormalization(name='bn18')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv6')(x)
x = BatchNormalization(name='bn19')(x)
x = Activation('relu')(x)

#x = UpSampling2D()(x)
x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv7')(x)
#soft attention
s = attention_gate(x, p2, 256)
x = L.Concatenate()([x, s])
x = BatchNormalization(name='bn20')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(256, (3, 3), padding='same', name='deconv8')(x)
x = BatchNormalization(name='bn21')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv9')(x)
x = BatchNormalization(name='bn22')(x)
x = Activation('relu')(x)

x = UpSampling2D()(x)
x = Conv2DTranspose(128, (3, 3), padding='same', name='deconv10')(x)
#soft attention
s = attention_gate(x, p1, 128)
x = L.Concatenate()([x, s])
x = BatchNormalization(name='bn23')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv11')(x)
x = BatchNormalization(name='bn24')(x)
x = Activation('relu')(x)

x = UpSampling2D()(x)
x = Conv2DTranspose(64, (3, 3), padding='same', name='deconv12')(x)
#soft attention
s = attention_gate(x, p0, 64)
x = L.Concatenate()([x, s])
x = BatchNormalization(name='bn25')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(3, (3, 3), padding='same', name='deconv13')(x)
x = BatchNormalization(name='bn26')(x)
x = UpSampling2D()(x)
pred = Activation('sigmoid')(x)
#pred = Reshape((256,256))(x)


model = Model(inputs=img_input, outputs=pred)

model.compile(optimizer= SGD(learning_rate=0.001, momentum=0.9, decay=0.0005, nesterov=False), loss= ["binary_crossentropy"]
                  , metrics=['accuracy'])

# (Add your data loading pipeline here)
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
model.save("Att_Segnet.keras")
