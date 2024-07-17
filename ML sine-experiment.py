import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.keras import layers

SEED = 42069
SAMPLES = 1000
np.random.seed(SEED)
tf.random.set_seed(SEED)
x_values = np.random.uniform(low=0,high=2*math.pi,size=SAMPLES)
np.random.shuffle(x_values)
y_values = np.sin(x_values)
print(y_values.shape)
plt.plot(x_values,y_values,'b.')
plt.show()
noise = 0.1*np.random.randn(SAMPLES)
y_values = y_values+noise
print(noise.shape)
plt.plot(x_values,y_values,'r.')
plt.show()
# Splitting Data
Train_Split = int(0.8*SAMPLES)
Test_Split = int(0.1*SAMPLES+Train_Split)
x_train,x_validation,x_test=np.split(x_values,[Train_Split,Test_Split])
y_train,y_validation,y_test=np.split(y_values,[Train_Split,Test_Split])
assert(x_train.size+x_validation.size+x_test.size) == SAMPLES
plt.plot(x_train,y_train,'b.',label='train')
plt.plot(x_test,y_test,'r.',label='Test')
plt.plot(x_validation,y_validation,'y.',label='Validation')
plt.legend()
plt.show()

#####################
#Define a Model
model=tf.keras.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(1,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(16,activation='sigmoid'))
model.add(layers.Dense(1))
model.compile(optimizer="rmsprop",loss="mse",metrics=['mae'])
model.summary()

#Training and Evaluate the model
history=model.fit(x_train,y_train,epochs=500,batch_size=16,validation_data=(x_validation,y_validation))
loss=model.evaluate(x_test,y_test)
predictions = model.predict(x_test)
plt.clf()
plt.plot(x_test,y_test,'b.',label="Actual")
plt.plot(x_test,predictions,'r.',label = "Predicted")
plt.legend()
plt.show()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset_generator():
    for value in x_test:
        yield[np.array(value,dtype=np.float32,ndmin=2)]
converter.representative_dataset = representative_dataset_generator
tflite_model = converter.convert()
open("sine_model_quantized.tflite",'wb').write(tflite_model)
