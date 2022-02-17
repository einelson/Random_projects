from tensorflow import keras
from keras.utils import np_utils
import time
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('./data/diabetes.csv')

# convert targets into 1 or 0
# No, Negative, Male = 0
# Yes, Positive, Female = 1
df = df.replace(to_replace =["No", "Negative", 'Male'], value =0)
df = df.replace(to_replace =["Yes", "Positive", 'Female'], value =1)


data=df.iloc[:,:-1].to_numpy()
data_targets=df.iloc[:,-1]
data_targets=np_utils.to_categorical(data_targets, num_classes=2)
# print(data)

# split the data
xTrain, xTest, yTrain, yTest = train_test_split(data, data_targets, test_size = 0.1)


# build neural network

# Block 1
inputs=keras.Input(shape=(16,))
x=keras.layers.Dense(256)(inputs)
x=keras.layers.Dense(256, activation='relu')(x)
x=keras.layers.Dense(128)(x)
x=keras.layers.Dense(128, activation='relu')(x)
x=keras.layers.Dense(128)(x)
block_1_output = keras.layers.Dense(64)(x)

# Block 2
# x=keras.layers.Dense(128)(inputs)
# x=keras.layers.Dense(128)(x)
# x=keras.layers.Dense(64, activation='relu')(x)
# block_2_output=keras.layers.Dense(64)(x)


# combine blocks
# combined_output = keras.layers.concatenate([block_1_output, block_2_output])
# x=keras.layers.Dense(128, activation='relu')(combined_output)
outputs = keras.layers.Dense(2, activation='sigmoid')(x)

# save model and graph
model=keras.Model(inputs=inputs, outputs=outputs, name="diabetes_model")
keras.utils.plot_model(model, "./data/eeg_model.png", show_shapes=True)
model.summary()

# compile and fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=xTrain, y=yTrain,epochs=50,batch_size=10, validation_data=(xTest,yTest))

score, acc = model.evaluate(x=xTest, y=yTest)
network = model.predict(xTest)

print('Accuracy: ',100*(acc))
# back to home directory, save the model
run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
new_dir='./models/'+str(100*acc)+'_'+run_id+'.h5'
model.save(new_dir)



