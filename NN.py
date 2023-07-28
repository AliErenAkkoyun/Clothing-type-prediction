from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels) , (test_images , test_datas) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255
test_images = test_images/255

plt.grid(False)
plt.imshow(test_images[1])


model = keras.Sequential([
keras.layers.Flatten(input_shape=(28,28)),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss= "sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train_images,train_labels, epochs=9)

test_loss , test_acc = model.evaluate(test_images, test_datas)

prediction = model.predict(test_images)


for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i])
	plt.xlabel("Gercek Ürün: "+ class_names[test_datas[i]]) # test_label[i] yazıyordu
	plt.title("Tahmin Edilen Ürün: " + class_names[np.argmax(prediction[i])])
	plt.show()
