import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import os

np.set_printoptions(linewidth=150)
plt_fontsize = matplotlib.rcParams['font.size']

class AI:
    def __init__(self):
        # loading the training and testing data, which are images for this example
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()

        # unprocessed data, kept for later
        self.raw_test_images, self.raw_test_labels = self.test_images, self.test_labels

        self.model_version = "0.1"

    # making a window and displaying the last couple
    def mnist_peek(self, rows, cols):
        fig, axs = plt.subplots(rows, cols)
        for i in range(rows):
            for j in range(cols):
                axs[i, j].imshow(self.train_images[i*cols + j], cmap='gray')
                axs[i, j].set_title(self.train_labels[i*cols + j], fontsize=plt_fontsize)
                axs[i, j].axis('off')
        plt.show()

    # mnist_peek(6, 6)

    def save_model(self, name):
        model_path = f"./Number_Recognition/models/{name}.keras"
        self.model.save(model_path)

    def load_model(self, name):
        model_path = f"./Number_Recognition/models/{name}.keras"
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            return "Model path does not exist"
        
    def train(self):
        # # printing the multidimensional arrays 
        # # (because we cant show or even imagine an array with 60k x 28 x 28 elements,
        # #  its just printing the dimensions)
        # print("tensor:\t\t shape")
        # print("-"*22)
        # print("training images:", train_images.shape)
        # print("training labels:", train_labels.shape)
        # print("test images:\t", test_images.shape)
        # print("test labels:\t", test_labels.shape)

        # # to print the pixel values of the test image 0
        # print(self.test_images[0])

        # # inspect the first image using the matplotlib graphics library
        # print('label: ', self.test_labels[0])
        # plt.imshow(self.test_images[0], cmap=plt.cm.binary)
        # plt.show()


        self.train_images = self.train_images.reshape((60000, 28 * 28)) # reshape flattens 28 x 28 arrays
        self.test_images = self.test_images.reshape((10000, 28 * 28))   # to vectors of 784 elements

        self.train_images = self.train_images.astype('float32') / 255   # cast as floats
        self.test_images = self.test_images.astype('float32') / 255     # and rescale to [0, 1]

        self.train_labels = tf.keras.utils.to_categorical(self.train_labels) # encode with the nifty `to_categorical` function
        self.test_labels  = tf.keras.utils.to_categorical(self.test_labels)

        # an empty model
        self.model = tf.keras.models.Sequential()

        # add 3 layers
        self.model.add(tf.keras.layers.Input((28 * 28, )))
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dense(10, activation='softmax'))

        # loss, optimizer and metrics are chosen at compilation
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=['accuracy']
        )

        # training - fit to input data
        self.model.fit(self.train_images, self.train_labels, epochs=5, batch_size=128)
        self.save_model(self.model_version)

    def fine_tune_model(self, ft_image_path=None, ft_label=None):
        if ft_image_path and ft_label is not None:
            print("fine tuning on:", ft_label)
            # img
            ft_img = tf.keras.preprocessing.image.load_img(ft_image_path, color_mode='grayscale', target_size=(28, 28))
            ft_img_array = tf.keras.preprocessing.image.img_to_array(ft_img)
            ft_img_array = ft_img_array.reshape(1, 28 * 28)
            ft_img_array = ft_img_array.astype('float32') / 255

            # label
            ft_label_array = tf.keras.utils.to_categorical([ft_label], num_classes=10)

            print(ft_label_array)

            # Adjust the learning rate for fine-tuning
            self.model.optimizer.learning_rate.assign(0.0001)
            # breakpoint()
            # if isinstance(learning_rate, tf.Variable):
            #     learning_rate.assign(0.00001)
            # elif isinstance(learning_rate, tf.Tensor):
            #     self.model.optimizer.learning_rate = tf.keras.backend.variable(0.00001)
            # else:
                # tf.keras.backend.set_value(self.model.optimizer.learning_rate, 0.00001)


            # train the model again
            self.model.fit(ft_img_array, ft_label_array, epochs=10, batch_size=1)
        else:
            if ft_image_path and ft_label is None:
                return (False, "No label provided")
            elif ft_label is not None and ft_image_path is None:
                return (False, "No image path provided")
            else:
                return (False, "No image path and label provided")

            

    def predict(self, image_path = None):
        if image_path:
            img = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(28, 28))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img.reshape(1, 28 * 28)
            img = img.astype('float32') / 255
            preds = self.model.predict(img, verbose=0)
            argm = np.argmax(preds[0])
            fig, ax = plt.subplots()
            for i, x in enumerate(preds[0]):
                if i == argm:
                    ax.bar(i, x, color='red')
                    # print(f"\033[1mclass: {i:2} with probability: {x:.15f}\033[0m")
                else:
                    ax.bar(i, x, color='blue')
                    # print(f"class: {i:2} with probability: {x:.15f}")

            ax.set_title(f"Predicted class: {argm}")
            ax.set_xticks(range(10))

            ax.set_ylim(0, 1)  # Set the y-axis range between 0 and 1
            ax.set_yticks(np.arange(0, 1.1, 0.2))  # Set y-ticks from 0 to 1 with increments of 0.2

            plt.show()
            return argm
        else:
            return "No image path provided"

if __name__ == "__main__":
    ai = AI()
    ai.train()