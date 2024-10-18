import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import os


class AI:
    def __init__(self):
        np.set_printoptions(linewidth=150)
        self.plt_fontsize = matplotlib.rcParams['font.size']
        # loading the training and testing data, which are images for this example
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()

        # unprocessed data, kept for later
        self.raw_test_images, self.raw_test_labels = self.test_images, self.test_labels

        # finding the latest model version
        versions = [v.rsplit('.keras')[0] for v in os.listdir('./Number_Recognition/models') if v.endswith('.keras')]
        versions.sort()
        self.model_version = versions[-1]

    # making a window and displaying the last couple
    def mnist_peek(self, rows, cols):
        fig, axs = plt.subplots(rows, cols)
        for i in range(rows):
            for j in range(cols):
                axs[i, j].imshow(self.train_images[i*cols + j], cmap='gray')
                axs[i, j].set_title(self.train_labels[i*cols + j], fontsize=self.plt_fontsize)
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

        self.train_images = self.train_images.reshape((60000, 28 * 28)) # reshape flattens 28 x 28 arrays
        self.train_images = self.train_images.astype('float32') / 255   # cast as floats

        self.test_images = self.test_images.reshape((10000, 28 * 28))   # to vectors of 784 elements
        self.test_images = self.test_images.astype('float32') / 255     # and rescale to [0, 1]

        self.train_labels = tf.keras.utils.to_categorical(self.train_labels) # encode with the nifty `to_categorical` function
        self.test_labels  = tf.keras.utils.to_categorical(self.test_labels)

        # an empty model
        self.model = tf.keras.models.Sequential()

        # add 3 layers
        # sigmoid(w1+a1) * (wn +an) +b)
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
            ft_img_array = self.load_custom_images(ft_image_path)

            # label
            ft_label_array = tf.keras.utils.to_categorical([ft_label], num_classes=10)

            # Adjust the learning rate for fine-tuning
            self.model.optimizer.learning_rate.assign(0.0001)

            # train the model again
            self.model.fit(ft_img_array, ft_label_array, epochs=10, batch_size=1)
            self.save_model(self.model_version)
        else:
            if ft_image_path and ft_label is None:
                return (False, "No label provided")
            elif ft_label is not None and ft_image_path is None:
                return (False, "No image path provided")
            else:
                return (False, "No image path and label provided")

            

    def predict(self, image_path = None):
        if image_path:
            img = self.load_custom_images(image_path)

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
        
    def load_custom_images(self, path):
        image = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale', target_size=(28, 28))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.reshape(1, 28 * 28)
        image = image.astype('float32') / 255

        return image

    def update(self):
        new_images_array = [image for image in os.listdir('./Number_Recognition/images') if image.endswith('.bmp')]
        new_image_labels = [int(image.split('_')[-1].split('.')[0]) for image in new_images_array]
        self.new_images_array = []

        for image in new_images_array:
            path = os.path.join('./Number_Recognition/images', image)
            self.new_image_array = self.load_custom_images(path)
            self.new_images_array.append(self.new_image_array)

        # Convert the list of image arrays to a NumPy array
        self.new_images_array = np.vstack(self.new_images_array)

        # One-hot encode the labels

        if len(self.train_images.shape) == 3:
            self.train_images = self.train_images.reshape((self.train_images.shape[0], 28 * 28))

        if len(self.train_labels.shape) == 1:
            self.train_labels = tf.keras.utils.to_categorical(self.train_labels, num_classes=10)

        self.new_image_labels = tf.keras.utils.to_categorical(new_image_labels, num_classes=10)

        self.train_images = np.vstack((self.train_images, self.new_images_array))
        self.train_labels = np.vstack((self.train_labels, self.new_image_labels))

        # Adjust the learning rate for update
        self.model.optimizer.learning_rate.assign(0.0001)

        self.model.fit(self.train_images, self.train_labels, epochs=5, batch_size=128)

        new_version = float(self.model_version) + 0.1
        self.save_model(str(new_version))


if __name__ == "__main__":
    ai = AI()
    ai.train()
    ai.update()