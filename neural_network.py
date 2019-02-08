import data_loader
import numpy as np

class neural_network:

    def __init__(self, training_data, training_labels):
        self.weights = np.random.rand(10,784)
        self.bias = np.random.randn(10)
        self.training_data = training_data
        self.training_labels = training_labels

    def learn(self, learning_rate):
        ind = 0
        ind1 = 0
        cost = 0
        while ind + 784 < len(self.training_data):
            training_image = self.training_data[ind:ind+784]
            training_label = self.training_labels[ind1]
            training_image = np.add(-1.0*np.matmul(self.weights, training_image), self.bias)
            activations0 = np.divide(1.0, 1.0+np.exp(training_image))
            activations = np.subtract(training_label, activations0)

            self.bias += learning_rate*activations
            for i in range(0,10):
                cost += activations[i] ** 2
                self.weights[i] += learning_rate*activations[i]
            if ind1 % 10000 == 0:
                print("\nCorrect:     {}".format(training_label))
                print("Neural Nets: {}".format(activations0.round(2)))
                print("Cost: {}".format(float(cost/(20*ind1))))
            ind += 784
            ind1 += 1
        print(ind1)

    def evaluate(self, checking_data, checking_labels):
        score = 0
        ind = 0
        ind1 = 0
        while ind + 784 < len(checking_data):
            checking_image = checking_data[ind:ind + 784]
            checking_label = checking_labels[ind1]
            check = np.add(-1.0 * np.matmul(self.weights, checking_image), self.bias)
            activations = np.divide(1.0, 1.0 + np.exp(check))
            activations = list(activations)
            checking_label = list(checking_label)
            index1 = activations.index(max(activations))
            index2 = checking_label.index(max(checking_label))
            if index1 == index2:
                score += 1
            ind+=784
            ind1+=1
        print("Score: {}".format(score))

if __name__ == "__main__":
    training_data = data_loader.load_training_images()
    training_labels = data_loader.load_training_labels()
    checking_data = data_loader.load_checking_images()
    checking_labels = data_loader.load_checking_labels()
    learning_rate = 1
    persona = neural_network(training_data, training_labels)
    persona.learn(learning_rate)
    persona.learn(learning_rate)
    persona.learn(learning_rate)
    persona.evaluate(checking_data, checking_labels)
