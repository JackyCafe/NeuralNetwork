import numpy as np
from scipy.special import expit

class neuralNetWork:
    def __init__(self, inputNode, hiddenNode, outputNode, learning_rate):
        self.iNode = inputNode
        self.hNode = hiddenNode
        self.oNode = outputNode
        self.lr = learning_rate
        # input->hidden 權重
        self.wih = np.random.normal(0.0, pow(self.hNode, -0.5), (self.hNode, self.iNode))
        # hidden->output 權重
        self.who = np.random.normal(0.0, pow(self.oNode, -0.5), (self.oNode, self.hNode))
        self.ac_fn = lambda x: expit(x)

    def train(self):
        pass

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_input = np.dot(self.wih, inputs)
        hidden_output = self.ac_fn(hidden_input)
        final_hidden = np.dot(self.who, hidden_output)
        final_output = self.ac_fn(final_hidden)
        return final_output


if __name__ == '__main__':
    inputNode = 2
    hiddenNode = 3
    outputNode = 1
    lr = 0.03
    n = neuralNetWork(inputNode, hiddenNode, outputNode, lr)
    input = np.array([[1, 2]])
    a = n.query(input)
    print(a)