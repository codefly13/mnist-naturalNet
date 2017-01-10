import numpy as np
# Third-party libraries
from sklearn import svm

from tensorflow.examples.tutorials.mnist import input_data
# 下载mnist数据集
mnist = input_data.read_data_sets('/tmp/', one_hot=True)


def svm_baseline():
    import time
    training_data, test_data = (mnist.train.images, np.argmax(mnist.train.labels, axis=1)), (mnist.test.images, np.argmax(mnist.test.labels, axis=1))
    s = time.time()
    # train
    clf = svm.SVC(C=10.0, kernel='rbf', degree=2, gamma='auto')
    clf.fit(training_data[0], training_data[1])
    print(time.time() - s)
    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print("Baseline classifier using an SVM.")
    print("%s of %s values correct." % (num_correct, len(test_data[1])))




if __name__ == "__main__":
    svm_baseline()

    
