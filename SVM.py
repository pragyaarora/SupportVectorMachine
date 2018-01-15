import random
import numpy as np
from libsvm.svmutil import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')
np.random.seed(1)

def trainLinearSVM(x,y,c):
    p = svm_problem(y,x)
    param = svm_parameter('-t 0 -c ' + str(c))
    return svm_train(p, param)

def trainRBFSVM(x,y,c,alpha):
    p = svm_problem(y,x)
    param = svm_parameter('-t 2 -c ' + str(c)+' -g '+str(alpha))
    return svm_train(p, param)

def linearKernel(x_train,y_train,x_test,y_test):
    c_list = []
    acc_list = []

    for i in range(-4, 9):
        c = 2**i
        model = trainLinearSVM(x_train, y_train, c)
        p_label, p_acc, p_val = svm_predict(y_test, x_test, model)

        c_list.append(c)
        acc_list.append(p_acc[0])

    v,index = max([(v, i) for i, v in enumerate(acc_list)])
    print("******************Linear Kernel******************")
    print("Maximum accuracy :"+str(v))
    print("Best C :"+str(c_list[index]))
    print("*************************************************")
    plt.scatter(np.log(c_list), acc_list)
    plt.plot(np.log(c_list), acc_list, color="#52D017")
    plt.xlabel('log C')
    plt.ylabel("Accuracy")
    plt.title("C vs Accuracy")
    plt.show()

def rbfKernel(x_train,y_train):
    c_list = [2 ** i for i in range(-4, 9)]
    alpha_list = [2 ** i for i in range(-4, 9)]
    cv_results = np.zeros((13, 13))

    random_indexes = random.sample(range(0, len(x_train)), int(len(x_train) / 2))
    
    x_train_rand = [x_train[i] for i in random_indexes]
    y_train_rand = [y_train[i] for i in random_indexes]

    x_subset = []
    y_subset = []
    subset_size = int(len(x_train_rand) / 5)

    for i in range(5):
        x_subset.append(x_train_rand[i * subset_size:(i + 1) * subset_size])
        y_subset.append(y_train_rand[i * subset_size:(i + 1) * subset_size])

    for i in range(len(c_list)):
        c = c_list[i]
        for j in range(-4, 9):
            alpha = alpha_list[j]
            avg_acc = []
            for k in range(5):
                x_valid_set = x_subset[k]
                y_valid_set = y_subset[k]

                x_train_subset = x_subset[0:k] + x_subset[k + 1:]
                y_train_subset = y_subset[0:k] + y_subset[k + 1:]

                x_train_set = [x for x_t in x_train_subset for x in x_t]
                y_train_set = [y for y_t in y_train_subset for y in y_t]

                model = trainRBFSVM(x_train_set, y_train_set, c, alpha)
                p_label, p_acc, p_val = svm_predict(y_valid_set, x_valid_set, model)
                avg_acc.append(p_acc[0])

            cv_results[i, j] = np.average(avg_acc)


    i, j = np.unravel_index(cv_results.argmax(), cv_results.shape)
    print("******************RBF Kernel******************")
    print("Cross Validation Results")
    print(cv_results)
    print("Highest train accuracy : "+str(cv_results[i, j]))
    print("C : "+str(c_list[i]))
    print("Alpha : "+str(alpha_list[j]))
    print("***********************************************")
    return c_list[i],alpha_list[j]

if __name__ == "__main__":
    y_train, x_train = svm_read_problem('ncrna_s.train')
    y_test, x_test = svm_read_problem('ncrna_s.test')
    linearKernel(x_train,y_train,x_test,y_test)
    c, alpha = rbfKernel(x_train,y_train)
    model = trainRBFSVM(x_train,y_train,c,alpha)
    p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
    print()
    print("Test accuracy "+str(p_acc[0]))