import numpy as np
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

""" 
You are allowed to change the names of function "arguments" as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.
"""



def get_labeled_features(file_path):
    """Read data from train.csv and split into train and dev sets. Do any
       preprocessing/augmentation steps here and return final features.
       

    Args:
        file_path (str): path to train.csv

    Returns:
        phi_train, y_train, phi_dev, y_dev
    """
    df = pd.read_csv(file_path)
    df = df.drop(['type'], axis = 1)

    df['fixed acidity'] = (df['fixed acidity'] - np.min(df['fixed acidity'])) / (np.max(df['fixed acidity']) - np.min(df['fixed acidity']))
    df['residual sugar'] = (df['residual sugar'] - np.min(df['residual sugar'])) / (np.max(df['residual sugar']) - np.min(df['residual sugar']))
    df['free sulfur dioxide'] = (df['free sulfur dioxide'] - np.min(df['free sulfur dioxide'])) / (np.max(df['free sulfur dioxide']) - np.min(df['free sulfur dioxide']))
    df['total sulfur dioxide'] = (df['total sulfur dioxide'] - np.min(df['total sulfur dioxide'])) / (np.max(df['total sulfur dioxide']) - np.min(df['total sulfur dioxide']))
    df.pH = (df.pH - np.min(df.pH)) / (np.max(df.pH) - np.min(df.pH))
    df.alcohol = (df.alcohol - np.min(df.alcohol)) / (np.max(df.alcohol) - np.min(df.alcohol))

    phi = df.drop(['quality'], axis = 1)
    y = df.quality

    l = len(df.quality)
    phi_train = phi[:, :int(0.75*l)]
    y_train = y[:, :int(0.75*l)]
    phi_dev = phi - phi_train
    y_dev = y - y_train

    return phi_train, y_train, phi_dev, y_dev                   
    
"""done"""

def get_test_features(file_path):
    """Read test data, perform required preproccessing / augmentation
       and return final feature matrix.

    Args:
        file_path (str): path to test.csv

    Returns:
        phi_test: matrix of size (m,n) where m is number of test instances
                  and n is the dimension of the feature space.
    """
    df1 = pd.read_csv(file_path)
    df1 = df1.drop(['type'], axis = 1)

    df1['fixed acidity'] = (df1['fixed acidity'] - np.min(df1['fixed acidity'])) / (np.max(df1['fixed acidity']) - np.min(df1['fixed acidity']))
    df1['residual sugar'] = (df1['residual sugar'] - np.min(df1['residual sugar'])) / (np.max(df1['residual sugar']) - np.min(df1['residual sugar']))
    df1['free sulfur dioxide'] = (df1['free sulfur dioxide'] - np.min(df1['free sulfur dioxide'])) / (np.max(df1['free sulfur dioxide']) - np.min(df1['free sulfur dioxide']))
    df1['total sulfur dioxide'] = (df1['total sulfur dioxide'] - np.min(df1['total sulfur dioxide'])) / (np.max(df1['total sulfur dioxide']) - np.min(df1['total sulfur dioxide']))
    df1.pH = (df1.pH - np.min(df1.pH)) / (np.max(df1.pH) - np.min(df1.pH))
    df1.alcohol = (df1.alcohol - np.min(df1.alcohol)) / (np.max(df1.alcohol) - np.min(df1.alcohol))

    phi_test = df1
    
    return phi_test   
   
    
"""done"""

def compute_RMSE(phi, w , y) :
   """Return root mean squared error given features phi, and true labels y."""
   error = np.sqrt(mean_squared_error(w.T.dot(phi)),y)
   return error    

"""done"""

def generate_output(phi_test, w):
    """writes a file (output.csv) containing target variables in required format for Submission."""
    a = []
    for i in range(0,len(w)-1)
        a.append(i) 

    d = pd.DataFrame(a,columns = ['id'])
    d = d.join(w))
    d.to_csv('output.csv',index=False)

   pass                     

"""done"""
    
   
def closed_soln(phi, y):
    """Function returns the solution w for Xw = y."""
    return np.linalg.pinv(phi).dot(y)  
    
"""done"""
   


def gradient_descent(phi_train, y_train, phi_dev, y_dev) :
   # Implement gradient_descent using Mean Squared Error Loss
   # You may choose to use the dev set to determine point of convergence

    pre_mse = 1
    mse = pre_mse
    alpha = 0.05
    m = np.shape(phi_train)[0]  # total number of samples
    n = np.shape(phi_train)[1]  # total number of features
 
    phi_train = np.concatenate((np.ones((m, 1)), phi_train), axis=1)
    W = np.random.randn(n + 1, )
    mse = 10
    epsilon = 0.01

    while mse - pre_mse < 1e-4
        error = phi_train.dot(W) - y_train
   #     cost = (1 / 2 * m) * np.sum(error ** 2)
        W -= alpha * (1 / m) * phi_train.T.dot(error)
        pre_mse = mse
        mse = mean_squared_error(y_dev, phi_dev.dot(W))  

    return W

"""done"""

def sgd(phi, y, phi_dev, y_dev) :
    
    # Implement stochastic gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence

    pre_mse = 1
    mse = pre_mse
    epsilon = 0.01
    k=40
    alpha = 0.05

    while mse - pre_mse < 1e-4
        
        temp1 = phi.sample(k)
        temp2 = y.sample(k)

        phi_tr = temp1.iloc[:,0:13].values
        y_tr = temp2.iloc[:,-1].values
                
        m = np.shape(phi_tr)[0]  # total number of samples
        n = np.shape(phi_tr)[1]  # total number of features
 
        phi_tr = np.concatenate((np.ones((m, 1)), phi_tr), axis=1)
        w = np.random.randn(n + 1, )

        error = y_tr - phi_tr.dot(w)
        w -= alpha * (1 / m) * phi_tr.T.dot(error)
        alpha /= 1.02
        pre_mse = mse
        mse = mean_squared_error(y_dev, phi_dev.dot(w)) 
        
    return w


def pnorm(phi, y, phi_dev, y_dev, p) :
   # Implement gradient_descent with p-norm regularization using Mean Squared Error Loss
   # You may choose to use the dev set to determine point of convergence
    w = np.random.random((phi.shape[1],1))
    n = phi.shape[0]
    L = []
    lam = 1e-5
    pre_rmse = compute_RMSE(phi_dev, w , y_dev) + lam*np.sum(np.absolute(w)**p)
    alpha = 0.01
    while 1:
        grad = (phi.T)@(y - phi@w)*(-2/n) + lam*(w**(p-1))
        w -= alpha*grad/np.sqrt(np.sum(grad**2))
        rmse = compute_RMSE(phi_dev, w , y_dev) + lam*np.sum(np.absolute(w)**p)
        conv = abs(pre_rmse - rmse)
        if conv < 1e-4:
            break
        pre_rmse = rmse
        L.append(rmse)
    plt.plot(range(len(L)),L)
    return w   

def plot_rmse(phi,y):

    rmse = []
    size_of_train = [0.1, 0.25, 0.5, 0.75, 0.9]

    for i in 1:5 : 

        l = len(y)
        phi_train = phi[:, :int(size_of_train[i]*l)]
        y_train = y[:, :int(size_of_train[i]*l)]
        phi_dev = phi - phi_train
        y_dev = y - y_train
        w1 = gradient_descent(phi_train, y_train, phi_dev, y_dev)
        r1 = compute_RMSE(phi_dev, w1, y_dev)
        rmse[i] = r1

    plt.plot(size_of_train , rmse)
    plt.xlabel('size_of_train')
    plt.ylabel('rmse value')

    return 0


def main():
    """ 
    The following steps will be run in sequence by the autograder.
    """
    ######## Task 2 #########
    phi, y, phi_dev, y_dev = get_labeled_features('train.csv')
    w1 = closed_soln(phi, y)
    w2 = gradient_descent(phi, y, phi_dev, y_dev)
    r1 = compute_RMSE(phi_dev, w1, y_dev)
    r2 = compute_RMSE(phi_dev, w2, y_dev)
    print('1a: ')
    print(abs(r1-r2))
    w3 = sgd(phi, y, phi_dev, y_dev)
    r3 = compute_RMSE(phi_dev, w3, y_dev)
    print('1c: ')
    print(abs(r2-r3))

    ######## Task 3 #########
    w_p2 = pnorm(phi, y, phi_dev, y_dev, 2)  
    w_p4 = pnorm(phi, y, phi_dev, y_dev, 4)  
    r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
    r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
    print('2: pnorm2')
    print(r_p2)
    print('2: pnorm4')
    print(r_p4)

    ######## Task 6 #########
    
    # Add code to run your selected method here
    # print RMSE on dev set with this method

main()
