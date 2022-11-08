import numpy as np
class LinearRegression:
    
    def __init__(self, type, L, alpha):
        self.W=None
        self.L=L
        self.type=type
        if(type==0):
            self.L=None
        self.alpha=alpha
        
    
    def norm(self, X):
        temp=np.matmul(X.T, X)
        return temp[0][0]    
    
    def set(self, X_train):
        self.W=np.zeros((len(X_train[0]), 1))
    def update(self, X_train, Y_train):     # updates weight only once; one iteration of gradient descent
        
        # temp0=np.matmul(X_train.T, self.weights)
        temp1=np.matmul(np.matmul(X_train.T, X_train), self.W)
        Y_temp=Y_train.reshape((Y_train.shape[0], 1))
        temp2=np.matmul(X_train.T, Y_temp).reshape(self.W.shape)
        if(self.type==0):
            self.W=self.W-(self.alpha*((temp1-temp2)/X_train.shape[0]))
            
        elif(self.type==1):
            # lasso regression
            temp3=np.ones(self.W.shape)
            for i in range(temp3.shape[0]):
                if(self.W[i][0]==0):
                    temp3[i][0]=0
                elif(self.W[i][0]>0):
                    temp3[i][0]=1
                else:
                    temp3[i][0]=-1
            self.W=self.W-(self.alpha*(((temp1-temp2)/X_train.shape[0])+self.L*temp3))
                    
        else:
            # ridge 
            self.W=self.W-(self.alpha*(((temp1-temp2)/X_train.shape[0])+self.L*self.W))
        
    def fit(self, X_train, Y_train, iter=10000):
        
        self.set(X_train)
        for i in range(iter):
            self.update(X_train, Y_train)
        
    def predict(self, X_test):
        prediction=np.matmul(X_test, self.W)
        return np.reshape(prediction, (prediction.shape[0],))
    
    
    
    def get_weight(self):
        return self.W
    
class ISTA:
    def __init__(self, L):
        self.L=L
        self.theta=None
        
    def soft(self, x, T):
        if(x<=-T):
            return x+T
        elif(abs(x)<=T):
            return 0
        else:
            return x-T
        
    def soft_vector(self, theta, T):
        new_theta=np.copy(theta)
        # print(new_theta.shape)
        for i in range(new_theta.shape[0]):
            new_theta[i][0]=self.soft(theta[i][0], T)
        return new_theta
        
    def init_weights(self, X_train):
        self.theta=np.zeros((X_train.shape[1], 1))
        v, w=np.linalg.eig(np.matmul(X_train.T, X_train))
        self.alpha=np.amax(v)

    
    def update(self, X_train, Y_train):
        temp=self.theta+(np.matmul(X_train.T, (Y_train-np.matmul(X_train, self.theta))))/self.alpha
        # print(self.theta.shape)
        self.theta=self.soft_vector(temp, self.L/(2*self.alpha))
        # print(self.theta.shape)
        
    def fit(self, X_train, Y_train, iter):
        self.init_weights(X_train)
        for i in range(iter):
            self.update(X_train, Y_train)
            
    def get_weight(self):
        return self.theta

    def predict(self, X_test):
        return np.matmul(X_test, self.theta)