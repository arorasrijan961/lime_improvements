import numpy as np

def sample(num, shape):
    
    n=shape[0]
    m=shape[1]
    
    result=np.zeros((num, m))
    for i in range(num):
        result[i]=np.random.normal((m, ))
        
    return result    

def get_weight(x, samples, width=None):
    n=samples.shape[0]
    m=samples.shape[1]
    if(width==None):
        width=np.sqrt(m)*0.75
    result=np.zeros((n,))
    for i in range(n):
        norm=np.linalg.norm(x-samples[i])
        result[i]=np.sqrt(np.exp(-(norm)**2/(width**2)))
        
    return result


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
    
class Modified_ISTA:
    def __init__(self, L, p, epsilon):
        self.L=L
        self.theta=None
        self.p=p
        self.epsilon=epsilon
        
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
            new_theta[i][0]=self.soft(theta[i][0], T[1])
        return new_theta
        
    def init_weights(self, X_train):
        self.theta=np.zeros((X_train.shape[1], 1))
        v, w=np.linalg.eig(np.matmul(X_train.T, X_train))
        self.alpha=np.amax(v)

    def pnorm(self):
        result=[]
        for i in range(self.theta.shape[0]):
            result.append((abs(self.theta[i][0])+self.epsilon)**(self.p-1))
            
        return result
    
    def update(self, X_train, Y_train):
        temp=self.theta+(np.matmul(X_train.T, (Y_train-np.matmul(X_train, self.theta))))/self.alpha
        # print(self.theta.shape)
        pnorm=self.pnorm()
        l=self.pnorm()
        for i in range(len(l)):
            l[i]=self.L*l[i]/(2*self.alpha)
        self.theta=self.soft_vector(temp, l)
        # print(self.theta)
        # print(self.theta.shape)
        
    def fit(self, X_train, Y_train, iter):
        self.init_weights(X_train)
        for i in range(iter):
            self.update(X_train, Y_train)
            
    def get_weight(self):
        return self.theta

    def predict(self, X_test):
        return np.matmul(X_test, self.theta)
