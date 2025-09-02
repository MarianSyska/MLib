import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_sin():
    # Create training and testing data
    batch_size = 100
    
    x_train = np.linspace(-2*np.pi, 2*np.pi, num=1000).reshape(-1, batch_size, 1)
    x_test = np.random.rand(1000, 1)* 4 * np.pi - 2 * np.pi
    
    y_train = np.sin(x_train)
    y_test  = np.sin(x_test)
    
    return (x_train, y_train, x_test, y_test, batch_size)

def load_mnist():
            
        # Download the MNIST dataset
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto', data_home='data', cache=True)
        x, y = mnist.data, mnist.target
        
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


        # Adjust for bactch size
        batch_size = 64

        dest_len = ((len(x_train) // batch_size) * batch_size)
        x_train = x_train[:dest_len] 
        y_train = y_train[:dest_len] 
        
        
        x_train = x_train.reshape(-1, batch_size, 28*28)
        y_train = y_train.reshape(-1, batch_size, 1)
        
        
        dest_len = ((len(x_test) // batch_size) * batch_size)
        x_test = x_test[:dest_len] 
        y_test = y_test[:dest_len]
        
        x_test = x_test.reshape(-1, batch_size, 28*28)
        y_test = y_test.reshape(-1, batch_size, 1)
        
        
        return (x_train, y_train, x_test, y_test, batch_size)

    

