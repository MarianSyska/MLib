from matplotlib import pyplot as plt
import numpy as np

def main():
    shape_l1 = (3, 4)
    shape_l2 = (4, 2)

    weights_l1 = (np.random.rand(*shape_l1) * 2 - 1)  * np.sqrt( 6 / (shape_l1[0] + shape_l1[1]))
    weights_l2 = (np.random.rand(*shape_l2) * 2 - 1)  * np.sqrt( 6 / (shape_l2[0] + shape_l2[1]))

    x = np.random.rand(1, shape_l1[0])
    y = np.random.rand(1, shape_l2[1])
    print(f"X: \n {x}")
    print(f"Y: \n {y}")
    
    
def infer_net(x, y, weights_l1, weights_l2):
    # layer 1
    l1_out = x.dot(weights_l1)
    print(f"Weights L1: \n {weights_l1}")
    print(f"L1 Out: \n {l1_out}")

    # layer 2
    pred = l1_out.dot(weights_l2)
    print(f"Weights L2: \n {weights_l2}")
    print(f"Pred: \n {pred}")

    loss = np.mean((pred - y) ** 2)
    print(f"Loss: \n {loss}")
    return l1_out, pred, loss

def calc_grad(weights_l2, x, y, l1_out, pred):
    grad_loss = pred - y

    print(f"Gradient Loss: \n {grad_loss}")

    grad_weights_l2 = np.einsum('ni,nj->ij', l1_out, grad_loss)

    print(f"Gradient Weights L2 : \n {grad_weights_l2}")

    grad_l1_out =  grad_loss.dot(weights_l2.T)

    print(f"Gradient L1 Out: \n {grad_l1_out}")

    grad_weights_l1 = np.einsum('ni,nj->ij', x, grad_l1_out)

    print(f"Gradient Weights L2 : \n {grad_weights_l2}")
    return grad_weights_l1, grad_weights_l2

def step(grad_weights_l1, grad_weights_l2, weights_l1, weights_l2):
    step_weights_l1 = grad_weights_l1 * 0.1

    print(f"Step Weights L1: \n {step_weights_l1}")

    step_weights_l2 = grad_weights_l2 * 0.1

    print(f"Step Weights L2: \n {step_weights_l2}")

    weights_l1 -= step_weights_l1
    weights_l2 -= step_weights_l2
    return weights_l1, weights_l2

