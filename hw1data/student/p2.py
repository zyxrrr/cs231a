# CS231A Homework 0, Problem 2
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib


def main():
    # ===== Problem 2a =====
    # Define Matrix M and Vectors a,b,c in Python with NumPy

    M, a, b, c = None, None, None, None

    # BEGIN YOUR CODE HERE
    M=np.array([[1,2,3],[4,5,6],[7,8,9],[0,2,2]])
    a=np.array([[1],[1],[0]])
    b=np.array([[-1],[2],[5]])
    c=np.array([[0],[2],[3],[2]])
    # END YOUR CODE HERE

    # ===== Problem 2b =====
    # Find the dot product of vectors a and b, save the value to aDotb

    aDotb = None

    # BEGIN YOUR CODE HERE
    aT=a.T
    aDotb=aT.dot(b)
    print 'aDotb :\n',aDotb
    # END YOUR CODE HERE

    # ===== Problem 2c =====
    # Find the element-wise product of a and b

    # BEGIN YOUR CODE HERE
    aMulb=np.multiply(a,b)
    print 'the element-wise product of a and b :\n',aMulb
    # END YOUR CODE HERE

    # ===== Problem 2d =====
    # Find (a^T b)Ma

    # BEGIN YOUR CODE HERE
    resD=aDotb*M.dot(a)
    print '(a^T b)Ma :\n',resD
    # END YOUR CODE HERE

    # ===== Problem 2e =====
    # Without using a loop, multiply each row of M element-wise by a.
    # Hint: The function repmat() may come in handy.

    newM = None

    # BEGIN YOUR CODE HERE
    aTM=np.matlib.repmat(aT,4,1)
    newM=np.multiply(M,aTM)
    print 'multiply each row of M element-wise by a :\n',newM
    # END YOUR CODE HERE

    # ===== Problem 2f =====
    # Without using a loop, sort all of the values 
    # of M in increasing order and plot them.
    # Note we want you to use newM from e.

    # BEGIN YOUR CODE HERE
    sortedM=np.sort(newM, axis=None)
    print'sorted newM :\n',sortedM
    
    x=np.linspace(1,12,12)
    plt.scatter(x,sortedM)
    plt.show()
    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
