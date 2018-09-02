import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    N = X.shape[0]   # number of training examples
    C = W.shape[0]   # number of features
    D = W.shape[1]   # number of classes
   
    """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    for i in xrange(N):
        score = X[i, :].dot(W)
        score -= np.max(score)      # Correct for numerical stability
        class_score = score[y[i]]    # score for the actual label class  
        sum_exp = np.sum(np.exp(score))
        prob = np.exp(class_score) / sum_exp
        loss += -np.log(prob)    # sum the loss over examples
        for j in xrange(D):
            if j == y[i]:
                dW[:, y[i]] += (np.exp(score[j]) / sum_exp - 1) * X[i, :]
            else:
                dW[:, j] += (np.exp(score[j]) / sum_exp) * X[i, :]
    loss /= N
    loss += 0.5 * reg * np.sum(W * W)
    dW /= N
    dW += reg * W
    

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #loss=[]
  #dw = []
    
   
   #
    pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    
    """
      Softmax loss function, vectorized version.

      Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    

    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #loss = []
  #dW = []
    N = X.shape[0]      # N : number of training examples
    score = X.dot(W)     # Score = take dot product between input and weight matrix
    score = score - np.max(score, axis=1)[:, np.newaxis]  # for numeric stability
    loss = -np.sum(np.log(np.exp(score[np.arange(N), y]) / np.sum(np.exp(score), axis=1))) # sum(-log( loss over all examples)
    loss /= N              # take the average of loss
    loss += 0.5 * reg * np.sum(W * W)
    ind = np.zeros_like(score)
    ind[np.arange(N), y] = 1
    dW = X.T.dot(np.exp(score) / np.sum(np.exp(score), axis=1, keepdims=True) - ind)
    dW /= N
    dW += reg * W  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    return loss, dW

