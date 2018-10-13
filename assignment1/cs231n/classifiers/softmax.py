import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
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
  num_classes = W.shape[1]
  num_images = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
 
  scores = W.T @ X.T
  e = np.e
    
  for i in range(num_images):
    
    fy = scores[y[i]][i]
    f = scores[:,i]
    mx = np.max(f)
    f -= mx
    fy -= mx
    loss -= np.log(e**fy/np.sum(e**f))
    
  loss /= num_images
  loss += reg * np.sum(W * W)
  scores = W.T @ X.T

  e = np.e
  scores_exp = e**scores
  probs = np.zeros_like(scores)
  
  for i in range(num_images):
    sum_of_all_scores = np.sum(scores_exp[:,i])
    probs[:, i] = scores_exp[:, i]/sum_of_all_scores
    
    probs[y[i],i] -= 1
    dW += np.outer(X[i], probs[:,i])

    
    #print(np.sum(probs[:,i])) #check if probabilites sum to 1
 
  dW /= num_images
  dW += reg*W # regularize the weights

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
    
  e = np.e
  scores = W.T @ X.T
  scores_exp = e**scores

  loss = - np.sum( np.log(scores_exp[y, range(num_train)]/np.sum(scores_exp, axis=0) ) )

  scores = W.T @ X.T
  probs = scores_exp/np.sum(scores_exp, axis=0)
  dscores = probs
  dscores[y, range(num_train)] -= 1
  dW = X.T.dot(dscores.T)


  dW /= num_train
  dW += reg * W
    
  loss /= num_train
  loss += reg * np.sum(W * W)
  #print(np.sum(probs[:,0])) check if probs columns sum to 0

  
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

