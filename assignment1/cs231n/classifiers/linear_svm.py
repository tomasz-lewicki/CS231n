import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0
  for i in range(num_train):
    num_incorrect = 0
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        num_incorrect += 1
        dW[:, j] += X[i]
        
    dW[:, y[i]] -= num_incorrect*X[i]
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += reg*W # regularize the weights

  # Add regularization to the loss (assuming lambda=0.5)
  loss += reg * np.sum(W * W)


    
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_half_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]
  wrong_pred = np.zeros(num_train)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
    
  for i in range(num_train):
    
    scores = W.T.dot(X[i])
    margins = np.maximum(0, (scores - scores[y[i]] +1))
    margins[y[i]] = 0
    loss += np.sum(margins)
    wrong_pred[i] = np.sum(margins != 0)
    
    dW[:, margins>0] += np.expand_dims(X[i], axis=1)
    
    dW[:,y[i]] -= (wrong_pred[i]) * X[i]
    
  loss /= num_train
  loss += reg * np.sum(W * W)
      

    
  dW /= num_train
  dW += reg*W # regularize the weights
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW



def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]
  wrong_pred = np.zeros(num_train)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  scores = W.T @ X.T
  margins = np.maximum(0, (scores - scores[:, y] +1))
  print(scores[:, y].shape)
  print((scores - scores[:, y])[:][0])
  margins[y, np.arange(num_train)] = 0
  loss = np.sum(margins)
  
    
  margins[y] = 0
#     loss += np.sum(margins)
#     wrong_pred[i] = np.sum(margins != 0)
    
#     dW[:, margins>0] += np.expand_dims(X[i], axis=1)
    
#     dW[:,y[i]] -= (wrong_pred[i]) * X[i]
    
    
  loss /= num_train
  loss += reg * np.sum(W * W)
      
  dW /= num_train
  dW += reg*W # regularize the weights
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
