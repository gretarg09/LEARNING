Binary cross-entropy is a loss function that is used in in binary classification tasks.

## Goal
The goal of binary cross-entrophy loss is to measure the dissimilarity between the model's
predicted probabilities and the true class labels. The predicted probabilities refer to the 
output of the model, which should ideally be a probability that each sample belongs to class
1.

## Math

The mathimatical formula for Boinary Cross-Entropy loss for a single data point:

```python
loss = -[y * log(p) + (1 - y) * log(1 - p)]
```

In this formula:

* y is the true class label (0 or 1).
* p is the model's predicted probability for the class 1.
* log is the natural logarithm.

If y is 1, then the loss is -log(p), which goes to infinity as p goes to 0.
In other words, the loss is very high if the model thinks the sample does
not belong to class 1 when it actually does.

If y is 0, then the loss is -log(1 - p), which goes to infinity as p goes to 1.
This means the loss is very high if the model thinks the sample does belong to 
class 1 when it actually does not.

Basically, we use the fact that -log(p) goes to infinity when p approaches 0. By using the 
crossentropy we penalize the model when it is confident about the wrong result. The penalization
is exponential.

## Design
The Binary Cross-Entropy loss is designed to penalize models that are confident and wrong.

In practice, the Binary Cross-Entropy loss is calculated for each sample in the dataset, and the average loss is used to update the model's weights.
