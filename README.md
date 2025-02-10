# Takeaways
## Zero weights become rarer over training, consistently across act fxns
## Elementwise
## Weight norm increases over training
## Training loss is pretty consistent across activation functions
## Avg activation is essentially constant and a function of the activation function
### Relu > Gelu > Silu > trainable_erf > vecgelu
### Vector activation functions have very small avg activation
# Only relu creates truly dead neurons
# 75% of neurons don't receive 

What if we make this much deeper and decrease batch size accordingly

Gradient norms start at end layers
Then get higher in previous layers over course of training
vecgelu activation norm flattens out
Gelu gradient stuck in final layers
Vecgelu has relatively even gradient magnitude, high at beginning layer and decreases
  gets more even over run