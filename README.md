# Towards Certified Unlearning for Deep Neural Networks
The code is associated with *Towards Certified Unlearning for Deep Neural Networks* (ICML 2024). 


## 1.Environment

Experiments are performed on an Nvidia RTX A6000 with Cuda 11.3. 

Notice: Cuda is enabled for default settings.


## 2.Usage
We have three datasets for experiments, namely MNIST, CIFAR-10, and SVHN. Refer to Appendix D for the hyperparameter settings.


### 2.1 Training the Original Model

Run

```python train.py```

Hyperparameter settings can be found in the Appendix. After running this code file, an original model will be saved in the ```./model/``` directory.


### 2.2 Obtaining the Unlearned Model

Run

```python unlearn.py```

Hyperparameter settings can be found in the Appendix. After running this code file, an unlearned model will be saved in the ```./model/``` directory.


### 2.3 Testing the Unlearned Model

Change the PATH_unl variable in the ```test_unlearn.py``` file to the path of the unlearned model to be evaluated.

Run 

```python test_unlearn.py```

For the Membership Inference Attack evaluation, change the PATH_unl variable in the ```test_unlearn.py``` file to the path of the unlearned model to be evaluated.

Run

```python attack.py```
