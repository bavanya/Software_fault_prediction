# Software_fault_prediction

## Idea 1:

Treat the faulty data points i.e code files from which the OO metrics were extracted as anomalies.

Concepts tried using this approach:
1. Isolation forest.
2. Autoencoder.
3. Different clustering algorithms.

## Idea 2:

Apply deep learning treating it as a regression problem considering the OO metrics as the potential features.

Used different models like LSTMs, RNNs, MLPs, CNNs, GRUs and tried different architectures.
Also tried classical machine learning models for regression like linear regression, lasso regression, poisson regression and compared the performance with deep learning models.
Tried their ensembles and ideas like averaging, majority voting etc for better results.

Generated results for two cases:
1. Models trained on cummulative previous versions and tested on the latest version.
2. Models trained on immediate previous version and tested on the next version.

Adam optimizer, drop out, PCA, random oversampling, SMOTE, SVD are are the additional concepts used in the implementation of this idea. 

## Idea 3:

This idea was inspired by the paper- [An Improved CNN Model for Within-Project
Software Defect Prediction by Cong Pan, Minyan Lu, Biao Xu and Houleng Gao](https://www.mdpi.com/2076-3417/9/10/2138)

Since OO metrics do not capture the semantic information and syntax of the code, generate ASTs from the code files and extract important keywords and build regression models to predict the number of bugs in them.

To generate input vectors for the models using the keywords, we can try keyword-index matching. 
If we have pretrained models like bert specifically trained on language representation in JAVA codes, we could use them to obtain vectors which not only maintain the sequence of the keywords but also their semantic meanings.

CNN and RNN based models can be used here considering the sequential nature of the inputs. Similar to how these models are chosen for text classification problems.

The fundamental and the most important step for this approach is obtaining code embeddings which can be used for software bug prediction.
This is called code representation learning and we can follow the footsteps of BioBERT to generate a similar pretrained model for our usecase.

**Two approaches of using the embeddings**
1. Add an embedding layer to generate a higher dimensional matrix before the layers of CNN/RNN etc.
2. Consider the embeddings as individual features and train a CNN/RNN model on the resultant high dimensional data. 

**Intriguing point under this idea**

To what extent will the the code representations of common words change for different languages? For eg. Will the representations for words like if, else, while etc in the vector spaces of different corpus(different languages) be close to each other?(Considering that the vector spaces have common origin)

**Helpful resources related to this idea:**

[
Blog on representation learning](https://medium.com/@aganirbanghosh007/representation-learning-a-review-and-perspectives-ea923618d79c)

**Question**: 
1. Which idea do you think will give better results?
2. Which approach in idea 3 will give better results?
3. Will we face the curse of dimensionality in the second approach of idea 3?
