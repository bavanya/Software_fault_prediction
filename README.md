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
