# Software_fault_prediction

This work has been implemented over the course of 2 semesters. The findings, report and slides for the work done in the first semester is present in [Phase-1_Documentation](https://github.com/bavanya/Software_fault_prediction/tree/main/Phase-1_Documentation). Please refer to the [Final_Documentation](https://github.com/bavanya/Software_fault_prediction/tree/main/Final_documentation) folder for the conclusion of the work.

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


## Few questions to think about:
1. Which idea do you think will give better results?
2. Which approach in idea 3 will give better results?
3. Will we face the curse of dimensionality in the second approach of idea 3?

## Related papers:
[1] Software engineering book tenth edition. Ian Sommerville.

[2] Pan, Lei & Lu, Jingyan & Xu, Longfei & Gao,. (2019). An Improved CNN Model for Within-Project Software Defect Prediction. Applied Sciences. 9. 2138. 10.3390/app9102138. 

[3] Shyam R. Chidamber, Chris F. Kemerer: A Metrics suite for Object Oriented design. M.I.T. Sloan School of Management E53-315. 1993.

[4] Henderson-Sellers, B., Object-oriented metrics : measures of complexity, Prentice-Hall, pp.142-147, 1996.

[5]  Martin, R.C.: Agile Software Development: Principles, Patterns, and Practices. Alant Apt Series. Prentice Hall, Upper Saddle River, NJ, USA (2002)

[6] P. K. Goyal and G. Joshi, "QMOOD metric sets to assess quality of Java program," 2014 International Conference on Issues and Challenges in Intelligent Computing Techniques (ICICT), 2014, pp. 520-533, doi: 10.1109/ICICICT.2014.6781337.

[7] Tang M.H., Kao M.H., Chen M.H.: “An Empirical study on object-oriented Metrics” Software Metrics Symposium, 1999. IEEE Computer, Proceedings. PP. 242-249(1999). 

[8] M. Nevendra and P. Singh, "Software bug count prediction via AdaBoost.R-ET," 2019 IEEE 9th International Conference on Advanced Computing (IACC), 2019, pp. 7-12, doi: 10.1109/IACC48062.2019.8971588.

[9] X. Yang and W. Wen, “Ridge and Lasso Regression Models for Cross-Version Defect Prediction,” IEEE Trans. Reliab., vol. 67, no. 3, pp. 885–896, 2018.

[10] L. Yu, “Using Negative Binomial Regression Analysis to Predict Software Faultsௗ: A Study of Apache Ant,” I.J. Inf. Technol. Comput. Sci., vol. 4, no. 8, pp. 63–70, 2012.

[11] Santosh Singh Rathore and Sandeep Kumar. 2016. A Decision Tree Regression based Approach for the Number of Software Faults Prediction. SIGSOFT Softw. Eng. Notes 41, 1 (January 2016), 1–6. DOI:https://doi.org/10.1145/2853073.2853083

[12] Sushant Kumar Pandey, Anil Kumar Tripathi, BCV-Predictor: A bug count vector predictor of a successive version of the software system, Knowledge-Based Systems,
Volume 197, 2020, 105924, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2020.105924.
(https://www.sciencedirect.com/science/article/pii/S0950705120302604)

[13] Dam, Hoa & Pham, Trang & Ng, Shien & Tran, Truyen & Grundy, John & Ghose, Aditya & Kim, Taeksu & Kim, Chul-Joo. (2018). A deep tree-based model for software defect prediction. 

[14] H. K. Dam, T. Tran, T. Pham, S. W. Ng, J. Grundy and A. Ghose, "Automatic Feature Learning for Predicting Vulnerable Software Components," in IEEE Transactions on Software Engineering, vol. 47, no. 1, pp. 67-85, 1 Jan. 2021, doi: 10.1109/TSE.2018.2881961.

[15] Akimova, Elena & Bersenev, Alexander & Deikov, Artem & Kobylkin, Konstantin & Konygin, Anton & Mezentsev, Ilya & Misilov, Vladimir. (2021). A Survey on Software Defect Prediction Using Deep Learning. Mathematics. 9. 1180. 10.3390/math9111180. 

[16] White, M.; Vendome, C.; Linares-Vasquez, M.; Poshyvanyk, D. Toward Deep Learning Software Repositories. In Proceedings of the 2015 IEEE/ACM 12th Working Conference on Mining Software Repositories (MSR),
Florence, Italy, 16–17 May 2015; pp. 334–345.

[17] Fan, Guisheng & Diao, Xuyang & Yu, Huiqun & Yang, Kang & Chen, Liqiong. (2019). Software Defect Prediction via Attention-Based Recurrent Neural Network. Scientific Programming. 2019. 1-14. 10.1155/2019/6230953. 

[18] Li, J.; He, P.; Zhu, J.; Lyu, M.R. Software Defect Prediction via Convolutional Neural Network. In Proceedings
of the 2017 IEEE International Conference on Software Quality, Reliability and Security (QRS), Prague, Czech Republic, 25–29 July 2017; pp. 318–328.

[19] J. Li, P. He, J. Zhu and M. R. Lyu, "Software Defect Prediction via Convolutional Neural Network," 2017 IEEE International Conference on Software Quality, Reliability and Security (QRS), 2017, pp. 318-328, doi: 10.1109/QRS.2017.42.

