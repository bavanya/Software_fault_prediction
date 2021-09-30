# Software_fault_prediction

Idea 1:
Treat the faulty data points i.e code files from which the OO metrics were extracted as anomalies and perform isolation forest to predict them.

Idea 2:
Apply deep learning treating it as a regression problem considering the OO metrics as the potential features.

Idea 3:
Since OO metrics do not capture the semantic information and syntax of the code, generate ASTs from the code files and extract important keywords and build regression models to predict the number of bugs in them.
