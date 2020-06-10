# kaggle_credit
Credit Card fraud analysis
Dataset link: https://www.kaggle.com/mlg-ulb/creditcardfraud

# Goal
Compare 3 different models: Random Forests, Adaboost and Gradient Boost
using the dataset linked above.
The number of estimators, the maximum depth (for Random Forests only) and 
the learning rate (for Adaboost and Gradient boost only) were varied over 
a range of different values. The other parameters were left as the default
of the scikit-learn library. The three important metrics are accuracy (maximize
correct predictions), number of false negative (avoid missing the detection
of a fraudulent transaction) and Area Under the Curve (AUC).

# Key takeaways
1) The dataset is strongly unbalanced: 99.83% of the transactions are not fraudulent.
Training on the full dataset takes a lot of time, only Random Forest and Adaboost
were tested, Random Forest giving better results in term of accury, number of false
negative and AUC, probably due to bagging.

2) When a subset of not fraudulent transaction is taken in order to have a balanced 
dataset (50% of fraudulent transactions), the 3 models were tested.
Adaboost gave the best results, again in term of accury, number of false
negative and AUC, achieving at best perfect accuracy on the validation set.
