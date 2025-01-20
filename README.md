# 📚 Classifier study
![Language](https://img.shields.io/badge/python-3670A0?style=for-the-badge)

```math
\nabla_w \mathcal{L}(x, y, w) = (\hat{f}(x; w) - y) \left[ 1, x_{1},..., x_{p} \right]
```

Project undertaken as part of the ELEN0062 course given by Pr. Geurts and Pr. Wehenkel.<br>
Grades : 18.55/20<br>

## Quick summary

The project is composed of :
- A study of the performance of the Decision Tree (from scikit-learn)
- A study of the performances of the KNN (from scikit-learn)
- A study of the performances of the Perceptron (Custom implementation) and the derivation of the gradient descent for the cross-entrpoy loss
- Comparison of the three classifiers

## Content

- data.py : Generates training and testing data
- dt.py : Experiments on the decision tree
- knn.py : Experiments on the KNN
- perceptron.py : Implementation and experiments on the perceptron
- plot.py : Plotting functions

## Credits
- [Simon Gardier](https://github.com/simon-gardier) (Co-author)
- Camille Trinh (Co-author)
- [Sacha Lewin](https://iml.isach.be/) (data.py & plot.py author) 


