# NCAT

## Description
A Python library for implementing NCAT models and data generation.

---
## Installation
Install using pip:
```bash
pip install NCAT
```
---
## **NCAT: Neural Network Distilled from CATBoost**
**NCAT** is an ensemble of shallow neural network (NN) models distilled from CATBoost, a state-of-the-art model for tabular data. CATBoost is widely recognized for its fast, scalable performance and strong adaptability, making it one of the leading algorithms for tabular data tasks.

However, despite its strengths, CATBoost has certain limitations in industrial applications. These include:

- Limited adaptability to online data streams.
- Minimal tuning flexibility after deployment.
- The computational complexity of its ensemble structure.

 To address these challenges, model distillation was introduced. Distillation creates a simplified "student" model that mimics the predictions of a more complex "teacher" model while retaining similar accuracy.

NCAT applies this concept by using CATBoost as the teacher model and training multiple shallow neural networks (NNs) to replicate its prediction behavior. CATBoost typically comprises more than 1,000 decision trees, each with two layers and symmetric nodes. The model's final prediction is the cumulative sum of the leaf node values across all trees.

To replicate this behavior, NCAT groups trees into clusters based on their feature usage using k-NN clustering. Each cluster is then represented by a shallow NN with two low-dimensional layers. These NNs are trained to mimic the predictions of their respective tree clusters. Ultimately, the ensemble of these shallow NNs replaces the 1,000-tree ensemble, offering a compact and efficient approximation of CATBoost.

---

## **Data Augmentation for Distillation**

To build the distillation dataset, **Gibbs sampling** can be used to augment the original tabular data. By sampling each feature conditionally based on its probability distribution given the other features, additional data can be generated.

In industrial applications, where tabular datasets often have high dimensionality but limited samples, this augmentation process is essential. The enriched dataset ensures the student model is trained effectively to approximate the teacher model’s predictions while maintaining scalability and efficiency.

This approach allows NCAT to bridge the gap between the robust performance of CATBoost and the practical requirements of real-world machine learning applications, such as online adaptation and resource-constrained environments.

---

[1] Fakoor, Rasool, et al. "Fast, accurate, and simple models for tabular data via augmented distillation." Advances in Neural Information Processing Systems 33 (2020): 8671-8681.

[2] Ke, Guolin, et al. "DeepGBM: A deep learning framework distilled by GBDT for online prediction tasks." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019.

[3] Reinders, Christoph, and Bodo Rosenhahn. "Neural Random Forest Imitation." arXiv preprint arXiv:1911.10829 (2019).

[4] Hollmann, Noah, et al. "Tabpfn: A transformer that solves small tabular classification problems in a second." arXiv preprint arXiv:2207.01848 (2022).

[5] Bornschein, Jorg, Francesco Visin, and Simon Osindero. "Small data, big decisions: Model selection in the small-data regime." International conference on machine learning. PMLR, 2020.

[6] Peng, Yuting. "An Introduction of Prediction Models From the View of Integration Between Basic Models." 2021 2nd International Conference on Computing and Data Science (CDS). IEEE, 2021.

[7] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," NIPS Deep Learning and Representation Learning Workshop, 2015.

[8] Kingma, Diederik P. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
