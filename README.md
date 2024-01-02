# GR1
Báo Cáo Cuối Kì

Nguyễn Tiến Thành - 20215243

Concept of Machine Learning & Deep Learning algorithms

I. A brief of Machine Learning & Deep Learning

1. Machine Learning (ML):
   
- Definition: Machine learning is a branch of AI that enables computers to learn from data and improve their performance on a task over time without being explicitly programmed.

- Types of ML:
  - Supervised Learning: The algorithm is trained on a labeled dataset, where the input data is paired with corresponding output labels. The model learns to map inputs to outputs.
  - Unsupervised Learning: The algorithm is given unlabeled data and must find patterns or structure on its own.
  - Reinforcement Learning: The model learns through trial and error by interacting with an environment and receiving feedback in the form of rewards or penalties.
  
- Applications:
   - Classification: Assigning labels to input data.
   - Regression: Predicting numerical values.
   - Clustering: Identifying inherent patterns or groups in data.
   - Recommendation Systems: Recommending items based on user behavior.
  
- Algorithms:

Common algorithms include linear regression, decision trees, support vector machines, k-nearest neighbors, and more.

2. Deep Learning (DL):
- Definition: Deep learning is a subset of machine learning that involves neural networks with multiple layers (deep neural networks). These networks are capable of learning intricate hierarchical representations from data.
- Neural Networks:
   - Artificial Neurons: Mimic the basic functioning of biological neurons.
   - Layers: Input layer, hidden layers, and output layer.
   - Deep Networks: Multiple hidden layers enable the extraction of complex features.
  
- Types of DL Architectures:

   - Feedforward Neural Networks: Information flows in one direction.
   - Recurrent Neural Networks (RNN): Suitable for sequence data due to recurrent connections.
   - Convolutional Neural Networks (CNN): Effective for image and spatial data due to convolutional layers.
- Applications:

   - Image and Speech Recognition: Achieving state-of-the-art results in tasks like image classification and speech recognition.
   - Natural Language Processing (NLP): Applications in text generation, sentiment analysis, and language translation.
   - Generative Models: Creating new data instances, as seen in generative adversarial networks (GANs).

- Training:

Training deep networks often involves the use of large datasets and powerful computational resources

II. Type of Machine Learning algorithms

1. Supervised learning
In supervised learning, the machine is taught by example. The operator provides the machine learning algorithm with a known dataset that includes desired inputs and outputs, and the algorithm must find a method to determine how to arrive at those inputs and outputs. While the operator knows the correct answers to the problem, the algorithm identifies patterns in data, learns from observations and makes predictions. The algorithm makes predictions and is corrected by the operator – and this process continues until the algorithm achieves a high level of accuracy/performance.

Under the umbrella of supervised learning fall: Classification, Regression and Forecasting.

- Classification: In classification tasks, the machine learning program must draw a conclusion from observed values and determine to what category new observations belong. For example, when filtering emails as ‘spam’ or ‘not spam’, the program must look at existing observational data and filter the emails accordingly.
  
- Regression: In regression tasks, the machine learning program must estimate – and understand – the relationships among variables. Regression analysis focuses on one dependent variable and a series of other changing variables – making it particularly useful for prediction and forecasting.
  
- Forecasting: Forecasting is the process of making predictions about the future based on the past and present data, and is commonly used to analyse trends.

2. Semi-supervised learning
   
Semi-supervised learning is similar to supervised learning, but instead uses both labelled and unlabelled data. Labelled data is essentially information that has meaningful tags so that the algorithm can understand the data, whilst unlabelled data lacks that information. By using this
combination, machine learning algorithms can learn to label unlabelled data.

3. Unsupervised learning
   
Here, the machine learning algorithm studies data to identify patterns. There is no answer key or human operator to provide instruction. Instead, the machine determines the correlations and relationships by analysing available data. In an unsupervised learning process, the machine learning algorithm is left to interpret large data sets and address that data accordingly. The algorithm tries to organise that data in some way to describe its structure. This might mean grouping the data into clusters or arranging it in a way that looks more organised.

As it assesses more data, its ability to make decisions on that data gradually improves and becomes more refined.

Under the umbrella of unsupervised learning, fall:

- Clustering: Clustering involves grouping sets of similar data (based on defined criteria). It’s useful for segmenting data into several groups and performing analysis on each data set to find patterns.
  
- Dimension reduction: Dimension reduction reduces the number of variables being considered to find the exact information required.

4. Reinforcement learning
   
Reinforcement learning focuses on regimented learning processes, where a machine learning algorithm is provided with a set of actions, parameters and end values. By defining the rules, the machine learning algorithm then tries to explore different options and possibilities, monitoring and evaluating each result to determine which one is optimal. Reinforcement learning teaches the machine trial and error. It learns from past experiences and begins to adapt its approach in response to the situation to achieve the best possible result.
