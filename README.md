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
  
- Algorithms: Common algorithms include linear regression, decision trees, support vector machines, k-nearest neighbors, and more.

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

- Training: Training deep networks often involves the use of large datasets and powerful computational resources

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

III. Some common and popular machine learning algorithms

1. K-Nearest Neighbour

- Defination: K-Nearest Neighbours is one of the most basic yet essential classification algorithms in Machine Learning. It belongs to the supervised learning domain and finds intense application in pattern recognition, data mining, and intrusion detection. It is widely disposable in real-life scenarios since it is non-parametric, meaning, it does not make any underlying assumptions about the distribution of data (as opposed to other algorithms such as GMM, which assume a Gaussian distribution of the given data). We are given some prior data (also called training data), which classifies coordinates into groups identified by an attribute.

- Distance Metrics Used in KNN Algorithm
As we know that the KNN algorithm helps us identify the nearest points or the groups for a query point. But to determine the closest groups or the nearest points for a query point we need some metric. For this purpose, we use below distance metrics:

  - Euclidean Distance
    
  - Manhattan Distance
    
  - Minkowski Distance

   a.	Euclidean Distance

This is nothing but the cartesian distance between the two points which are in the plane/hyperplane. Euclidean distance can also be visualized as the length of the straight line that joins the two points which are into consideration. This metric helps us calculate the net displacement done between the two states of an object.

![image](https://github.com/thanhite7/GR1/assets/96159427/c920f9d0-5092-4515-b32a-bb528a8b4f46)


   b.	Manhattan Distance
   
This distance metric is generally used when we are interested in the total distance traveled by the object instead of the displacement. This metric is calculated by summing the absolute difference between the coordinates of the points in n-dimensions.

![image](https://github.com/thanhite7/GR1/assets/96159427/1611ae0c-9755-49dd-92e5-9ce57ecbe2f8)


   c.	Minkowski Distance
   
We can say that the Euclidean, as well as the Manhattan distance, are special cases of the Minkowski distance.

![image](https://github.com/thanhite7/GR1/assets/96159427/36962a0f-f973-43ac-be87-f15c19570346)

From the formula above we can say that when p = 2 then it is the same as the formula for the Euclidean distance and when p = 1 then we obtain the formula for the Manhattan distance. The above-discussed metrics are most common while dealing with a Machine Learning problem but there are other distance metrics as well like Hamming Distance which come in handy while dealing with problems that require overlapping comparisons between two vectors whose contents can be boolean as well as string values.


- Applications of the KNN Algorithm:
   - Data Preprocessing – While dealing with any Machine Learning problem we first perform the EDA part in which if we find that the data contains missing values then there are multiple imputation methods are available as well. One of such method is KNN Imputer which is quite effective ad generally used for sophisticated imputation methodologies.
     
   - Pattern Recognition – KNN algorithms work very well if you have trained a KNN algorithm using the MNIST dataset and then performed the evaluation process then you must have come across the fact that the accuracy is too high.
     
   - Recommendation Engines – The main task which is performed by a KNN algorithm is to assign a new query point to a pre-existed group that has been created using a huge corpus of datasets. This is exactly what is required in the recommender systems to assign each user to a particular group and then provide them recommendations based on that group’s preferences.

- Advantages of the KNN Algorithm:
  
   - Easy to implement as the complexity of the algorithm is not that high.
     
   - Adapts Easily – As per the working of the KNN algorithm it stores all the data in memory storage and hence whenever a new example or data point is added then the algorithm adjusts itself as per that new example and has its contribution to the future predictions as well.
     
   - Few Hyperparameters – The only parameters which are required in the training of a KNN algorithm are the value of k and the choice of the distance metric which we would like to choose from our evaluation metric.

- Disadvantages of the KNN Algorithm:

   - Does not scale – As we have heard about this that the KNN algorithm is also considered a Lazy Algorithm. The main significance of this term is that this takes lots of computing power as well as data storage. This makes this algorithm both time-consuming and resource exhausting.

   - Curse of Dimensionality – There is a term known as the peaking phenomenon according to this the KNN algorithm is affected by the curse of dimensionality which implies the algorithm faces a hard time classifying the data points properly when the dimensionality is too high.

   - Prone to Overfitting – As the algorithm is affected due to the curse of dimensionality it is prone to the problem of overfitting as well. Hence generally feature selection as well as dimensionality reduction techniques are applied to deal with this problem.
     
2. K-mean Clustering
   - Definition: K-means clustering is a partitioning technique in unsupervised machine learning that aims to group similar data points into K clusters.
     
We are given a data set of items, with certain features, and values for these features (like a vector). The task is to categorize those items into groups. To achieve this, we will use the K-means algorithm; an unsupervised learning algorithm. ‘K’ in the name of the algorithm represents the number of groups/clusters we want to classify our items into.

The algorithm will categorize the items into k groups or clusters of similarity. To calculate that similarity, we will use the Euclidean distance as a measurement.

- The algorithm works as follows:  
   - First, we randomly initialize k points, called means or cluster centroids.
     
   - We categorize each item to its closest mean and we update the mean’s coordinates, which are the averages of the items categorized in that cluster so far.
     
   - We repeat the process for a given number of iterations and at the end, we have our clusters.

The “points” mentioned above are called means because they are the mean values of the items categorized in them. To initialize these means, we have a lot of options. An intuitive method is to initialize the means at random items in the data set. Another method is to initialize the means at random values between the boundaries of the data set.

   ![image](https://github.com/thanhite7/GR1/assets/96159427/9e8313e7-2fe1-4d74-9dcd-9ba61c6abd79)
- Advantages:
  
   - Simplicity and Speed: K-means is computationally efficient and relatively simple to implement, making it suitable for large datasets and quick exploratory data analysis.
     
   - Scalability: The algorithm scales well with the number of data points, making it applicable to datasets with a large number of observations.
     
   - Versatility: K-means can be applied to various types of data, and it doesn't assume any specific distribution of the features.
     
   - Easy Interpretation: The results of K-means are easy to interpret, as each data point is assigned to a specific cluster, and the cluster centroids provide a representative summary of the cluster.
     
   - Applicability to Numerical Data: K-means is well-suited for numerical data and works effectively when the clusters have a spherical shape.
- Disadvantages:
  
   - Sensitivity to Initial Centroids: The algorithm's performance can be sensitive to the initial placement of centroids, potentially leading to suboptimal solutions.
     
   - Assumption of Spherical Clusters: K-means assumes that clusters are spherical and equally sized, which may not hold true for datasets with non-spherical or unevenly sized clusters.
     
   - Impact of Outliers: Outliers can significantly affect the centroids and, consequently, the cluster assignments in K-means. The algorithm is sensitive to anomalies in the data.
     
   - Requires Predefined Number of Clusters (K): The user must specify the number of clusters before running the algorithm, and choosing an inappropriate value for K can result in suboptimal clustering.
     
   - Limited to Euclidean Distance: K-means relies on Euclidean distance, which might not be suitable for datasets with features of different scales or non-numerical data.

- Applications:
   - Customer Segmentation: Businesses use K-means to group customers based on purchasing behavior, helping in targeted marketing strategies.
     
   - Image Compression: In image processing, K-means can be used to compress images by reducing the number of colors while preserving key features.
     
   - Anomaly Detection: K-means can identify unusual patterns or outliers in data, making it useful for anomaly detection in various domains, such as fraud detection.
     
3. Decision Tree
   
- Definition: Random forest is a commonly-used machine learning algorithm trademarked by Leo Breiman and Adele Cutler, which combines the output of multiple decision trees to reach a single result. Its ease of use and flexibility have fueled its adoption, as it handles both classification and regression problems.

- Decision Tree Terminologies:

   - Root Node: It is the topmost node in the tree,  which represents the complete dataset. It is the starting point of the decision-making process.
   - Decision/Internal Node: A node that symbolizes a choice regarding an input feature. Branching off of internal nodes connects them to leaf nodes or other internal nodes.
   - Leaf/Terminal Node: A node without any child nodes that indicates a class label or a numerical value.
   - Splitting: The process of splitting a node into two or more sub-nodes using a split criterion and a selected feature.
   - Branch/Sub-Tree: A subsection of the decision tree starts at an internal node and ends at the leaf nodes.
   - Parent Node: The node that divides into one or more child nodes.
   - Child Node: The nodes that emerge when a parent node is split.
     
![image](https://github.com/thanhite7/GR1/assets/96159427/fa70f8c7-7cfc-4cd0-bc71-11a4539acf84)


- Attribute Selection Measures:
  
   - Construction of Decision Tree: A tree can be “learned” by splitting the source set into subsets based on Attribute Selection Measures. Attribute selection measure (ASM) is a criterion used in decision tree algorithms to evaluate the usefulness of different attributes for splitting a dataset. The goal of ASM is to identify the attribute that will create the most homogeneous subsets of data after the split, thereby maximizing the information gain. This process is repeated on each derived subset in a recursive manner called recursive partitioning. The recursion is completed when the subset at a node all has the same value of the target variable, or when splitting no longer adds value to the predictions. The construction of a decision tree classifier does not require any domain knowledge or parameter setting and therefore is appropriate for exploratory knowledge discovery. Decision trees can handle high-dimensional data.

   - Entropy: is the measure of the degree of randomness or uncertainty in the dataset. In the case of classifications, It measures the randomness based on the distribution of class labels in the dataset.

The entropy for a subset of the original dataset having K number of classes for the ith node can be defined as:

![image](https://github.com/thanhite7/GR1/assets/96159427/d17afa70-7ec5-4c6a-a154-c877ac239d2a)

Where:

- S is the dataset sample.
  
- k is the particular class from K classes
  
- p(k) is the proportion of the data points that belong to class k to the total number of data points in dataset sample S. 

- Important points related to Entropy:
  
   - The entropy is 0 when the dataset is completely homogeneous, meaning that each instance belongs to the same class. It is the lowest entropy indicating no uncertainty in the dataset sample.
     
   - when the dataset is equally divided between multiple classes, the entropy is at its maximum value. Therefore, entropy is highest when the distribution of class labels is even, indicating maximum uncertainty in the dataset sample.
     
   - Entropy is used to evaluate the quality of a split. The goal of entropy is to select the attribute that minimizes the entropy of the resulting subsets, by splitting the dataset into more homogeneous subsets with respect to the class labels.
     
   - The highest information gain attribute is chosen as the splitting criterion (i.e., the reduction in entropy after splitting on that attribute), and the process is repeated recursively to build the decision tree.
     
- Gini Impurity or index:
Gini Impurity is a score that evaluates how accurate a split is among the classified groups. The Gini Impurity evaluates a score in the range between 0 and 1, where 0 is when all observations belong to one class, and 1 is a random distribution of the elements within classes. In this case, we want to have a Gini index score as low as possible. Gini Index is the evaluation metric we shall use to evaluate our Decision Tree Model.

![image](https://github.com/thanhite7/GR1/assets/96159427/03bad557-5d2b-44db-b4a6-49a19bf2bd5c)

- Information Gain:
Information gain measures the reduction in entropy or variance that results from splitting a dataset based on a specific property. It is used in decision tree algorithms to determine the usefulness of a feature by partitioning the dataset into more homogeneous subsets with respect to the class labels or target variable. The higher the information gain, the more valuable the feature is in predicting the target variable. 

The information gain of an attribute A, with respect to a dataset S, is calculated as follows:

![image](https://github.com/thanhite7/GR1/assets/96159427/45b6ea90-03d0-4e32-aac6-aa1676f2aefe)

where

- A is the specific attribute or class label
  
- |H| is the entropy of dataset sample S
  
- |HV| is the number of instances in the subset S that have the value v for attribute A

 - The algorithm repeats this action for every subsequent node by comparing its attribute values with those of the sub-nodes and continuing the process further. It repeats until it reaches the leaf node of the tree. The complete mechanism can be better explained through the algorithm given below.

Step-1: Begin the tree with the root node, says S, which contains the complete dataset.

Step-2: Find the best attribute in the dataset using Attribute Selection Measure (ASM).

Step-3: Divide the S into subsets that contains possible values for the best attributes.

Step-4: Generate the decision tree node, which contains the best attribute.

Step-5: Recursively make new decision trees using the subsets of the dataset created in step -3. Continue this process until a stage is reached where you cannot further classify the nodes and called the final node as a leaf nodeClassification and Regression Tree algorithm.

- Advantages of the Decision Tree:

    - It is simple to understand as it follows the same process which a human follow while making any decision in real-life.
    
    - It can be very useful for solving decision-related problems.
      
    - It helps to think about all the possible outcomes for a problem.
      
    - There is less requirement of data cleaning compared to other algorithms.
      
- Disadvantages of the Decision Tree:

    - The decision tree contains lots of layers, which makes it complex.
      
    - It may have an overfitting issue, which can be resolved using the Random Forest algorithm.
      
    - For more class labels, the computational complexity of the decision tree may increase.
 
IV. Basic Deep Learning algorithms

1. Artificial Neural Networks (ANN)

Artificial Neural Networks contain artificial neurons which are called units. These units are arranged in a series of layers that together constitute the whole Artificial Neural Network in a system. A layer can have only a dozen units or millions of units as this depends on how the complex neural networks will be required to learn the hidden patterns in the dataset.

Commonly, Artificial Neural Network has an input layer, an output layer as well as hidden layers. The input layer receives data from the outside world which the neural network needs to analyze or learn about. Then this data passes through one or multiple hidden layers that transform the input into data that is valuable for the output layer. Finally, the output layer provides an output in the form of a response of the Artificial Neural Networks to input data provided. 

In the majority of neural networks, units are interconnected from one layer to another. Each of these connections has weights that determine the influence of one unit on another unit. As the data transfers from one unit to another, the neural network learns more and more about the data which eventually results in an output from the output layer. 

![image](https://github.com/thanhite7/GR1/assets/96159427/3ff8e3e3-8f32-4462-93bb-5ff7241d5fd3)

The structures and operations of human neurons serve as the basis for artificial neural networks. It is also known as neural networks or neural nets. The input layer of an artificial neural network is the first layer, and it receives input from external sources and releases it to the hidden layer, which is the second layer. In the hidden layer, each neuron receives input from the previous layer neurons, computes the weighted sum, and sends it to the neurons in the next layer. These connections are weighted means effects of the inputs from the previous layer are optimized more or less by assigning different-different weights to each input and it is adjusted during the training process by optimizing these weights for improved model performance. 

- Compare Biological Neuron vs Artificial Neuron

![image](https://github.com/thanhite7/GR1/assets/96159427/c81834c4-62b4-4b3d-b211-9efa71ddcd51)

![image](https://github.com/thanhite7/GR1/assets/96159427/8c069a21-5d6d-4368-9519-a125b8db8998)

- Types of Artificial Neural Networks
  
   •	Feedforward Neural Network

   •	Convolutional Neural Network
  
   •	Modular Neural Network
  
   •	Radial basis function Neural Network
  
   •	Recurrent Neural Network:

Artificial Neural Networks (ANNs) have several advantages and disadvantages. 

- Advantages of ANN:
   •	Non-linearity: ANNs can model complex non-linear relationships between inputs and outputs, allowing them to solve problems that are not easily solvable with traditional linear models.

   •	Adaptability: ANNs can adapt their internal parameters based on the available data, allowing them to learn and improve their performance over time.
  
   •	Parallel processing: ANNs can perform computations in parallel, which enables them to process large amounts of data quickly and efficiently.
  
   •	Fault tolerance: ANNs can continue to function even if some of their components fail or are damaged. They can still provide reasonable outputs, making them robust in real-world applications.
  
   •	Feature extraction: ANNs can automatically learn and extract relevant features from raw data, eliminating the need for manual feature engineering.
  

- Disadvantages of ANN:
  
   •	Black box nature: ANNs often operate as "black boxes," meaning they provide accurate predictions but offer limited interpretability. Understanding the underlying reasoning behind their decisions can be challenging.
  
   •	Overfitting: ANNs are prone to overfitting, especially when the training data is limited or noisy. Overfitting occurs when the network becomes too specialized in the training data and performs poorly on unseen data.
  
   •	Computational complexity: Deep ANNs with many layers and neurons can be computationally expensive, requiring substantial computational resources and time for training and inference.
  
   •	Data requirements: ANNs typically require large amounts of labeled data to achieve good performance. Acquiring and labeling such datasets can be time-consuming and expensive.
  
   •	Hyperparameter tuning: ANNs have several hyperparameters that need to be carefully tuned to achieve optimal performance. Finding the right combination of hyperparameters can be a challenging and iterative process.



Applications of Artificial Neural Networks

- Social Media: Artificial Neural Networks are used heavily in Social Media. This is done by finding around 100 reference points on the person’s face and then matching them with those already available in the database using convolutional neural networks. 


- Marketing and Sales:  This uses Artificial Neural Networks to identify the customer likes, dislikes, previous shopping history, etc., and then tailor the marketing campaigns accordingly. 

- Healthcare: Artificial Neural Networks are used in Oncology to train algorithms that can identify cancerous tissue at the microscopic level at the same accuracy as trained physicians. Various rare diseases may manifest in physical characteristics and can be identified in their premature stages by using Facial Analysis on the patient photos. So the full-scale implementation of Artificial Neural Networks in the healthcare environment can only enhance the diagnostic abilities of medical experts and ultimately lead to the overall improvement in the quality of medical care all over the world. 

- Personal Assistants: These are personal assistants and an example of speech recognition that uses Natural Language Processing to interact with the users and formulate a response accordingly. Natural Language Processing uses artificial neural networks that are made to handle many tasks of these personal assistants such as managing the language syntax, semantics, correct speech, the conversation that is going on, etc.


2. Recurrent Neural Network (RNN)

Recurrent Neural Network(RNN) is a type of Neural Network where the output from the previous step is fed as input to the current step. In traditional neural networks, all the inputs and outputs are independent of each other. Still, in cases when it is required to predict the next word of a sentence, the previous words are required and hence there is a need to remember the previous words. Thus RNN came into existence, which solved this issue with the help of a Hidden Layer. The main and most important feature of RNN is its Hidden state, which remembers some information about a sequence. The state is also referred to as Memory State since it remembers the previous input to the network. It uses the same parameters for each input as it performs the same task on all the inputs or hidden layers to produce the output. This reduces the complexity of parameters, unlike other neural networks.

![image](https://github.com/thanhite7/GR1/assets/96159427/69c3c9b2-4d24-4ae1-b540-37cd6fdcb89d)

Information moves from the input layer to the output layer – if any hidden layers are present – unidirectionally in a feedforward neural network. These networks are appropriate for image classification tasks, for example, where input and output are independent. Nevertheless, their inability to retain previous inputs automatically renders them less useful for sequential data analysis.

![image](https://github.com/thanhite7/GR1/assets/96159427/3589f6ec-4e1f-4579-9a6b-6d96a153acda)

- Recurrent Neuron and RNN Unfolding
  
The fundamental processing unit in a Recurrent Neural Network (RNN) is a Recurrent Unit, which is not explicitly called a “Recurrent Neuron.” This unit has the unique ability to maintain a hidden state, allowing the network to capture sequential dependencies by remembering previous inputs while processing. Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) versions improve the RNN’s ability to handle long-term dependencies.

![image](https://github.com/thanhite7/GR1/assets/96159427/ccfde207-381a-4dfe-8223-25ef4e9be9d4)
![image](https://github.com/thanhite7/GR1/assets/96159427/f57eeea3-cb30-48d3-b8db-6a5068ad073f)

- Types Of RNN:
  
There are four types of RNNs based on the number of inputs and outputs in the network.

   - One to One

     ![image](https://github.com/thanhite7/GR1/assets/96159427/fa4fa180-54b0-4bfb-a3d3-b4eab7af03e5)

      
   - One to Many

     ![image](https://github.com/thanhite7/GR1/assets/96159427/46ec8fa7-cb71-4647-9c0b-15eaba3d99d3)

     
   - Many to One

     ![image](https://github.com/thanhite7/GR1/assets/96159427/8a7b28b7-210e-4906-adb6-132991703f62)

   - Many to Many

     ![image](https://github.com/thanhite7/GR1/assets/96159427/fda71896-95e1-4f7e-8ba1-66bb3d06d416)

- RNN architecture

![image](https://github.com/thanhite7/GR1/assets/96159427/44e66d64-688b-43b6-95a8-c0891cc6c554)

- How does RNN work?
The Recurrent Neural Network consists of multiple fixed activation function units, one for each time step. Each unit has an internal state which is called the hidden state of the unit. This hidden state signifies the past knowledge that the network currently holds at a given time step. This hidden state is updated at every time step to signify the change in the knowledge of the network about the past. The hidden state is updated using the following recurrence relation.

-The formula for calculating the current state:

![image](https://github.com/thanhite7/GR1/assets/96159427/a9705aa1-0608-4be6-80d2-fb3cb235c314)

where,

   - ht -> current state
     
   - ht-1 -> previous state
     
   - xt -> input state
     
Formula for applying Activation function(tanh)

![image](https://github.com/thanhite7/GR1/assets/96159427/795c4e34-7fe0-488e-8f91-77ddb3a05986)

where,

   - whh -> weight at recurrent neuron

   - wxh -> weight at input neuron

The formula for calculating output:

![image](https://github.com/thanhite7/GR1/assets/96159427/e61767a8-5c89-417a-a412-a6427f92ad4c)


   - Yt -> output
   - Why -> weight at output layer



Advantages and Disadvantages of Recurrent Neural Network

- Advantages:
  
   - An RNN remembers each and every piece of information through time. It is useful in time series prediction only because of the feature to remember previous inputs as well. This is called Long Short Term Memory.
     
   - Recurrent neural networks are even used with convolutional layers to extend the effective pixel neighborhood.

- Disadvantages:

   - Gradient vanishing and exploding problems.
     
   - Training an RNN is a very difficult task.
     
   - It cannot process very long sequences if using tanh or relu as an activation function.

- Applications of Recurrent Neural Network
  
   - Language Modelling and Generating Text
     
   - Speech Recognition
     
   - Machine Translation
     
   - Image Recognition, Face detection
     
   - Time series Forecasting



