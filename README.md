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
     
   - Document Classification: Text documents can be clustered based on their content, aiding in document classification or organizing large document collections.
     
   - Genomic Data Analysis: K-means is employed in bioinformatics to analyze gene expression data, identifying patterns and grouping genes with similar expression profiles.
     
   - Spatial Data Analysis: Geographic data, such as the clustering of geographical regions based on certain features, can be performed using K-means.






