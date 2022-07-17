# Parkinsons-prj
Predicts early detection of Parkinson's disease using Xg-boost and other machine learning algorithms.

## ABSTRACT
Parkinson’s disease (PD) can be detected at an early stage using voice signals. This project gives insights into different voice features that help for early PD detection. The pre-monitoring stage of Parkinson’s disease is very important and it can be detected and its severity can be known using XG Boost algorithm with the help of the speech features dataset. The accuracy and performance of KNN, Decision tree, Logistic regression are compared with the XGBoost algorithm based on the data of 48 healthy persons and 147 Parkinson’s disease patients.

Parkinson’s disease is a central nervous system disorder. It results in Nerve cell damage in the brain. The symptoms can differ from one person to another person. There are two types of symptoms that are observed. One is related to the movement(motor) and the other related to non-movement(non-motor). The movement symptoms include tremor in one hand, slow movement, loss of balance in the body, and no proper stiffness (improper balance). The non-movement symptoms include depression, loss of smell etc.
The patients affected with this disease will have speech disorders and are prevailed in them. Speech analysis and its distinguishing are necessary to recognize the early symptoms of Parkinson’s disease and can be treated at the early stage and can stop their symptoms from occurring. Thus, the patient’s life can be saved. This prediction of the disease would be very much useful in the medical field. Voice measurements like Dysarthria, Hypophonia, Tachyphemia can be detected at the early stage of the disease. This problem can be solved using Machine Learning algorithms like Logistic Regression, Decision tree, Random Forest, XG Boost and the comparison of accuracy and performance can be known.

## XG BOOST ALGORITHM:

1. XG Boost is a decision making tree-based algorithm designed using Machine learning that utilizes a gradient boosting framework. It is used in predicting problems involving unstructured information which includes text, artificial neural networks that end to surpass all other algorithms or frameworks. Although, when it comes to small-to-medium structured or tabulated data, decision-tree-based algorithms are considered to be best-in-class right now.
2. XGBoost is a well-prepared method. Sometimes, it can now no longer be enough to rely on the effects of the results which we got in machine learning modeling. Ensemble mastering gives scientific technique to unite the predictive energy of more than one learner. The resultant received is a single model which gives us mixed outputs from numerous models.
3. The models that shape the ensemble also referred to as base learners, might be both from the same learning algorithms or distinct learning algorithms. Bagging in addition to boosting are the broadly speaking used ensemble mastering methods. Though these strategies may be used with numerous statistical fashions, the maximum foremost utilization includes selection trees.

## XGBOOST Model Features:
The two important features which it focusses on is: 1) Computational Speed 2) Model Performance
The above two features are advanced in XGBOOST algorithm and thus making it unique when compared to other algorithms. The other advantage is that it handles a large amount of data more than the RAM capacity.
## XGBOOST PERFORMANCE:

1) Regularization Process:
This is viewed as one of the predominant factor of the calculation. Regularization is a procedure which can be utilized to solve the problem of overfitting of the XGBoost model.
2) Cross-Validating the Data:
Cross-validation of the data is the process that is used for getting in the capacity from sklearn. But the present algorithm that is XGBoost algorithm is already imbibed with an inbuilt function called CV which is the symbol for Cross Validation.
3) Absent Values:
It is as Absent values also called as missing values so that it can handle all the respective unknow or missing quantities. Some kind of patterns are detected in the missing quantities and then getting necessary inputs from them and realize them.
4) Flexibility:
Flexibility gives the help for all the important or main functions. They are utilized for assessing the model presentation and even further it can also deal with the approval measurements that are characterized by the client.
5) Saving & loading:
They are responsible for enabling to save the details of the network and further reload thereafter that saves the data and time.

## XGBOOST ALGORITHM WORKING PROCESS:
XGBoost is responsible for the gradient boosting and also for the tree based calculation process. It has a some distinct names like gradient boosting algorithm, gradient boosting ML algorithm, and it is so forth.

Boosting is the one and only troupe method where the previous model mistakes are corrected in the new models created. These new models continue the process until there is no other error is visible. Perhaps the best representaion of such a calculation is called AdaBoost Algorithm which is also called Adaptive Boosting Algorithm.
Gradient boosting is a model where the new models are being made that finds out the mistakes in the previous model and then later on a correction is added in order to make the last expectation.
The reason for it to be called Gradient Boosting Algorithm is that because of the usage of algorithm called Gradient. Arrangement of the climate strategies are upheld to the two sorts of issues that are displayed which are prescient.

## GRADIENT BOOSTING:
● It is an approach with the help of which new models can be created and the errors which are being made in the previous models like Logistic Regression, Decision tree and KNN. This is called Gradient Boosting.
● Boosting and Gradient are the two sub-terms that are present in the Gradient Boosting Algorithm. We realize that it is a boosting technique. Let us perceive how the term 'gradient' is connected in this aspect.
● Gradient boosting re-characterizes boosting as a mathematical improvement issue where the goal is to limit the loss function capacity of the model by adding powerless or weak learners utilizing gradient descent.
● Gradient Descent is a first-request iterative improvement calculation for tracking down a nearby least of a differentiable capacity term. As angle boosting depends on the limiting of the loss function, various sorts of loss functions can be utilized bringing about an adaptable method that can be applied to the multi-class arrangement, Regression, and so on. 3.5.5 Differences Between Single Process, Bagging, and Boosting: 1) Single Process Execution:

## Single Process Execution
● This process consists of a single tree and a single decision is to be taken. Only one iteration takes place in this process. ● Generally, Decision Trees follow this type of execution process and single response that is the output is given which is the result. ● It does not have multiple trees in its process of execution. It has only a single tree with only one iteration that is whether it belongs to true condition or false condition.
2) Bagging Process

## Bagging Process

Stowing, a Parallel outfit strategy (represents Bootstrap Aggregating), is a method to that decrease the distinction of the expectancy version through growing great rerecords withinside the training stage. This is introduced through arbitrary testing out with substitution from the primary set.
By testing out with substitution, some perceptions is probably rehashed in every new getting ready informational collection. On account of Bagging, every aspect has a comparable chance to reveal up in every other dataset. By increasing the scale of the training set, the version's prescient strength cannot be improved. It diminishes the extrude and slightly tunes the forecast to a regular result.
These multisets of information are utilized to prepare different models. Accordingly, we end up with a troupe of various models. The normal of the multitude of expectations from various models is utilized. This is more powerful than a model. The forecast can be the normal of the multitude of expectations given by the various models if there should be an occurrence of relapse. On account of characterization, the greater part vote is mulled over.
3) Boosting:  It is a process that occurs in Gradient Boosting Algorithm. The sequential process that takes place in this algorithm that is the feedback of the first tree will be the input to the second tree. Thus, the subsequent tree reduces the error when compared to the first tree. This is the reason the number of errors is reduced in this Gradient Boosting Algorithm when compared to other machine learning algorithms.

● The boosting outfit strategy for AI steadily adds feeble students prepared on weighted variants of the preparation dataset.
● The fundamental thought that underlies all boosting calculations and the key methodology utilized inside each boosting calculation.
● How the fundamental thoughts that underlie boosting could be investigated on new predictive type of demonstrating projects. 

Comparision of Bagging, Boosting in Algorithms XGBoost is a selection-tree that is primarily based on Machine Learning algorithm where it has set of instructions which makes usage of the framework which is called gradient boosting . In predicting the troublesrelated to unstructured data (images, text, etc.). Neural networks are responsible for performing all different frameworks. However, on the subject of small-to-medium structured/tabular data, selection tree primarily based totally algorithms are taken into consideration best-in-magnificenceproper now. Please see the chart beneath for the evolution of tree-primarily based totally algorithms over the years.
##  XGBoost Algorithm Development from Decision Trees

The set of rules that find out the difference itself withinside the respective ways: 1. Applications of various types: It is used for resolving problems related to regression related, classification related and user-described problems which can be predicted. 2. Transportability: It runs easily on any platform like OS X, Linux etc. 3. Languages: Supports all foremost programming languages which include C++, Python, Scala, Julia, R, and Java. 4. Cloud Integration: Supports platforms like AWS, Azure, and Yarn clusters and it also supports its functionality properly with Spark, and different ecosystems.

Build an intuition for XGBoost The simplest form of Decision Trees which can be easily understandable and also can be easily interpreted algorithms however constructing instinct for the next technology of tree-based algorithms may be some tricky. See beneath for a easy analogy to have a better understanding of the eemerging of tree-primarily based totally algorithms. For example: Let us imagine that you are recruiting a person for your job roleand interviewing numerous applicants with first-rate qualifications. Each and every step in the evolution process of tree-primarily based totally algorithms may be regarded as a model of the interview process. 
