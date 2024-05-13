# ScDR Code


High level description: This project aims at developing a Physics-enabled Dimensionality Reduction framework that leverages the properties of a physical system to perform in-substrate machine learning. The conventional machine learning methods employed for pattern recognition in physical structures typically rely on high-dimensional data obtained from a multitude of sensors. In contrast, the framework developed in this paper endeavors to achieve these learning objectives using only a sparse array of sensors, while imposing minimal computational demands on the processor. This is accomplished by adjusting the tunable components of the structure itself such that maximum information can be preserved at the limited sensing points. Here, the above concept is implemented to classify various sitting postures based on the force patterns exerted on the sitting surface of a chair. The sitting surface of the chair is modeled as a network of interconnected springs. Each spring represents a structural component of the chair, and collectively they simulate the behavior of the chair’s surface for different sitting postures. The adjacent springs in the network are connected at a point which is referred to as a sensing point. These are the points of application of forces on the chair. Based on the force pattern distribution due to sitting, and the values of spring constants, specific displacements can be obtained at those sensing points. A subset of these sensing points is selected for actual measurement of displacements. These sensing points are strategically chosen based on their significance in capturing relevant information about sitting postures. For different sitting postures, the measured displacement values at these selected sensing points are then combined with corresponding class labels and fed into a lightweight classifier. Based on the classification performance, the spring network is allowed to change the spring stiffness in such a way that maximum information can be preserved at the displacement values at the selected sensing points. During the training process, the spring stiffness values are allowed to change within certain bounds. These bounds are guided by the spring material properties and user’s sitting comfort. Once the spring network is trained for the given classes of postures, the sitting surface of the chair is designed with those trained stiffness values. The chair designed with the trained spring stiffness allows the real-time classification of the force patterns representing different sitting postures. Since, in this setup, the computation fabric required for successful execution of machine learning is the substrate itself, the entire paradigm is called Substrate-computed Dimensionality Reduction (ScDR).


Code details:

Main code: ScDR_training_algo.m

Algorithmic Flow:

1. The mean feature value for each of the classes in the dataset is computed. 
2. We compute the mean displacement for each class by inputting the mean feature values and observing the displacement at the selected output sensing points.
3. Then for all the training samples defined by the force values, the corresponding displacement values are observed.
4. We compute the learning loss using a Minimum Distance Classifier for all the training samples, based on the output displacement values. In other words, the displacement values of the training samples are compared with the mean displacement values of each class, using Euclidean distance, and loss is defined as the average number of samples from a class C that is close to mean displacement of any other class.
5. Next, for each spring in the network, we increase and decrease the stiffness values by a small amount \Delta and observe the corresponding loss values. This is done to guide the algorithm to obtain the direction of minimizing the loss.
6. Once the direction of minimum loss is obtained, the spring stiffness values are adjusted in that direction guided by an empirically chosen learning rate. While adjusting the spring stiffness, the bounds on the minimum and maximum possible values of spring stiffness are taken into account and the stiffness values are not allowed to exceed these bounds.
7. The above procedure is repeated for "epochs" number of steps that would allow the algorithm to find a set of spring stiffness values for which the learning loss is minimized.
8. Once the spring network is trained using the procedure explained above, for any unseen force patterns, the displacement values are obtained at the selected output points X and classified using Minimum Distance Classifier (MDC). This is accomplished by comparing the Euclidean distance between the displacement of the test sample and the mean displacement of each class.


More details on code functions, variables and structure:

1. Define and initialize variables: num_runs (number of runs of the algo), epochs (Number of epochs for loss minimzation), del_w (small nudge to find loss minimization direction), num_feat (number of features in the dataset), num_spr (number of springs in the system), n_fold (number of folds for K-fold validation)
2. Load the dataset in varaiables class_1_data and class_2_data for each class respectively
3. Shuffle data and split into train and test
4. Define learning hyperparameters: learning rate (lr_max, lr_dec), batch size
5. Define bounds of spring constants: k_min, k_max
6. Initialize weight (spring constant) matrix: w
7. Check weight matrix update to find direction of loss minimization
8. Change weights along the direction of loss minimization
9. Find the test accuracy for each validation fold


Function disp_vec:

Purpose: STATIC ANALYSIS OF A N X N GRID OF SPRINGS USING DIRECT STIFFNESS OPERATIONS

Code Flow:
1. Define the connectivity between local and global dof's in variable id
2. Form element stiffness matrices using function truss2d.m
3. Assemble global stiffness matrix in matrix kg
4. Form the global load vector in fg
5. Solve for nodal displacements vg = kg\fg
6. Find resultant displacement and store it in variable displacement
7. Find the dislacements at desired sensing points using variable disp_vec


Script: ScDR_data_generation:

Purpose: Generate dataset with different classes of sitting postures with force distributions as features

This code shows the data generation for three classes but can be extended to any class

1. Initialize the force distribution matrix Force_class_1, Force_class_2 and Force_class_3
2. Mean force values for high, midium and low force distributions: f_high, f_mid, f_low
3. Standard deviation of forces high, midium, low and null force distributions: sd_high, sd_mid, sd_low, sd_zero


Note: Store all the scripts for functions in the same directory as the main function

