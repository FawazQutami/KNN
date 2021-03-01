#  K Nearest Neighbor Algorithm - KNN
	K represents the number of the nearest neighbors
        that we used to classify new data points.
	
## KNN Algorithm:
		1. Load the data
		2. Initialize K - number of neighbors: Choosing the right value of K is called parameter tuning and 
			it’s necessary for better results:
				- K = sqrt (total number of data points).
				- Odd value of K is always selected to avoid confusion between 2 classes.		
		3. For each value in the x_test: 
			1. Calculate the distance between x_test value and all x_train values.
				We usually use Euclidean distance to calculate the nearest neighbor. 
				If we have two points (x1, y1) and (x2, y2), the formula for Euclidean distance will be: 
						distance = sqrt((x1 - x2)² + (y1 - y2)²)	
			2. Sort the claculated distances and indices from smallest to largest
			3. Return K indexs of the smallest distances
			4. VOTING for the outcome:
				1. Find the associated labels of the returned K indexs
				2. Get the most common class - the final outcome
		
## Distance Calculation Methods:
	There are various methods for calculating the distance
    between the points, of which the most commonly known
    methods are –
        1. Euclidean,
        2. Manhattan (for continuous) and
        3. Hamming distance (for categorical).
	
## Somthing to know about KNN:
		1. It is simple to implement and mostly used for classification.
		2. It is easy to interpret.
		3. Large datasets requires a lot of memory and gets slow
				to predict because of distance calculations.
		4. Accuracy can be broken down if there are many predictors
		5. It doesn't generate insights

Read more about [ k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
