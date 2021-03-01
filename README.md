#  K Nearest Neighbor Algorithm - KNN
	K represents the number of the nearest neighbors
        that we used to classify new data points.
	
	Choosing the right value of K is called parameter tuning and 
    it’s necessary for better results.
        - K = sqrt (total number of data points).
        - Odd value of K is always selected to avoid confusion between 2 
        classes.
		
	We usually use Euclidean distance to calculate the nearest
    neighbor. If we have two points (x1, y1) and (x2, y2). 
    The formula for Euclidean distance (d) will be: 
        distance = sqrt((x1 - x2)² + (y1 - y2)²)
		
	There are various methods for calculating the distance
    between the points, of which the most commonly known
    methods are –
        1. Euclidean,
        2. Manhattan (for continuous) and
        3. Hamming distance (for categorical).
	
## Somthing to know:
    1. It is simple to implement and mostly used for classification.
    2. It is easy to interpret.
    3. Large datasets requires a lot of memory and gets slow
            to predict because of distance calculations.
    4. Accuracy can be broken down if there are many predictors
    5. It doesn't generate insights

Read about [ k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
