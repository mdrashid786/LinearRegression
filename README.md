# LinearRegression
import pandas as pd
import numpy as np
from sklearn import linear_model

r=pd.read_csv("C:\\Users\\ali\\OneDrive\\Desktop\\aniya.csv")
r
 experince	test_score	interview_score	salary
0	      0        	8.0	               9	50000
1      	0        	8.0              	 6	45000
2     	5        	6.0	               7	60000
3     	2         10.0	            10	65000
4	      7	        9.0              	6 	70000
5      	3	        7.0	              10	62000
6     	10      	NaN	              7	  72000
7      	11	      7.0	              8	  80000

import math
r.test_score=r.test_score.fillna(r.test_score.median())
r

 experince	test_score	interview_score	salary
0	      0        	8.0	               9	50000
1      	0        	8.0              	 6	45000
2     	5        	6.0	               7	60000
3     	2         10.0	            10	65000
4	      7	        9.0              	6 	70000
5      	3	        7.0	              10	62000
6     	10      	8.0	              7	  72000
7      	11	      7.0	              8	  80000

model=linear_model.LinearRegression()
model.fit(r.drop('salary',axis='columns'),r.salary)

LinearRegression()

model.predict([[2,9,6]])
array([53205.96797671])
model.coef_
array([2812.95487627, 1845.70596798, 2205.24017467])
model.intercept_
17737.26346433771
2812.95487627*2+1845.70596798*9+ 2205.24017467*6+17737.26346433771
2813.53205.967976717715
2814.model.predict([[12,10,10]])
array([92002.18340611])
final answer

