Caitlin Conn ENPM661 Project 2 Readme.txt
Directory ID: 114048295

This text provides the necessary instructions to run the proj1_caitlin_conn_sourcecode.py file to output the results for a Dijkstra Approach to a point robot search problem.
################################################################################################

Project # 2 Important Notes:
* The node indices start from 1 (start node), not 0. The first node (start node) is assumed to have no parent, so its parent node index is 0.
* Imported required libraries include the following and are listed in the beginning of the python script:
import matplotlib.pyplot as plt 
import numpy as np 
import time
import math
import cv2 
import imutils

from datetime import datetime

Packages Installed:
•	pip3 install opencv-python
•	pip install images2gif
•	pip install pillow
•	pip install vidmaker
•	pip install pygame
•	pip install imutils



* Please see the submitted .txt and .odt files under the 'Test Case Results' folder in the student's .zip folder to view the outputed results (of 3 test cases) written to the nodePath.txt, Nodes.txt, and NodesInfo.txt files. Each test case has their own folder for organization purposes.
	
################################################################################################

Project # 2 Run Instructions: 
To run the code to solve the search problem using Dijkstra's algorithm using Python, perform the following: 

If using a command line terminal, put the proj1_caitlin_conn_sourcecode.py file in the current working directory and enter the following command in the terminal to run the python program:
 
	python3 ./proj1_caitlin_conn_sourcecode
	
Otherwise, use an IDE to open and run the proj1_caitlin_conn_sourcecode.py file, by hitting the 'run' button.

------------------------------------------------------------------------------------------------

The user should modify the following input parameters (initial node and goal node) by editing the following parameters at the bottom of the proj1_caitlin_conn_sourcecode.py file using an IDE or text editor as shown below:

The below lines are listed in the python script and can be edited to test the results. The user can set the variables initial_node and goal_node to one of the 3 test cases as shown below, or uncomment variables initial_node_custom_user_case and goal_node_custom_user_case to enter their own random custom test case.

################################################################################################

# *** Define/change input parameters that can be modified by the user to run different test cases on the code ***
    
# Function outputs to textfile the order of states from the initial to the goal node in column-wise order, such that:
# For example, for state 1 4 7 2 5 8 3 6 0, the eight puzzle state  is:
#1 2 3
#4 5 6
#7 8 0
    
# Case 1 provided 
initial_node_case1 = [1, 2, 4, 6, 0, 3, 7, 5, 8]
goal_node_case1 = [1, 2, 3, 4, 5, 0, 7, 8, 6]

# Case 2 provided
initial_node_case2 = [4, 2, 3, 7, 1, 6, 8, 5, 0]
goal_node_case2 = [1, 2, 3, 4, 5, 6, 7, 8, 0]

# Case 3 referenced in lecture slides
initial_node_case3 = [2, 1, 7, 8, 6, 0, 3, 4, 5]
goal_node_case3 = [1, 8, 7, 2, 0, 6, 3, 4, 5]

# USER CAN UNCOMMENT THE BELOW TWO LINES TO ENTER CUSTOM TEST CASES 
# IF DONE SO, ASSIGN BELOW initial_node = initial_node_custom_user_case
# IF DONE SO, ASSIGN BELOW goal_node = goal_node_custom_user_case

# initial_node_custom_user_case = [...] 
# goal_node_custom_user_case = [...]

# MODIFY THE BELOW TWO LINES TO CHANGE INPUT PARAMETERS TO CODE
initial_node = initial_node_case3
goal_node = goal_node_case3

main(initial_node, goal_node)

# END OF SOURCE CODE FILE

################################################################################################

