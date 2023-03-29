# -*- coding: utf-8 -*-

"""
Created on Tue Mar 21 16:05:08 2023

@author: caitlin.p.conn

"""
# Required imported libraries
import matplotlib.animation as animation
import matplotlib.pyplot as plt 
import numpy as np 
import cProfile
import heapq as hq
import time
import math
import cv2 

from datetime import datetime
from PIL import Image

# https://stackoverflow.com/questions/69671976/python-function-to-find-a-point-of-an-equilateral-triangle
def find_equal_triangle_coordinate(pt1, pt2):
                    
    pt3_x = (pt1[0]+pt2[0]+np.sqrt(3)*(pt1[1]-pt2[1]))/2 
    pt3_y = (pt1[1]+pt2[1]+np.sqrt(3)*(pt1[0]-pt2[0]))/2  
    unknown_pt = np.array([pt3_x, pt3_y])
    
    return unknown_pt

# Resource https://www.geeksforgeeks.org/program-find-line-passing-2-points/
def compute_line_abc(pt_a, pt_b):

    a_val = pt_b[1] - pt_a[1]
    b_val = pt_a[0] - pt_b[0]
    c_val = (a_val*(pt_a[0])) + (b_val*(pt_a[1]))

    return a_val, b_val, c_val

def create_map_grid():
    
    # Define map grid side
    map_height = 250
    map_width = 600
    map_grid = np.ones((map_height,map_width,3), dtype = np.uint8)
    # map_grid = np.ones((map_height,map_width,3))
    # map_grid = np.zeros((map_height,map_width,3), dtype = np.uint8)
    
    # Define obstacle and wall color
    obstacle_color = (255,0,0) #RED
    # Define obstacle clearance color
    clearance_color = (102, 0, 0) #DARK RED #(0,255,0) #GREEN
            
    triangle_factor = 1000
    c = 5 #5 pixels
    
    #############################################################
    
    # Display hexagon

    hexagon_x_center = 300 #100+50+150
    hexagon_y_center = 125
    hex_edge_length = 75

    # Resource: quora.com/How-do-you-find-the-distance-between-the-edges-and-the-center-of-a-regular-hexagon-if-you-know-the-length-of-its-sides
    hex_dist_center_to_edge = hex_edge_length * math.sqrt(3)/2

    # Hexagon Vertex 1 - Top
    v1_x = int(100 + 50 + 150)
    v1_y = int(125 + hex_dist_center_to_edge)

    vertex1 = [hexagon_x_center, hexagon_y_center]
    vertex2 = [v1_x,v1_y]
    result = find_equal_triangle_coordinate(vertex1, vertex2)

    # Hexagon Center Coordinate
    map_grid = cv2.circle(map_grid, (hexagon_x_center,hexagon_y_center), radius=5, color=(255,0,0), thickness=-1)

    # Hexagon Vertex 2
    v2_x = 100 + 50 + 150 + hex_dist_center_to_edge
    v2_y = int(result[1])

    # Hexagon Vertex 6
    v6_x = v1_x - hex_dist_center_to_edge
    v6_y = int(result[1])

    # Hexagon Vertex 3
    v3_x = int(v2_x)
    v3_y = int(result[1]) - hex_edge_length

    # Hexagon Vertex 4
    v4_x = int(v1_x)
    v4_y = int(125 - hex_dist_center_to_edge) 

    # Hexagon Vertex 5
    v5_x = int(v6_x)
    v5_y = int(result[1])-hex_edge_length

    pt1 = [v1_x,v1_y+c]
    pt2 = [v2_x+c,v2_y+c]
    pt3 = [v3_x+c,v3_y-c]
    pt4 = [v4_x,v4_y-c]
    pt5 = [v5_x-c,v5_y-c]
    pt6 = [v6_x-c,v6_y+c]

    l1a, l1b, l1c = compute_line_abc(pt1, pt2)
    l2a, l2b, l2c = compute_line_abc(pt2, pt3)
    l3a, l3b, l3c = compute_line_abc(pt3, pt4)
    l4a, l4b, l4c = compute_line_abc(pt4, pt5)
    l5a, l5b, l5c = compute_line_abc(pt5, pt6)
    l6a, l6b, l6c = compute_line_abc(pt6, pt1)
    
    pt1_i = [v1_x,v1_y]
    pt2_i = [v2_x,v2_y]
    pt3_i = [v3_x,v3_y]
    pt4_i = [v4_x,v4_y]
    pt5_i = [v5_x,v5_y]
    pt6_i = [v6_x,v6_y]

    l1a_i, l1b_i, l1c_i = compute_line_abc(pt1_i, pt2_i)
    l2a_i, l2b_i, l2c_i = compute_line_abc(pt2_i, pt3_i)
    l3a_i, l3b_i, l3c_i = compute_line_abc(pt3_i, pt4_i)
    l4a_i, l4b_i, l4c_i = compute_line_abc(pt4_i, pt5_i)
    l5a_i, l5b_i, l5c_i = compute_line_abc(pt5_i, pt6_i)
    l6a_i, l6b_i, l6c_i = compute_line_abc(pt6_i, pt1_i)
    
    #############################################################
    
    tri_low_pt = [460,25]
    tri_up_pt = [460,225]
    tri_right_pt = [510,125]

    t1a, t1b, t1c = compute_line_abc(tri_low_pt, tri_up_pt)
    t2a, t2b, t2c = compute_line_abc(tri_low_pt, tri_right_pt)
    t3a, t3b, t3c = compute_line_abc(tri_up_pt, tri_right_pt)
    
    tri_low_pt = [460,25]
    tri_up_pt = [460,225]
    tri_right_pt = [510,125]

    t1a, t1b, t1c = compute_line_abc(tri_low_pt, tri_up_pt)
    t2a, t2b, t2c = compute_line_abc(tri_low_pt, tri_right_pt)
    t3a, t3b, t3c = compute_line_abc(tri_up_pt, tri_right_pt)
    
    #############################################################
    
    # Set wall color 
    #cv2.rectangle(map_grid, (1,250), (600,1), obstacle_color, 5)
    for y in range(map_height):
        for x in range(map_width):
            
            # Plot walls
            # map_height = 250
            # map_width = 600
            
            # Plot horizontal walls clearance
            if (x >= 0 and x < map_width and y >= 5 and y<10) or (x >= 0 and x < map_width and y >= 240  and y < 245):
                map_grid[y,x] = clearance_color
            
            # Plot horizontal walls
            if (x >= 0 and x < map_width and y >= 0 and y<5) or (x >= 0 and x < map_width and y >= 245  and y<map_height):
                map_grid[y,x] = obstacle_color
             
            # Plot vertical walls clearance
            if (x >= 5 and x < 10 and y >= 0 and y<map_height) or (x >= 590 and x < 595 and y >= 0 and y<map_height):
                map_grid[y,x] = clearance_color
            
            # Plot vertical walls 
            if (x >= 0 and x < 5 and y >= 0 and y<map_height) or (x >= 595 and x < map_width and y >= 0 and y<map_height):
                map_grid[y,x] = obstacle_color                

            # Display rectangles 
            # Plot lower rectange obstacle space
            if x >= 100-c and x < 150+c and y >= 0-c and y <= 100+c:
                map_grid[y,x] = clearance_color
            # Plot lower rectange clearance
            if x >= 100 and x <= 150 and y >= 0 and y <= 100:
                map_grid[y,x] = obstacle_color

            # Plot upper rectange clearance
            if x >= 100-c and x <= 150+c and y >= 150-c and y <= 250+c:
                map_grid[y,x] = clearance_color
            # Plot upper rectange obstacle space
            if x >= 100 and x <= 150 and y >= 150 and y <= 250:
                map_grid[y,x] = obstacle_color
                

            # Display hexagon
            if ( ((l1b*y)+(l1a*x)-l1c) >= 0  and ((l2b*y)+(l2a*x)-l2c) >= 0) and ((l3b*y)+(l3a*x)-l3c) >= 0 and ((l4b*y)+(l4a*x)-l4c) >= 0 and ((l5b*y)+(l5a*x)-l5c) >= 0 and ((l6b*y)+(l6a*x)-l6c) >= 0:
                map_grid[y,x] = clearance_color

            if ( ((l1b_i*y)+(l1a_i*x)-l1c_i) >= 0  and ((l2b_i*y)+(l2a_i*x)-l2c_i) >= 0) and ((l3b_i*y)+(l3a_i*x)-l3c_i) >= 0 and ((l4b_i*y)+(l4a_i*x)-l4c_i) >= 0 and ((l5b_i*y)+(l5a_i*x)-l5c_i) >= 0 and ((l6b_i*y)+(l6a_i*x)-l6c_i) >= 0:
                map_grid[y,x] = obstacle_color

            # Display triangle 
            if ( ((t1b*y)+(t1a*x)-(t1c-triangle_factor)) >= 0  and ((t2b*y)+(t2a*x)-(t2c+triangle_factor)) <= 0 and ((t3b*y)+(t3a*x)-(t3c-triangle_factor)) >= 0):
                map_grid[y,x] = clearance_color

            if ( ((t1b*y)+(t1a*x)-t1c) >= 0  and ((t2b*y)+(t2a*x)-t2c) <= 0 and ((t3b*y)+(t3a*x)-t3c) >= 0):
                map_grid[y,x] = obstacle_color
    
    return map_grid, map_height, map_width 

###############################################################################

def check_node_in_obstacle_space(child_node_x, child_node_y, obstacle_matrix):
    
    return obstacle_matrix[child_node_y][child_node_x] == -1

###############################################################################
# Function also determines the validity of the swap/action
def generate_child_node(obstacle_matrix, c2c_matrix, parent_node, action, map_grid, map_height, map_width):
    
    valid_move = False # boolean truth value of valid swap
    parent_cost_to_come = parent_node[0]
    parent_node_x = parent_node[3][0]
    parent_node_y = parent_node[3][1]
    child_node_x = 0
    child_node_y = 0
        
    # check for valid moves
    is_node_obstacle = False
    
    if action == 1: #left (-1,0)
        cost_to_move = 1
        if parent_node_x != 0: 
            child_node_x = parent_node_x - 1
            child_node_y = parent_node_y            

    elif action == 2: #up (0,1)
        cost_to_move = 1
        if parent_node_y != map_height - 1: 
            child_node_x = parent_node_x 
            child_node_y = parent_node_y + 1

    elif action == 3: # right (1,0)
        cost_to_move = 1
        if parent_node_x != map_width - 1:
            child_node_x = parent_node_x + 1
            child_node_y = parent_node_y

    elif action == 4: # down (0,-1)
        cost_to_move = 1
        if parent_node_y != 0: 
            child_node_x = parent_node_x 
            child_node_y = parent_node_y - 1

    elif action == 5: # right & up (1,1)
        cost_to_move = 1.4
        if parent_node_x != map_width - 1 and parent_node_y != map_height - 1: 
            child_node_x = parent_node_x + 1
            child_node_y = parent_node_y + 1

    elif action == 6: # left & up (-1,1)
        cost_to_move = 1.4
        if parent_node_x != 0 and parent_node_y != map_height - 1: 
            child_node_x = parent_node_x - 1
            child_node_y = parent_node_y + 1
            
    elif action == 7: # right & down (1,-1)
        cost_to_move = 1.4
        if parent_node_x != map_width - 1 and parent_node_y != 0: 
            child_node_x = parent_node_x + 1
            child_node_y = parent_node_y - 1

    elif action == 8: # left & down (-1,-1)
        cost_to_move = 1.4
        if parent_node_x != 0 and parent_node_y != 0: 
            child_node_x = parent_node_x - 1
            child_node_y = parent_node_y - 1

    is_node_obstacle = check_node_in_obstacle_space(child_node_x, child_node_y, obstacle_matrix)

    valid_move = not is_node_obstacle

    cost_to_move = round(cost_to_move+parent_cost_to_come,1)

    # returned node is the resulting child node of the requested action
    return cost_to_move, valid_move, child_node_x, child_node_y, is_node_obstacle

            
    backtrack_path_list = []
    path_list = []
        
    first_parent_coord = visited_queue[len(visited_queue)-1][2]
    curr_elem_x = visited_queue[len(visited_queue)-1][3][0]
    curr_elem_y = visited_queue[len(visited_queue)-1][3][1]
    
    path_list.append(visited_queue[len(visited_queue)-1])
    parent_coord = first_parent_coord
    
    while(not((curr_elem_x == initial_node_coord[0]) and (curr_elem_y == initial_node_coord[1]))):
        for visited_elem in visited_queue:
            curr_elem_x = visited_elem[3][0]
            curr_elem_y = visited_elem[3][1]
            if visited_elem[1] == parent_coord:
                parent_coord = visited_elem[2]
                path_list.append(visited_elem)
                break
    
    # for elem in visited_queue:
    #     print("Visited Queue Current Element: ", elem)
    # print()

    # for p in path_list:
    #     print("Path List Current Element: ", p)
    # print()
    
    backtrack_path_list = np.flipud(path_list)

    return backtrack_path_list

#################################################################################################

def initialize_c2c_matrix(map_grid, map_height, map_width):  
    
    # Create boolean arrays to represent the various regions of the map
    free_space = np.mean(map_grid, axis=-1) == 1
    obstacle_space = np.logical_not(free_space)
    
    # Create c2c_matrix using the boolean arrays
    c2c_matrix = np.zeros((map_height, map_width))
    c2c_matrix[free_space] = np.inf
    c2c_matrix[obstacle_space] = round(-1,1)

    # Set the starting point to 0
    c2c_matrix[0, 0] = round(0,1)

    return c2c_matrix  

#################################################################################################

def display_map_grid_plot(map_grid, x, y, point_thickness, goal_found, goal_x, goal_y, curr_x, curr_y):
    found_goal_bool = (curr_x == goal_x and curr_y == goal_y)
    if found_goal_bool == True:
        map_grid = cv2.circle(map_grid, (x, y), radius=0, color=(0, 0, 255), thickness=point_thickness)

    plt.figure()
    plt.title('Node Explored')
    plt.imshow(map_grid.astype(np.uint8), origin="lower")
    plt.show()
    
    return

#################################################################################################

def print_function(i, valid_move, is_node_obstacle, plot_fig, map_grid):
    if i == 1:
        print("Action Left (-1,0)")
    elif i == 2:
        print("Action Up (0,1)")
    elif i == 3:
        print("Action Right (1,0)")
    elif i == 4:
        print("Action Down (0,-1)")
    elif i == 5:
        print("Action Right & Up (1,1)")
    elif i == 6:
        print("Action Left & Up (-1,1)")
    elif i == 7:
        print("Action Right & Down (1,-1)")
    elif i == 8:
        print("Action Left & Down (-1,-1)")
        
    print("Is Valid Move Boolean -> ? : ", valid_move)
    
    print("is_node_obstacle: ", is_node_obstacle)
    
    print()
        
    if plot_fig == True:
        plt.figure()
        plt.title('Node Explored')
        plt.imshow(map_grid.astype(np.uint8), origin="lower")
        plt.show()
        
    return
       
#################################################################################################

# Function uses backtracking to find the node pathway from the initial node to goal node
# Source: https://numpy.org/doc/stable/reference/generated/numpy.fliplr.html
# Must use flipud function to ensure using a forward search strategy!!
# def generate_path(visited_queue, initial_node_coord):                
# # Main function that calls subfunctions that perform search operations

def dijkstra_approach_alg(obstacle_matrix, c2c_matrix, initial_node_coord, goal_node_coord, map_grid, map_height, map_width):
    
    fig, ax = plt.subplots()
        
    curr_parent_idx = 0 # Parent index
    node_idx = 1 # WAS 1 # Node index 
    
    debug_counter = 0

    # Create empty queues
    visited_queue = [] # explored, valid nodes
    open_queue = [] # keeps track of node queue to be processed
    
    show_grid = True
    goal_found = False # When true, stop searching 
    
    ##############################################################################
        
    initial_node = (0, node_idx, curr_parent_idx, initial_node_coord)
    
    hq.heappush(open_queue, initial_node) 
    hq.heapify(open_queue)
    
    ##############################################################################
            
    # Process next node in queue
    # When all children are checked, remove next top node from data structure
    while (len(open_queue) != 0): # Stop search when node queue is empty 
        
        debug_counter = debug_counter + 1

        curr_node = hq.heappop(open_queue)
        
        visited_queue.append(curr_node) #(node_idx, curr_parent_idx, curr_node))
        
        #######################################################################

        if show_grid == True and debug_counter % 5000 == 0:
            
            print("debug_counter: ", debug_counter)
            print("Current Parent Node:")
            print(curr_node)
            print()
            
            # display_map_grid_plot(map_grid, curr_node[3][0], curr_node[3][1], point_thickness, goal_found, goal_node_coord[0], goal_node_coord[1],  curr_node[3][0],  curr_node[3][1])
        
        #######################################################################
        
        # Evaluate children
        curr_parent_idx = curr_parent_idx + 1 
        i = 1
        
        while i < 9:
                
            cost_to_move, valid_move, child_node_x_valid, child_node_y_valid, is_node_obstacle = generate_child_node(obstacle_matrix, c2c_matrix, curr_node, i, map_grid, map_height, map_width)
                                    
            explored = False
            explored = any(elem[3] == (child_node_x_valid, child_node_y_valid) for elem in visited_queue)
                    
            is_in_open = any(item[3] == (child_node_x_valid, child_node_y_valid) for item in open_queue)
            
            idx = next((idx for idx, item in enumerate(open_queue) if item[3] == (child_node_x_valid, child_node_y_valid)), None)
            child_c2c_val_stored = open_queue[idx][0] if idx is not None else None            
            
            if valid_move == True and explored == False:
                
                is_equal = (child_node_x_valid == goal_node_coord[0] and child_node_y_valid == goal_node_coord[1]) # check if goal node reached
                
                if (is_equal == True): 
                    
                    goal_found = True
                    
                    node_idx = node_idx + 1 
    
                    child_node = (999999999, node_idx, curr_parent_idx,(child_node_x_valid, child_node_y_valid))
                    visited_queue.append(child_node)
                    
                    print("Last Child Node (Goal Node): \n", child_node)
                    print()
                    print("Problem solved, now backtrack to find pathway!")
                    print()
                    print("______________________________________________")
                    print()       
                    
                    return visited_queue, goal_found, fig, ax
                
                else: # Goal state not found yet
                      
                    if (explored == False and is_in_open == False):
                        
                        node_idx = node_idx + 1 
                        
                        child_node = (cost_to_move, node_idx, curr_parent_idx,(child_node_x_valid, child_node_y_valid))
                        
                        hq.heappush(open_queue, child_node) 
                        hq.heapify(open_queue)

                    else: # Just update C2C and parent 
                        
                        if (explored == False and is_in_open == True):
                            
                            cost_to_move_new =  cost_to_move
                                
                            if cost_to_move_new < child_c2c_val_stored:
                                child_node = (cost_to_move_new, node_idx, curr_parent_idx, (child_node_x_valid, child_node_y_valid))
                                open_queue[idx] = child_node
                                hq.heapify(open_queue)
                            
            # else:
            #     print("Move not valid.")
                
            plot_fig = True
            
            # print_function(i, valid_move, is_node_obstacle, plot_fig, map_grid)
                
            i = i + 1

    # End of outter while loop    
    print("Goal Found Boolean: ", goal_found)
    print()
        
    return visited_queue, goal_found, fig, ax
    
#################################################################################################
    
# Function that prints results of eight_puzzle_problem function and times BFS and Traceback operations
def main(): 

    map_grid, map_height, map_width = create_map_grid()
    c2c_matrix = initialize_c2c_matrix(map_grid, map_height, map_width)
    obstacle_matrix = c2c_matrix.copy()
    
    plt.figure()
    plt.title('Initial Map Grid')
    plt.imshow(map_grid.astype(np.uint8), origin="lower")
    plt.show()
    
    ###########################################################################################################

    x_initial, y_initial = eval(input("Enter start node's (x,y) coordinates seperated by comma. Ex: 1,2 "))
    print("Start node x-coordinate:", x_initial)
    print("Start node y-coordinate:",y_initial)
    
    initial_valid = check_node_in_obstacle_space(x_initial, y_initial, obstacle_matrix)
    
    if (initial_valid == True):
        print("Re-enter initial node, coordinates not within freespace.")
        return
    
    x_goal, y_goal = eval(input("Enter goal node's (x,y) coordinates seperated by comma. Ex: 1,2 "))
    print("Goal node x-coordinate:",x_goal)
    print("Goal node y-coordinate:",y_goal)
    print()
    
    goal_valid = check_node_in_obstacle_space(x_goal, y_goal, obstacle_matrix)
    
    if (goal_valid == True):
        print()
        print("Re-enter goal node, coordinates not within freespace.")
        return

    print("########################################################")
    
    initial_node_coord = (x_initial, y_initial)
    goal_node_coord = (x_goal, y_goal)
    
    ###########################################################################################################
    
    # Using different ways to compute time algorithm takes to solve problem
    start1 = time.time()
    start2 = datetime.now()
    
    visited_queue, goal_found, fi, ax = dijkstra_approach_alg(obstacle_matrix, c2c_matrix, initial_node_coord, goal_node_coord, map_grid, map_height, map_width)    
    end1 = time.time()
    end2 = datetime.now()
    
    print("Was goal found ? : ", goal_found)
    print()
        
    #################################################################################################
    # FUNCTION TO GENERATE NODE PATHWAY FROM INITIAL TO GOAL NODE
    # backtrack_node_path_arry = generate_path(node_info, goal_node_idx)
    
    #################################################################################################
    
    # Soure for time functions 
    # https://stackoverflow.com/questions/27779677/how-to-format-elapsed-time-from-seconds-to-hours-minutes
    # -seconds-and-milliseco
    
    # Method 1
    hrs, remain = divmod(end1 - start1, 3600)
    mins, secs = divmod(remain, 60)
    print("- 8 Puzzle Problem solved in (hours:min:sec:milliseconds) (Method 1): {:0>2}:{:0>2}:{:05.2f}".format(int(hrs),int(mins),secs))
    
    # Method 2
    runtime=end2-start2
    print("- 8 Puzzle Problem solved in (hours:min:sec:milliseconds) (Method 2): " + str(runtime))
    print()
    
    print("Start node x-coordinate:", x_initial)
    print("Start node y-coordinate:",y_initial)
    print()
    print("Goal node x-coordinate:",x_goal)
    print("Goal node y-coordinate:",y_goal)
    print()
    
    print("Start node coordinate:", (x_initial,y_initial))
    print()
    print("Goal node coordinate:", (x_goal,y_goal))
    print()
    
    # Create animation visualzation video
    # https://www.rapidtables.com/web/color/RGB_Color.html
    # https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html
    out = cv2.VideoWriter('conn_dijkstra_alg_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 50, (map_width,map_height))

    start_goal_pt_thickness = 3
    for v, visited_node in enumerate(visited_queue):
        # print(visited_node)
        # if v==0:
        #     map_grid = cv2.circle(map_grid, (x_initial,y_initial), radius=0, color=(255, 0, 255), thickness=start_goal_pt_thickness)
        # elif v == len(visited_queue)-1:
        #     map_grid = cv2.circle(map_grid, (x_goal,y_goal), radius=0, color=(0, 255, 0), thickness=start_goal_pt_thickness)
        # else:
        map_grid[visited_node[3][1],visited_node[3][0]] = (0, 255, 255)
        output_frame = cv2.flip(map_grid, 0) # change y axis
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR) # color conversion
        out.write(output_frame)
    
    map_grid = cv2.circle(map_grid, (x_initial,y_initial), radius=0, color=(255, 0, 255), thickness=start_goal_pt_thickness)
    map_grid = cv2.circle(map_grid, (x_goal,y_goal), radius=0, color=(0, 255, 0), thickness=start_goal_pt_thickness)
    output_frame = cv2.flip(map_grid, 0) # change y axis
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR) # color conversion
    out.write(output_frame)
    
    out.release()
    
    # using list comprehension 
    # coord = [v_list[3] for v_list in visited_queue]
    # x_coords = [pt[0] for pt in coord]
    # y_coords = [pt[1] for pt in coord]
    # map_grid[y_coords, x_coords] = (0, 255, 255)
    
    # Display last frame to Spyder Python IDE
    ax.set_title('Final Map Grid')    
    ax.axis('off')
    ax.imshow((map_grid).astype(np.uint8), animated=True, origin="lower")
    
    print("Code Script Complete.")    


# Call main function
main()

# if __name__ == '__main__':
#     cProfile.run('main()')

# END OF SOURCE CODE FILE
#####################################################################################################