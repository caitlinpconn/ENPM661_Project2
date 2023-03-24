# -*- coding: utf-8 -*-

"""
Created on Tue Mar 21 16:05:08 2023

@author: caitlin.p.conn
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt 
import numpy as np 
import cProfile
import math
import cv2 

# Required imported libraries
from collections import deque # speeds up python queue data structure operation
from datetime import datetime
import heapq as hq
import time

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
    map_grid = np.ones((map_height,map_width,3))
    
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

            #print(map_grid[0,559]) #IndexError: index 250 is out of bounds for axis 0 with size 250
                

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


# DIJKSTRA's ALGORITHM APPROACH FOR FORWARD SEARCH POINT ROBOT PROBLEM

###############################################################################

def check_node_in_obstacle_space(child_node_x, child_node_y, obstacle_matrix):
    
    return obstacle_matrix[child_node_y][child_node_x] == -1

###############################################################################
# Function also determines the validity of the swap/action
def generate_child_node(obstacle_matrix, c2c_matrix, parent_node, action, map_grid, map_height, map_width):
    
    valid_move = False # boolean truth value of valid swap
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
        if parent_node_y != map_height: 
            child_node_x = parent_node_x 
            child_node_y = parent_node_y + 1

    elif action == 3: # right (1,0)
        cost_to_move = 1
        if parent_node_x != map_width:
            child_node_x = parent_node_x + 1
            child_node_y = parent_node_y

    elif action == 4: # down (0,-1)
        cost_to_move = 1
        if parent_node_y != 0: 
            child_node_x = parent_node_x 
            child_node_y = parent_node_y - 1

    elif action == 5: # right & up (1,1)
        cost_to_move = 1.4
        if parent_node_x != map_width and parent_node_y != map_height: 
            child_node_x = parent_node_x + 1
            child_node_y = parent_node_y + 1

    elif action == 6: # left & up (-1,1)
        cost_to_move = 1.4
        if parent_node_x != 0 and parent_node_y != map_height: 
            child_node_x = parent_node_x - 1
            child_node_y = parent_node_y + 1
            
    elif action == 7: # right & down (1,-1)
        cost_to_move = 1.4
        if parent_node_x != map_width and parent_node_y != 0: 
            child_node_x = parent_node_x + 1
            child_node_y = parent_node_y - 1

    elif action == 8: # left & down (-1,-1)
        cost_to_move = 1.4
        if parent_node_x != 0 and parent_node_y != 0: 
            child_node_x = parent_node_x - 1
            child_node_y = parent_node_y - 1

    is_node_obstacle = check_node_in_obstacle_space(child_node_x, child_node_y, obstacle_matrix)

    if is_node_obstacle == False:
        valid_move = True
    else:
        valid_move = False
        
    cost_to_move = round(cost_to_move,1)

    # returned node is the resulting child node of the requested action
    return cost_to_move, valid_move, child_node_x, child_node_y, is_node_obstacle

#################################################################################################

# Function uses backtracking to find the node pathway from the initial node to goal node
# Source: https://numpy.org/doc/stable/reference/generated/numpy.fliplr.html
# Must use flipud function to ensure using a forward search strategy!!
def generate_path(visited_queue, initial_node_coord):
            
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

def get_c2c_value(c2c_matrix, map_grid, map_height, map_width, child_x, child_y):
    
    c2c_value = c2c_matrix[child_y,child_x]
    
    return c2c_value

#################################################################################################

def update_child_parent(open_queue, node_idx, curr_parent_idx, child_c2c_val_updated):
    
    for c in range(len(open_queue)):
        if open_queue[c][1] == node_idx:
            # print( open_queue[c])
            # print(type( open_queue[c][2]))
            # print(type())
            tmp_list = list(open_queue[c])
            # print("tmp_list: ", tmp_list)
            tmp_list[2] = curr_parent_idx
            tmp_list[0] = child_c2c_val_updated
            # print("tmp_list: ", tmp_list)
            tmp_tuple = tuple(tmp_list)
            # print("tmp_tuple: ", tmp_tuple)
            
            open_queue[c] = tmp_tuple

            #open_queue[c][2] = curr_parent_idx
            break
        
    # c = 0  
    # open_queue[c][2] = [curr_parent_idx for c in range (len(open_queue)) if open_queue[c][1] == node_idx]
   
    # open_dict = {node[1]: {'parent_idx': node[2]} for node in open_queue}
    # if node_idx in open_dict:
    #     # open_dict[node_idx]['parent_idx'] = curr_parent_idx
    #     print(len(open_queue))
    #     print(node_idx)
    #     t = open_queue[node_idx]
    #     tmp_list = list(open_queue[node_idx])
    #     tmp_list[2] = curr_parent_idx
    #     tmp_list[0] = child_c2c_val_updated
    #     tmp_tuple = tuple(tmp_list)
    #     open_queue[node_idx] = tmp_tuple
    #     #open_queue[node_idx][2]  = open_dict[node_idx]['parent_idx']
    
    # print("Node updated")
    # print()
    
    return

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

def print_function(i, valid_move, is_node_obstacle, case_type, plot_fig, map_grid):
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

    if case_type == 1:
        print("Node will be colored blue")
    elif case_type ==  2:
        print("Node in open, but not explored, will be colored blue")
    
    print()
        
    if plot_fig == True:
        plt.figure()
        plt.title('Node Explored')
        plt.imshow(map_grid.astype(np.uint8), origin="lower")
        plt.show()
        
    return
        
#################################################################################################
                
# Main function that calls subfunctions that perform search operations
def dijkstra_approach_alg(obstacle_matrix, c2c_matrix, initial_node_coord, goal_node_coord, map_grid, map_height, map_width):
    
    ##############################################################################
    
    fig, ax = plt.subplots()
    
    point_thickness = 5 # Thickness of initial and goal node circle pointer
    curr_parent_idx = 0 # Parent index
    debug_counter = 0
    node_idx = 1 # WAS 1 # Node index 
    
    ims = []

    # Create empty queues
    visited_queue = [] # explored, valid nodes
    open_queue = [] # keeps track of node queue to be processed
    
    show_grid = True
    goal_found = False # When true, stop searching 
    
    ##############################################################################
    
    # height = map_height # height of image shape 
    # width = map_width   # width of image shape 
    # size = (width, height) # size = (height, width) 
    
    
    # Define the codec and create VideoWriter object
    # out = cv2.VideoWriter('new_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 2, size) 
    # out = cv2.VideoWriter('new_video2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 2, size) 
    
    ##############################################################################
        
    initial_node = (0, node_idx, curr_parent_idx, initial_node_coord)
    
    hq.heappush(open_queue, initial_node) 
    hq.heapify(open_queue)
    
    # print("First node appended: \n", hq.heappop(open_queue))
    # print()
    
    ##############################################################################
            
    # Process next node in queue
    # When all children are checked, remove next top node from data structure
    while (len(open_queue) != 0): # Stop search when node queue is empty 
        
        curr_node = hq.heappop(open_queue)
        
        # print("Node Popped from of Num Queue, Curr Length: ", len(open_queue))
        # print()

        # print("Parent IDX: ", curr_parent_idx)
        # print()
        
        # print("Current Parent Node:")
        # print(curr_node)
        # print()
        
        debug_counter = debug_counter + 1

        # if debug_counter % 5000 == 0:
        #     print("debug_counter: ", debug_counter)
        #     print("Current Parent Node:")
        #     print(curr_node)
        #     print()

        visited_queue.append(curr_node) #(node_idx, curr_parent_idx, curr_node))
        map_grid[curr_node[3][1], curr_node[3][0]] = (0,255,255)
        map_grid = cv2.circle(map_grid, initial_node_coord, radius=0, color=(255, 0, 255), thickness=point_thickness)
  
        ax.axis('off')
        im = ax.imshow((map_grid).astype(np.uint8), animated=True, origin="lower")
        ims.append([im])
        
        
        # plt.figure()
        # plt.title('Node Explored')
        # plt.imshow(map_grid.astype(np.uint8), origin="lower")
        # plt.show()
        
        # parent_is_equal = (curr_node[3][0] == goal_node_coord[0] and curr_node[3][1] == goal_node_coord[1]) # check if goal node reached

        if show_grid == True and debug_counter % 5000 == 0:
            
            print("debug_counter: ", debug_counter)
            print("Current Parent Node:")
            print(curr_node)
            print()
            
            # display_map_grid_plot(map_grid, curr_node[3][0], curr_node[3][1], point_thickness, goal_found, goal_node_coord[0], goal_node_coord[1],  curr_node[3][0],  curr_node[3][1])
    
        # Evaluate children
        # node_idx = node_idx + 1 
        curr_parent_idx = curr_parent_idx + 1 
        found = 0
        i = 1
        
        while i < 9:
            
            case_type = 0

            if found == 1:
                break
                
            cost_to_move, valid_move, child_node_x_valid, child_node_y_valid, is_node_obstacle = generate_child_node(obstacle_matrix, c2c_matrix, curr_node, i, map_grid, map_height, map_width)
                                    
            if valid_move == True:
                
                # print("Valid Move Boolean -> True")
                # print("Child Node X-Coord: ", child_node_x_valid)
                # print("Child Node Y-Coord: ", child_node_y_valid)
                
                is_equal = (child_node_x_valid == goal_node_coord[0] and child_node_y_valid == goal_node_coord[1]) # check if goal node reached
                # print()
                # print("- - - - - - - - - - - - - - - - - - ")
                # print()
                # print("Equal to Goal State ? : ", is_equal)
                # print()
                
                if (is_equal == True): 
                    
                    found = 1
                    goal_found = True
                    
                    node_idx = node_idx + 1 
                        
                    parent_c2c_val_stored = get_c2c_value(c2c_matrix, map_grid, map_height, map_width, curr_node[3][0], curr_node[3][1])
                    if parent_c2c_val_stored == np.inf:
                        parent_c2c_val_stored = 0
                    parent_c2c_val_stored = round(parent_c2c_val_stored,1)
                        
                    child_c2c_val_updated =  parent_c2c_val_stored + cost_to_move
                    child_c2c_val_updated = round(child_c2c_val_updated,1)
    
                    child_node = (child_c2c_val_updated, node_idx, curr_parent_idx,(child_node_x_valid, child_node_y_valid))
                    visited_queue.append(child_node)
                    
                    #map_grid[child_node[3][1], child_node[3][0]] = (0,255,255)
                    map_grid = cv2.circle(map_grid, (child_node[3][0], child_node[3][1]), radius=0, color=(0, 0, 255), thickness=point_thickness)
                    
                    backtrack_path_list = generate_path(visited_queue, initial_node_coord)
                    
                    # for path_elem in backtrack_path_list:
                    #     map_grid[path_elem[3][1], path_elem[3][0]] = (255,0,0)
                       
                    # Draw backtrack path lines
                    for curr_elem in range(len(backtrack_path_list)-1):
                        next_elem = curr_elem + 1
                        map_grid = cv2.line(map_grid, backtrack_path_list[curr_elem][3], backtrack_path_list[next_elem][3], (255,0,0), 1)

                    ax.axis('off')
                    plt.title("Final Map Frame")
                    plt.figure()
                    im = ax.imshow((map_grid).astype(np.uint8), animated=True, origin="lower")
                    
                    ims.append([im])
                    
                    print("Goal Node, Node will be colored dark blue")
                    print("Last Child Node (Goal Node): \n", child_node)
                    print()
                    print("##############################################")
                    print()
                    print("Problem solved, now backtrack to find pathway!")
                    print()
                    print("______________________________________________")
                    print()       
                    
                    return visited_queue, goal_found, fig, ims
                
                else: # Goal state not found yet
                    
                    # print("child_node_x_valid: ", child_node_x_valid)
                    # print("child_node_y_valid: ", child_node_y_valid)
                    
                    # Check to see if current state or node has been visited already
#                    explored = False
#                    is_in_open = False
#                    for j in range (len(visited_queue)):
#                        if (visited_queue[j][3] == (child_node_x_valid, child_node_y_valid)): # check node indices
#                            explored = True
#                            break
#                        else:
#                            explored = False
                      
                    explored = False
#                    visited_nodes = set()
#
#                    for j in range(len(visited_queue)):
#                        if (child_node_x_valid, child_node_y_valid) in visited_nodes:
#                            explored = True
#                            break
#                        else:
#                            explored = False
                            
                    explored = any(elem[3] == (child_node_x_valid, child_node_y_valid) for elem in visited_queue)

                            
#                    for k in range (len(open_queue)):
#                        if (open_queue[k][3] == (child_node_x_valid, child_node_y_valid)): # check node indices
#                            is_in_open = True
#                            break
#                        else:
#                            is_in_open = False
                            
                    is_in_open = any(item[3] == (child_node_x_valid, child_node_y_valid) for item in open_queue)


                    parent_c2c_val_stored = get_c2c_value(c2c_matrix, map_grid, map_height, map_width, curr_node[3][0], curr_node[3][1])
                    if parent_c2c_val_stored == np.inf:
                        parent_c2c_val_stored = 0
                        
                    parent_c2c_val_stored = round(parent_c2c_val_stored,1)
                        
                    child_c2c_val_stored = get_c2c_value(c2c_matrix, map_grid, map_height, map_width, child_node_x_valid, child_node_y_valid)
                    if child_c2c_val_stored == np.inf:
                        child_c2c_val_stored = 0
                    #manhat_dist = compute_manhat_dist_between_nodes(child_node_x_valid, child_node_y_valid, curr_node[3][0], curr_node[3][1])
                    
                    child_c2c_val_stored = round(child_c2c_val_stored,1)

                    if (explored == False and is_in_open == False and is_node_obstacle == False):
                        
                        node_idx = node_idx + 1 
                        
                        #print("Child not yet visited/explored, open_queue appended")
                        child_c2c_val_updated =  parent_c2c_val_stored + cost_to_move
                        
                        child_c2c_val_updated = round(child_c2c_val_updated,1)
                        
                        #child_c2c_val_updated =  parent_c2c_val_stored + manhat_dist
                        c2c_matrix[child_node_y_valid][child_node_x_valid] = child_c2c_val_updated
                        child_node = (child_c2c_val_updated, node_idx, curr_parent_idx,(child_node_x_valid, child_node_y_valid))
                        open_queue.append(child_node)
                        
                        #print("Node will be colored blue")
                        case_type = 1

                        # node_idx = node_idx + 1 

                    else: # Just update C2C and parent 
                        
                        if (explored == False and is_in_open == True):
                            #print("Child already visited/explored, open_queue stays same, C2C UPDATED")
                            child_new_c2c_val = parent_c2c_val_stored + cost_to_move
                            #child_new_c2c_val = parent_c2c_val_stored + manhat_dist
                            
                            child_c2c_val_updated = round(child_c2c_val_updated,1)

                            if child_new_c2c_val < child_c2c_val_stored:
                                child_c2c_val_updated = child_new_c2c_val
                                c2c_matrix[child_node_y_valid][child_node_x_valid] = child_c2c_val_updated
                                update_child_parent(open_queue, node_idx, curr_parent_idx, child_c2c_val_updated)
                                
                            
                            # print("Node in open, but not explored, will be colored blue") 
                            case_type = 2
                        
            # else:
            #     print("Move not valid.")
                
            plot_fig = True
            
            # print_function(i, valid_move, is_node_obstacle, case_type, plot_fig, map_grid)
                
            # print("is_node_obstacle: ", is_node_obstacle)

            ##################################################################################
            # Verification Statements
            # print("Length of Open Queue: ", len(open_queue))
            # print("Length of Visited Queue: ", len(visited_queue))
            # print("Goal Found Boolean: ", goal_found)
            # print()

            # print("Parent IDX: ", curr_parent_idx)
            # if valid_move == True:
            #     print("Current Valid Child IDX: ", node_idx)

            i = i + 1
            case_type = 0
            
            # if i != 9:
            #     print()

            # print("#################################################################")
            # print()

    # End of outter while loop    
    print("Goal Found Boolean: ", goal_found)
    print()
    
    #plt.figure()
    #plt.imshow(map_grid)
        
    return visited_queue, goal_found, fig, ims
    
#################################################################################################
    
# Function that prints results of eight_puzzle_problem function and times BFS and Traceback operations
def main(): 

    map_grid=[]
    map_grid, map_height, map_width = create_map_grid()
    c2c_matrix = initialize_c2c_matrix(map_grid, map_height, map_width)
    obstacle_matrix = c2c_matrix.copy()
    
    # print(np.shape(obstacle_matrix))
    
    # rval = np.shape(obstacle_matrix)[0]
    # cval = np.shape(obstacle_matrix)[1]
    # for rv in range(rval):
    #     for cv in range(cval):
    #         print("Row:", rv, ",", "Column:", cv, ", Cost:", obstacle_matrix[rv][cv])
    #     print()
        
    # import pandas as pd
    # print(pd.DataFrame(obstacle_matrix))    
    
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
    
    visited_queue, goal_found, fig, ims = dijkstra_approach_alg(obstacle_matrix, c2c_matrix, initial_node_coord, goal_node_coord, map_grid, map_height, map_width)
    
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
    
    print("Working on saving animation")

    ani = animation.ArtistAnimation(fig, ims, interval=2-0, blit=True, repeat_delay=1000)
    ani.save("movie3.gif")
    print("Animation done")
  
    print("DONE")
    
    # print(np.shape(ims[0].shape))
    

# Call main function
main()


# if __name__ == '__main__':
#     cProfile.run('main()')

# END OF SOURCE CODE FILE
#####################################################################################################