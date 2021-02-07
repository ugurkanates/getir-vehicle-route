import os
import json
from datetime import datetime, timedelta
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

""" Getir's Vehicles Routing Problem (VRP).
 Ugurkan Ates
 Task started on 04.01.2021
 Task complete on 07.01.2021
 Used Google OR-Tools for defining a unique case. It doesn't directly fit into any of regular problems

 1- First I had to use  data['starts'] - data['ends'] structure. Because our vehicle's have already existing
    locations. With starts not much to do you just plug indexes. Ends is the problem. Our problem should end
    when you visited target locations(jobs). But since this is fixed structure you have to give a location.
    Usually this is depot location since after travel complete they return to refill. But for our problem
    we simply ignore depot since our vehicles are superman basically. Don't need no depot.
    So in order to make it work I defined a dummy depot with cost 0 to all ends. I found this is used in some parts
    of mail-list where they used like this if you wan't ignore endings. So end is set as len(vehicles)*[0] list

    With a setup like this depot will be always end (and won't interfere with solution - because if it did
    0 cost would break it.) Only trick to remove from printing-solution list basically dropping it.

    I also had to add 0 to first element of each row. So indexes get +1 up everywhere but I handled this in code.

 2- Second issue I had was with distance_matrix and target locations of our jobs. Since our problem is not typical
    VRP or TSP you don't have to visit all nodes if not required. If they provide a better solution (somehow 
    if going 1-2-4 is better than going 1-4 then do it obviously) they are used otherwise not. Issue here is 
    there is no easy system for such unique use-case. Because it's only done for assigment sake and doesn't 
    realistically match pickup-deliveries system of real life I had to create a solution for this. 
    What I did end up is using a a similiar to a system 'disjunctions' inside of Ortools. ,
    Which are used in if you want to imagine arbitary  limititatons , perfect match for our use case. 
    I set up this in for loop with manager obj so our vehicles terminate
    the solution/finish when they visit our required job_locations.

    Rest is boilerplate code mostly modified/used from Google Ortools.
    Algorithm used is PATH_CHEAPEST_ARC for finding cheapest distances. I tried others this seems best.


    Instead of HTTP-Microservice I handled file locally
    But easily replacable in input part with requests.get json 

    Cheers, Ugurkan.

    Notice : I didn't do a lot of error checks or test code for inputs or breaking problem basically.
    Notice 2 : I made a debug mode to print more detailed. It's enabled for default but you can disable
    if you only want to see relevant results to you(task output)

    by the way I learned this problem more closely resembles MIP(multi int prob) then a VRP.
    

"""

DEBUG_PRINT = False

"""Parse json """

def parse_json(url=""):
    with open(url, 'r') as f:
        distros_dict = json.load(f)
    return distros_dict
"""Get  starting indexes  from json """
def get_veh_start_pos(jsonObj):
    st_pos_list = []
    # input error check - needs to be done
    for vec in jsonObj:
        st_pos_list.append(vec["start_index"])
    return st_pos_list

"""Get location_index of jobs from json """
def get_target_pos_job(jsonObj):
    tar_pos_list = []
    # input error check - needs to be done
    for vec in jsonObj:
        tar_pos_list.append(vec['location_index'])
    return tar_pos_list

"""Get vehicle ids from json """
def get_veh_ids(jsonObj):
    # -1 added as sentinel value which seemingly required for inactive nodes.
    # https://github.com/google/or-tools/issues/1202

    veh_ind_list = [-1]
    # input error check - needs to be done
    for vec in jsonObj:
        veh_ind_list.append(vec['id'])
    return veh_ind_list

def augment_matrix(snap_json):
    test_mat= snap_json['matrix']
    dummy_row = [[0]*(len(test_mat[0])+1)]
    dummy_row.extend(test_mat)
    # I already added first dummy one.
    for i in range(1,(len(dummy_row))):
        dummy_row[i].insert(0,0) # first element of row-add 0 as int
    return dummy_row

def create_data_model():
    """Stores the data for the problem."""
    snap_json = parse_json("getir.json")
    data = {}


    data['distance_matrix'] = augment_matrix(snap_json)

    data['num_vehicles'] = len(snap_json['vehicles'])
    data['veh_index'] = get_veh_ids(snap_json['vehicles'])
    #data['depot'] = 0
    #data['pickups_deliveries'] = [[7,5]]

    data['starts'] = [sum(x) for x in zip(get_veh_start_pos(snap_json['vehicles']),[1]*len(snap_json['vehicles']))]
    data['ends'] = [0]*len(snap_json['vehicles'])
    data['jobs'] = get_target_pos_job(snap_json['jobs'])
    return data

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0
    """
    Total delivery duration (integer) : sum of estimated delivery durations for
    each order in seconds
    • Dictionary with vehicle ids as keys and routes as a list of ordered order
    ids.
    Example :
    {’1’ : [’1’, ’4’, ’2’] , ’2’: [’3’], ’3’:[] }"""

    ret_arr = dict.fromkeys(range(1,data['num_vehicles']+1))
    route_dist_sum = 0
    for vehicle_id in range(data['num_vehicles']):
        ret_arr[vehicle_id+1] = []
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            ret_arr[vehicle_id+1].append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)

        plan_output += '\n'#.format(manager.IndexToNode(index)) remove dummy depot
        x = '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        if (DEBUG_PRINT == True):
            print(plan_output)
        route_dist_sum += route_distance
        max_route_distance = max(route_distance, max_route_distance)
    if (DEBUG_PRINT == True):
        print('Maximum of the route distances: {}m'.format(max_route_distance))
    return route_dist_sum,ret_arr



def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'],data['starts'],data['ends'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        10000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)
    
    # Define Transportation Requests.
    # [START pickup_delivery_constraint]
    """for request in data['pickups_deliveries']:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(
                delivery_index))
        #routing.solver().Add(
        #    distance_dimension.CumulVar(pickup_index) <=
        #    distance_dimension.CumulVar(delivery_index))
    # [END pickup_delivery_constraint]"""

    # Make other locations not visible to vehicles
    for request in data['jobs']:
        index = manager.NodeToIndex(request)
        routing.VehicleVar(index).SetValues(data['veh_index'])



    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        total_sum,custom_dic = print_solution(data, manager, routing, solution)
    print("Total Seconds",total_sum,"\n Dictionary for each vehicle route",custom_dic)
    if ( DEBUG_PRINT == True):
        print(solution)


if __name__ == '__main__':
    main()


"""
Some of references I used during this.
    # https://github.com/google/or-tools/issues/1202
    # https://stackoverflow.com/questions/65519429/google-or-tools-to-force-disjunctions-prevent-certain-locations-from-being-conne
    # https://github.com/google/or-tools/blob/stable/ortools/constraint_solver/doc/PDP.md
    # https://github.com/google/or-tools/blob/stable/ortools/constraint_solver/samples/vrp_pickup_delivery.py 
    # https://groups.google.com/u/1/g/or-tools-discuss/c/6_qv2U7kpiM  - if start - ends index are same as any of pick-deliver auto segfault
    # https://github.com/google/or-tools/issues/334
    # https://activimetrics.com/blog/ortools/exploring_disjunctions/


"""