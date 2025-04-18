from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import tensorflow as tf

# largest number representable exactly as both signed int32 and float32
# (converting int32_max to float32 rounds up one, which causes overflow)
scale = 2147483520


def integer_distance_matrix(distances):
    max_distance = tf.reduce_max(distances)
    factor = scale / max_distance
    scaled = distances * factor
    int_distances = tf.cast(scaled, tf.int32)
    return int_distances.numpy()


def defrag(distances, time_limit=180, log_search=False):
    int_distances = integer_distance_matrix(distances)
    manager = pywrapcp.RoutingIndexManager(int_distances.shape[0], 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def get_route(solution):
        indexes = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            indexes.append(index)
            index = solution.Value(routing.NextVar(index))
        return tf.constant(indexes)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int_distances[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.log_search = log_search
    search_parameters.time_limit.seconds = time_limit

    solution = routing.SolveWithParameters(search_parameters)
    return get_route(solution)
