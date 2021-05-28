import csv
import numpy as np
from tabulate import tabulate
from itertools import product
from collections import defaultdict

TOWNS = ['Town01', 'Town02']
TRAFFICS = ['empty', 'regular', 'dense']
WEATHERS = {
    1: 'train', 3: 'train', 6: 'train', 8: 'train',
    10: 'test', 14: 'test',
}

def parse_results(path):
    
    finished = defaultdict(lambda: [])
    
    with open(path+'.csv', 'r') as file:
        log = csv.DictReader(file)
        for row in log:
            finished[(
                row['town'],
                int(row['traffic']),
                WEATHERS[int(row['weather'])],
            )].append((
                float(row['route_completion']),
                int(row['lights_ran']),
                float(row['duration'])
            ))
    
    
    for town, weather_set in product(TOWNS, set(WEATHERS.values())):
        
        output = "\n"
        output += "\033[1m========= Results of {}, weather {} \033[1m=========\033[0m\n".format(
                town, weather_set)
        output += "\n"
        list_statistics = [['Traffic', *TRAFFICS], [args.metric]+['N/A']*len(TRAFFICS),['Duration']+['N/A']*len(TRAFFICS)]
    
        
        for traffic_idx, traffic in enumerate(TRAFFICS):
            runs = finished[town,TRAFFICS.index(traffic),weather_set]

            if len(runs) > 0:
                route_completion, lights_ran, duration = zip(*runs)

                mean_lights_ran = np.array(lights_ran)/np.array(duration)*3600

                if args.metric == 'Success Rate':
                    list_statistics[1][traffic_idx+1] = "{}%".format(100*round(np.mean(np.array(route_completion)==100), 2))
                elif args.metric == 'Route Completion':
                    list_statistics[1][traffic_idx+1] = "{}%".format(round(np.mean(route_completion), 2))
                elif args.metric == 'Lights Ran':
                    list_statistics[1][traffic_idx+1] = "{} per hour".format(round(np.mean(mean_lights_ran), 2))
                
                list_statistics[2][traffic_idx+1] = "{}s".format(round(np.mean(duration), 2))
                
        output += tabulate(list_statistics, tablefmt='fancy_grid')
        output += "\n"

        print(output)
            
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--metric', default='Success Rate', choices=[
        'Success Rate', 'Route Completion', 'Lights Ran'
    ])
    
    args = parser.parse_args()
    
    parse_results(args.config)
