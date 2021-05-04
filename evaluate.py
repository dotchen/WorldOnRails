from runners import ChallengeRunner

def main(args):

    scenario = 'assets/all_towns_traffic_scenarios.json'
    # scenario = 'assets/no_scenarios.json'
    # route = 'assets/routes_dev.xml'
    route = 'assets/routes_training/route_10.xml'

    args.agent = 'autoagents/image_agent'
    # args.agent = 'autoagents/collector_agents/lidar_q_collector'
    args.agent_config = 'config.yaml'
    # args.agent_config = 'config_lidar.yaml'

    port = args.port
    tm_port = port + 2
    runner = ChallengeRunner(args, scenario, route, port=port, tm_port=tm_port)
    runner.run()



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--timeout', default="60.0",
                        help='Set the CARLA client timeout value in seconds')

    parser.add_argument('--port', type=int, default=2000)

    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')
    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    args = parser.parse_args()

    main(args)
