from agent.agent_partae import PartAEAgent
from agent.agent_scene import GraphAE
from agent.pqnet import PQNET


def get_agent(config):
    if config.module == 'part_ae':
        return PartAEAgent(config)
    elif config.module == 'graph':
        return GraphAE(config)
    else:
        raise ValueError

