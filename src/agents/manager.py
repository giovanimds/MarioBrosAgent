import pickle
from src.helpers.config import NUM_AGENTS
from src.agents.agent import Mario as Agent  # Using Mario as the Agent class

class AgentManager:
    def __init__(self):
        self.agents = [Agent() for _ in range(NUM_AGENTS)]

    def initialize_agents(self):
        for agent in self.agents:
            agent.initialize()

    def run_agents(self):
        for agent in self.agents:
            agent.run()

    def stop_agents(self):
        for agent in self.agents:
            agent.stop()

    def save_agents_state(self, filepath):
        state = [agent.get_state() for agent in self.agents]
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)