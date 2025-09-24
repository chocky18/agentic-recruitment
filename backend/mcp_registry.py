# backend/mcp_registry.py
class MCPRegistry:
    def __init__(self):
        self.agents = {}

    def register(self, name, agent):
        self.agents[name] = agent

    def call(self, name, *args, **kwargs):
        if name in self.agents:
            return self.agents[name].run(*args, **kwargs)
        return "Agent not found!"

# Register agents
registry = MCPRegistry()
registry.register("resume_upload", ResumeUploadAgent())
registry.register("jd_generator", JDGeneratorAgent())
registry.register("candidate_matcher", CandidateMatcherAgent())
