from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import OpenAIAgentTool  # Import OpenAI tool if needed for tasks

@CrewBase
class BiologyPromptCrew():
    """BiologyPrompt crew for generating and validating biology-related prompts."""

    @agent
    def navigator(self) -> Agent:
        """Skill Identification Agent"""
        return Agent(
            config=self.agents_config['navigator'],  # Load Navigator config from YAML
            verbose=True
        )

    @agent
    def creator(self) -> Agent:
        """Prompt Generation Agent"""
        return Agent(
            config=self.agents_config['creator'],  # Load Creator config from YAML
            verbose=True,
            tools=[OpenAIAgentTool()]  # Tool for GPT integration
        )

    @agent
    def inspector(self) -> Agent:
        """Rubric Validation Agent"""
        return Agent(
            config=self.agents_config['inspector'],  # Load Inspector config from YAML
            verbose=True
        )

    @task
    def navigator_task(self) -> Task:
        """Task for skill identification."""
        return Task(
            config=self.tasks_config['navigator_task'],  # Load task config from YAML
        )

    @task
    def creator_task(self) -> Task:
        """Task for prompt generation."""
        return Task(
            config=self.tasks_config['creator_task'],  # Load task config from YAML
        )

    @task
    def inspector_task(self) -> Task:
        """Task for rubric validation."""
        return Task(
            config=self.tasks_config['inspector_task'],  # Load task config from YAML
        )

    @crew
    def crew(self) -> Crew:
        """Creates the BiologyPrompt crew."""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,  # Run tasks sequentially
            verbose=True,
        )
