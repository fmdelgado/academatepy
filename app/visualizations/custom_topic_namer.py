from turftopic.namers.base import TopicNamer, DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT
from langchain.schema import SystemMessage, HumanMessage, AIMessage

class LocalTopicNamer(TopicNamer):
    """Name topics using a locally running LLM (e.g., ChatOllama).

    Parameters
    ----------
    llm : ChatOllama
        A locally running LangChain-compatible chat model instance.
    prompt_template : str
        Prompt template to use when naming the topic.
    system_prompt : str
        System prompt to use for the language model.
    """

    def __init__(
        self,
        llm,
        prompt_template: str = DEFAULT_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.llm = llm
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt

    def name_topic(
        self,
        keywords: list[list[str]],
    ) -> str:
        # Format the user prompt using the provided keywords
        user_content = self.prompt_template.format(keywords=", ".join(keywords))

        # Construct the message sequence for the chat model
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content)
        ]

        # Call the local LLM
        response = self.llm.invoke(messages)

        # Extract and return the content of the AI's response
        return response.content.strip()

