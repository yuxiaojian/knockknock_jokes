from typing import Annotated, Sequence, TypedDict, Literal, Sequence, Optional
from uuid import uuid4

import chromadb
from chromadb.config import Settings

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

import pprint

from config import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    OPENAI_MODEL,
    logging,
)

# TypedDict
class Joke(TypedDict):
    """Joke to tell user."""
    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class JokeAgent():
    """
    Agent that tells a joke.
    """
    def __init_vector_store(self):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        chroma_client = chromadb.Client(settings=CHROMA_SETTINGS)
        vector_store = Chroma(client=chroma_client, collection_name="jokes", embedding_function=embeddings)

        # Add some documents to the vector store
        document_1 = Document(
            page_content="Caterpillar. Caterpillar really slow, but watch me turn into a butterfly and steal the show.",
            id=1,
        )

        document_2 = Document(
            page_content="Cargo. Cargo 'vroom vroom', but planes go 'zoom zoom'!",
            id=2,
        )

        document_3 = Document(
            page_content="Why did the tornado break up with the hurricane? Because it couldn't handle the emotional whirlwind!",
            id=3,
        )

        documents = [
            document_1,
            document_2,
            document_3
        ]

        uuids = [str(uuid4()) for _ in range(len(documents))]

        vector_store.add_documents(documents=documents, ids=uuids)

        return vector_store
    
    def __init_graph(self):
        """
        Creates the graph for the agent.
        """
        workflow = StateGraph(AgentState)

        # Add the nodes
        workflow.add_node("agent", self.__agent)
        workflow.add_node("joke_teller", self.__joke_teller)
        workflow.add_node("rewrite", self.__rewrite)

        # Add the edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            self.__need_a_joke,
        )

        workflow.add_conditional_edges(
            "joke_teller",
            # Assess agent decision
            self.__grade_jokes,
        )

        workflow.add_edge("rewrite", "joke_teller")

        graph = workflow.compile()
        return graph


    def __init__(self) -> None:
        self.__vector_store= self.__init_vector_store()
        self.__retriever = self.__vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": 20}
            )
        self.graph = self.__init_graph()
    
    # initial agent node
    def __agent(self, state):
        """
        Determines whether if user wants a joke or else

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        logging.info("---CALL AGENT---")
        
        class Joke_Or_Answer(TypedDict):
            """A joke or an answer."""
        
            binary_score: Annotated[str, ..., "If the user needs a joke 'yes', or 'no'"]
            starter_joke: Annotated[str, ..., "Generate a starter joke"]
            answer: Annotated[str, ..., "If the user didn't ask a joke, provide the answer to the input"]

        messages = state["messages"]
        # LLM
        model = ChatOpenAI(temperature=0, streaming=True).with_structured_output(Joke_Or_Answer)

        # Prompt
        prompt = PromptTemplate(
            template=""" 
            Here are the user query: \n\n {query} \n\n
            Set 'binary_score' to 'yes' if the user wants a joke or set it to 'no' \
            Create a starter joke to 'starter_joke' if 'binary_score' is 'yes' \
            Answer the user query to 'answer' if 'binary_score' is not 'yes'
            """,
            input_variables=["query"],
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt 
            | model )

        response = chain.invoke(messages[0].content)
        
        score = response['binary_score']
        if score == "yes":
            logging.info("---DECISION: NEED A JOKE ---")
            msg = [
                HumanMessage(
                    content=f"{messages[0].content} but not the same as '{response['starter_joke']}'",
                )
            ]
            return {"messages": msg}

        else:
            logging.info("---DECISION: ANSWER FROM AGENT ---")
            return {"messages": [ AIMessage(content=f"{response['answer']}") ]}
        
    # Edge to decide whether need a joke or not
    def __need_a_joke(self, state) -> Literal["joke_teller",END]:

        messages = state["messages"]
        last_message = messages[-1]
        if last_message.type == "human":
            return "joke_teller"
        else:
            return END

    # Joke Teller node
    def __joke_teller(self, state):
        """
        Invokes the Joke Teller to tell a knock knock joke

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        logging.info("---CALL JOKE TELLER---")
        messages = state["messages"]

        system = """You are a hilarious comedian. Your specialty is telling jokes. \
        Return a joke that has the setup  and the final punchline.
        
        Here are some examples of jokes:
        
        example_user: Tell me a joke about planes
        example_assistant: {{"setup": "Why don't planes ever get tired?", "punchline": "Because they have rest wings!", "rating": 2}}

        example_user: Tell me another joke about planes
        example_assistant: {{"setup": "Cargo", "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!", "rating": 10}}

        example_user: Now about caterpillars
        example_assistant: {{"setup": "Caterpillar", "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!", "rating": 5}}

        
        DO NOT repeat old jokes:\n
        {old_jokes}
        """

        # Create a prompt that asks the user for a joke. temperature=0.5 to bring in some randomness
        model = ChatOpenAI(temperature=0.9, streaming=True, model=OPENAI_MODEL).with_structured_output(Joke)
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{user_input}")])

        chain =  prompt | model
        response = chain.invoke({"old_jokes": messages[-1].content, "user_input": messages[0].content})
        # Add the response to the messages
        return {"messages": [ AIMessage(content=f"{response['setup']} {response['punchline']}") ]}
    
    # Edge to decide whether a repetitive
    def __grade_jokes(self, state)-> Literal["rewrite", END]:
        """
        Determines whether the generated jokes are a repetition.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        logging.info("---CHECK DUPLICATION---")

        # Data model
        class grade(BaseModel):
            """Binary score for relevance check."""
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        # LLM
        model = ChatOpenAI(temperature=0, streaming=True)

        # LLM with tool and validation
        llm = model.with_structured_output(grade)

        # Prompt
        prompt = PromptTemplate(
            template="""You are an audience assessing if the new joke is a repetition or duplication of old jokes. \n 
            Here are the old jokes: \n\n {old_jokes} \n\n
            Here is the new joke: {new_joke} \n
            Give a binary score 'yes' or 'no' score to indicate whether the new joke is a repetition of the old jokes.""",
            input_variables=["old_jokes", "new_joke"],
        )

        messages = state["messages"]
        new_joke = messages[-1].content

        chain = (
            {"old_jokes": self.__retriever | format_docs, "new_joke": RunnablePassthrough()}
            | prompt 
            | llm )

        scored_result = chain.invoke(new_joke)

        score = scored_result.binary_score
        #score = "yes"

        if score == "yes":
            logging.info("---DECISION: A DUPLICATE JOKE ---")
            return "rewrite"

        else:
            logging.info("---DECISION: A NEW JOKE ---")
            # add the new joke to the vector store
            self.__vector_store.add_documents(documents=[Document(page_content=new_joke)], ids=str(uuid4()))
            return END
        
    def __rewrite(self, state):
        """
        Return old jokes to produce a better joke.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with old jokes
        """

        logging.info("---REWRITE THE JOKE---")
        messages = state["messages"]
        new_joke = messages[-1].content
        logging.info(f"new_joke:{new_joke}")

        old_jokes = format_docs(self.__retriever.invoke(new_joke))
        logging.info(f"old_jokes:{old_jokes}")
        # question = messages[0].content
        
        msg = [
            HumanMessage(
                content=f"""{old_jokes}""",
            )
        ]

        return {"messages": msg }


if __name__ == '__main__':
    # Initialize the agent
    agent = JokeAgent()

    inputs = {
        "messages": [
            ("user", "why so many gum trees in australia?"),
        ]
    }

    for output in agent.graph.stream(inputs):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")