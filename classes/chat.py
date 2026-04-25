import streamlit as st
from openai import OpenAI
from itertools import groupby
from types import GeneratorType
import pandas as pd
import json
from classes.description import TeamDescription, TeamStyleDescription
from classes.embeddings import TeamEmbeddings

from settings import USE_GEMINI
if USE_GEMINI:
    from settings import USE_GEMINI, GEMINI_API_KEY, GEMINI_CHAT_MODEL

from settings import USE_GEMINI, USE_LM_STUDIO

if USE_GEMINI:
    from settings import GEMINI_API_KEY, GEMINI_CHAT_MODEL
elif USE_LM_STUDIO:
    from settings import LM_STUDIO_API_KEY, LM_STUDIO_CHAT_MODEL, LM_STUDIO_API_BASE
else:
    from settings import (
        GPT_BASE,
        GPT_KEY,
        GPT_CHAT_MODEL,
        GPT_SUPPORTS_REASONING,
        GPT_AVAILABLE_REASONING_EFFORTS,
        GPT_SUPPORTS_TEMPERATURE,
    )

from classes.description import (
    PlayerDescription,
    CountryDescription,
    PersonDescription,
)
from classes.embeddings import PlayerEmbeddings, CountryEmbeddings, PersonEmbeddings

from classes.visual import Visual, DistributionPlot, DistributionPlotPersonality

import utils.sentences as sentences
from utils.gemini import convert_messages_format

# Helper function to clean metric names for better display
def clean_metric_name(name: str) -> str:
    name = name.replace("_", " ")
    if "pct" in name:
        name = name.replace("pct", "").strip()
        name = f"{name} (%)"
    return name

class Chat:
    function_names = []

    def __init__(self, chat_state_hash, state="empty"):

        if (
            "chat_state_hash" not in st.session_state
            or chat_state_hash != st.session_state.chat_state_hash
        ):
            # st.write("Initializing chat")
            st.session_state.chat_state_hash = chat_state_hash
            st.session_state.messages_to_display = []
            st.session_state.chat_state = state
        if isinstance(self, PlayerChat):
            self.name = self.player.name
        elif isinstance(self, PersonChat):
            self.name = self.person.name
        else:
            pass

        # Set session states as attributes for easier access
        self.messages_to_display = st.session_state.messages_to_display
        self.state = st.session_state.chat_state

    def instruction_messages(self):
        """
        Sets up the instructions to the agent. Should be overridden by subclasses.
        """
        return []

    def add_message(self, content, role="assistant", user_only=True, visible=True):
        """
        Used by app.py to start off the conversation with plots and descriptions.
        """
        message = {"role": role, "content": content}
        self.messages_to_display.append(message)

    # def get_input(self):
    #     """
    #     Get input from streamlit."""

    #     if x := st.chat_input(
    #         placeholder=f"What else would you like to know about {self.player.name}?"
    #     ):
    #         if len(x) > 500:
    #             st.error(
    #                 f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
    #             )

    #         self.handle_input(x)

    def handle_input(self, input, reasoning_effort=None, temperature=1, stream=False):
        """
        The main function that calls the GPT-4 API and processes the response.
        """

        # Get the instruction messages.
        messages = self.instruction_messages()

        # Add a copy of the user messages. This is to give the assistant some context.
        messages = messages + self.messages_to_display.copy()

        # Get relevant information from the user input and then generate a response.
        # This is not added to messages_to_display as it is not a message from the assistant.
        get_relevant_info = self.get_relevant_info(input)

        # Now add the user input to the messages. Don't add system information and system messages to messages_to_display.
        self.messages_to_display.append({"role": "user", "content": input})

        messages.append(
            {
                "role": "user",
                "content": f"Here is the relevant information to answer the users query: {get_relevant_info}\n\n```User: {input}```",
            }
        )

        # Remove all items in messages where content is not a string
        messages = [
            message for message in messages if isinstance(message["content"], str)
        ]

        # Show the messages in an expander
        st.expander("Chat transcript", expanded=False).write(messages)

        # Check if use gemini is set to true
        if USE_GEMINI:
            import google.generativeai as genai

            converted_msgs = convert_messages_format(messages)

            # # save converted messages to json
            # with open("data/wvs/msgs_1.json", "w") as f:
            #     json.dump(converted_msgs, f)

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"],
            )
            chat = model.start_chat(history=converted_msgs["history"])
            response = chat.send_message(content=converted_msgs["content"])

            answer = response.text
        elif USE_LM_STUDIO:
            client = OpenAI(api_key=LM_STUDIO_API_KEY, base_url=LM_STUDIO_API_BASE)
            if stream:
                # Collect chunks eagerly so the generator over the list is
                # near-instantaneous — preventing Streamlit re-runs from
                # hitting the same generator while it is still executing.
                chunks = [
                    chunk.choices[0].delta.content
                    for chunk in client.chat.completions.create(
                        model=LM_STUDIO_CHAT_MODEL,
                        messages=messages,
                        temperature=temperature,
                        stream=True,
                    )
                    if chunk.choices and chunk.choices[0].delta.content
                ]

                def streamed_chunks():
                    yield from chunks

                answer = streamed_chunks()
            else:
                response = client.chat.completions.create(
                    model=LM_STUDIO_CHAT_MODEL,
                    messages=messages,
                    temperature=temperature,
                )
                answer = response.choices[0].message.content
        else:
            client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)
            if stream:
                if GPT_SUPPORTS_REASONING:
                    reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        reasoning={"effort": reasoning_effort},
                        stream=True,
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        temperature=temperature,
                        stream=True,
                    )
                else:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        stream=True,
                    )

                def streamed_chunks():
                    for event in response_stream:
                        if event.type == "response.output_text.delta":
                            yield event.delta

                answer = streamed_chunks()
            else:
                if GPT_SUPPORTS_REASONING:
                    reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        reasoning={"effort": reasoning_effort},
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        temperature=temperature,
                    )
                else:
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                    )

                answer = response.output_text
        message = {"role": "assistant", "content": answer}

        # Add the returned value to the messages.
        self.messages_to_display.append(message)

    def display_content(self, content):
        """
        Displays the content of a message in streamlit. Handles plots, strings, and StreamingMessages.
        """
        if isinstance(content, str):
            st.write(content)

        # Visual
        elif isinstance(content, Visual):
            content.show()

        else:
            # So we do this in case
            try:
                content.show()
            except:
                try:
                    st.write(content.get_string())
                except:
                    raise ValueError(
                        f"Message content of type {type(content)} not supported."
                    )

    def display_messages(self):
        """
        Displays visible messages in streamlit. Messages are grouped by role.
        If message content is a Visual, it is displayed in a st.columns((1, 2, 1))[1].
        If the message is a list of strings/Visuals of length n, they are displayed in n columns.
        If a message is a generator, it is displayed with st.write_stream
        Special case: If there are N Visuals in one message, followed by N messages/StreamingMessages in the next, they are paired up into the same N columns.
        """
        # Group by role so user name and avatar is only displayed once

        # st.write(self.messages_to_display)

        for key, group in groupby(self.messages_to_display, lambda x: x["role"]):
            group = list(group)

            if key == "assistant":
                avatar = "data/ressources/img/twelve_chat_logo.svg"
            else:
                try:
                    avatar = st.session_state.user_info["picture"]
                except:
                    avatar = None

            message_block = st.chat_message(name=key, avatar=avatar)
            with message_block:
                for message in group:
                    content = message["content"]
                    if isinstance(content, GeneratorType):
                        final_text = st.write_stream(content)
                        message["content"] = final_text
                    else:
                        self.display_content(content)

    def save_state(self):
        """
        Saves the conversation to session state.
        """
        st.session_state.messages_to_display = self.messages_to_display
        st.session_state.chat_state = self.state


class PlayerChat(Chat):
    tools = [
        {
            "type": "function",
            "name": "get_player_summary",
            "description": "Returns a data-driven statistical summary of the selected player.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "search_football_knowledge",
            "description": "Searches a knowledge base for information relevant to a question about data analytics in football, especially about forwards.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    ]

    def __init__(self, chat_state_hash, player, players, state="empty"):
        self.embeddings = PlayerEmbeddings()
        self.player = player
        self.players = players
        super().__init__(chat_state_hash, state=state)

    def _get_player_summary(self):
        return PlayerDescription(self.player).synthesize_text()

    def _search_knowledge(self, query):
        results = self.embeddings.search(query, top_n=5)
        return "\n".join(results["assistant"].to_list())

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.player.name}?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x, stream=True)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        if USE_GEMINI or USE_LM_STUDIO:
            first_messages = [
            {"role": "system", "content": "You are a UK-based football scout."},
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of a football scouting platform. "
                    f"The user has selected the player {self.player.name}, and the conversation will be about them. "
                    "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query  to provide 2 sentence answers."
                    " Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
            return first_messages
        else:
            return [
                {
                    "role": "system",
                    "content": (
                        "You are a UK-based football scout. "
                        f"The user has selected the player {self.player.name}, and the conversation will be about them. "
                        "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                        "Choose the tool that best fits the user's query to respond."
                        "- If the user is asking for information about the player, use the get_player_summary function. "  
                        "- If the user is asking for general football knowledge, use the search_football_knowledge function. "
                        "- If none of the tools are relevant to the user's query, respond directly to the user that the question is outside your scope. "
                        "- If the user asks about a different player, respond that you can only answer questions about the selected player and if they want information about a different player, they need to select that player first on the sidebar."
                        "All user messages will be prefixed with 'User:' and enclosed with ```. "
                        "When responding to the user, speak directly to them. "
                        "Use the information provided before the query to provide 2 sentence answers."
                        "Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                    ),
                }
            ]

    def handle_input(self, input, reasoning_effort=None, temperature=1, stream=False):
        if USE_GEMINI or USE_LM_STUDIO:
            super().handle_input(input, reasoning_effort=reasoning_effort, temperature=temperature, stream=stream)
            return
        # OpenAI function-calling path
        messages = self.instruction_messages()
        messages = messages + self.messages_to_display.copy()
        messages = [m for m in messages if isinstance(m["content"], str)]
        messages.append({"role": "user", "content": f"```User: {input}```"})

        self.messages_to_display.append({"role": "user", "content": input})

        client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)

        # Call 1: model picks a tool if relevant, or answers directly if not
        r1 = client.responses.create(
            model=GPT_CHAT_MODEL,
            input=messages,
            tools=self.tools,
            tool_choice="auto",
        )
        fc = next((item for item in r1.output if item.type == "function_call"), None)

        if fc is None:
            # Model decided no tool was needed — use its response directly
            st.expander("Chat transcript", expanded=False).write(
                [{"role": m.get("role"), "content": m.get("content", "")} for m in messages if isinstance(m, dict)]
            )
            self.messages_to_display.append({"role": "assistant", "content": r1.output_text})
            return

        if fc.name == "get_player_summary":
            result = self._get_player_summary()
        else:
            result = self._search_knowledge(json.loads(fc.arguments)["query"])

        # Call 2: final answer, no more tools
        tool_inputs = list(messages) + list(r1.output) + [
            {"type": "function_call_output", "call_id": fc.call_id, "output": result}
        ]

        formatted = []
        for item in tool_inputs:
            if isinstance(item, dict):
                if item.get("type") == "function_call_output":
                    formatted.append({"tool_result": item["output"] or "(empty)", "call_id": item["call_id"]})
                else:
                    formatted.append({"role": item.get("role"), "content": item.get("content", "")})
            elif hasattr(item, "type"):
                if item.type == "function_call":
                    formatted.append({"tool_call": item.name, "arguments": json.loads(item.arguments)})
                # reasoning items are skipped
        st.expander("Chat transcript", expanded=False).write(formatted)
       
        if stream:
            if GPT_SUPPORTS_REASONING:
                reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                response_stream = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    reasoning={"effort": reasoning_effort},
                    stream=True,
                )
            elif GPT_SUPPORTS_TEMPERATURE:
                response_stream = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    temperature=temperature,
                    stream=True,
                )
            else:
                response_stream = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    stream=True,
                )

            def streamed_chunks():
                for event in response_stream:
                    if event.type == "response.output_text.delta":
                        yield event.delta

            answer = streamed_chunks()
        else:
            if GPT_SUPPORTS_REASONING:
                reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                response = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    reasoning={"effort": reasoning_effort},
                )
            elif GPT_SUPPORTS_TEMPERATURE:
                response = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    temperature=temperature,
                )
            else:
                response = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                )
            answer = response.output_text

        self.messages_to_display.append({"role": "assistant", "content": answer})

    def get_relevant_info(self, query):
        # Used by the Gemini/LM Studio path via super().handle_input

        # If there is no query then use the last message from the user
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a description of the player in terms of data: \n\n"
        description = PlayerDescription(self.player)
        ret_val += description.synthesize_text()

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nHere is a description of some relevant information for answering the question:  \n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += f"\n\nIf none of this information is relevent to the users's query then use the information below to remind the user about the chat functionality: \n"
        ret_val += "This chat can answer questions about a player's statistics and what they mean for how they play football."
        ret_val += "The user can select the player they are interested in using the menu to the left."

        return ret_val


class WVSChat(Chat):
    def __init__(
        self,
        chat_state_hash,
        country,
        countries,
        description_dict,
        thresholds_dict,
        state="empty",
    ):
        # TODO:
        self.embeddings = CountryEmbeddings()
        self.country = country
        self.countries = countries
        self.description_dict = description_dict
        self.thresholds_dict = thresholds_dict
        super().__init__(chat_state_hash, state=state)

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.country.name}?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x, stream=True)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        # TODO: Update first_messages
        first_messages = [
            {"role": "system", "content": "You are a researcher."},
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of a data analysis platform. "
                    f"The user has selected the country {self.country.name}, and the conversation will be about different core value measured in the World Value Survey study. "
                    # "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query to provide 2 sentence answers."
                    " Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
        return first_messages

    def get_relevant_info(self, query):

        # If there is no query then use the last message from the user
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a description of the country in terms of data: \n\n"
        description = CountryDescription(
            self.country, self.description_dict, self.thresholds_dict
        )
        ret_val += description.synthesize_text()

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nHere is a description of some relevant information for answering the question:  \n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += f"\n\nIf none of this information is relevant to the users's query then use the information below to remind the user about the chat functionality: \n"
        ret_val += "This chat can answer questions about a country's core values."
        ret_val += "The user can select the country they are interested in using the menu to the left."

        return ret_val


class PersonChat(Chat):
    def __init__(self, chat_state_hash, person, persons, state="empty"):
        self.embeddings = PersonEmbeddings()
        self.person = person
        self.persons = persons
        super().__init__(chat_state_hash, state=state)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        first_messages = [
            {"role": "system", "content": "You are a recruiter."},
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of personality test platform. "
                    f"The user has selected the person {self.person.name}, and the conversation will be about them. "
                    "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query  to provide 2 sentence answers."
                    " Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
        return first_messages

    def get_relevant_info(self, query):

        # If there is no query then use the last message from the user
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a description of the person in terms of data: \n\n"
        description = PersonDescription(self.person)
        ret_val += description.synthesize_text()

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nHere is a description of some relevant information for answering the question:  \n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += f"\n\nIf none of this information is relevent to the users's query then use the information below to remind the user about the chat functionality: \n"
        ret_val += "This chat can answer questions about person's statistics and what they mean about their personality."
        ret_val += "The user can select the persons they are interested in using the menu to the left."

        return ret_val

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.person.name}?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x, stream=True)


class TeamBuildUpChat(Chat):
    tools = [
        {
            "type": "function",
            "name": "get_team_style_summary",
            "description": "Returns a data-driven summary of the selected team's build-up style.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "compare_team_styles",
            "description": "Compares the selected team's build-up style with another team.",
            "parameters": {
                "type": "object",
                "properties": {
                    "other_team_name": {
                        "type": "string",
                        "description": "The name of the other team to compare with the selected team.",
                    }
                },
                "required": ["other_team_name"],
            },
        },
    ]

    STYLE_METRICS = [
        "prop_direct",
        "prop_goalkeeper_involved",
        "avg_successful_passes",
        "avg_phase_duration_seconds",
        "avg_players_involved",
        "build_ups_per_game",
    ]

    

    def __init__(self, chat_state_hash, team, teams, state="empty"):
        self.team = team
        self.teams = teams
        super().__init__(chat_state_hash, state=state)

    def get_input(self):
        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.team.name}'s build-up style?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )
            self.handle_input(x, stream=True)

    def instruction_messages(self):
        if USE_GEMINI or USE_LM_STUDIO:
            return [
                {"role": "system", "content": "You are a football build-up analyst."},
                {
                    "role": "user",
                    "content": (
                        f"The user has selected {self.team.name} and asks about team build-up style. "
                        "Use only the provided style information and keep responses to 2 concise sentences."
                    ),
                },
            ]

        return [
            {
                "role": "system",
                "content": (
                    "You are a football build-up style analyst. "
                    f"The selected team is {self.team.name}. "
                    "Choose the tool that best fits the user's query to respond. "
                    "- If the user asks for the selected team's style summary, use get_team_style_summary. "
                    "- If the user asks to compare style with another team, use compare_team_styles with other_team_name. "
                    "- If the user asks about a different scope than style, reply that this chat focuses on team build-up style. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them and keep responses to 2 concise sentences. "
                    "Do not add information beyond the function output."
                ),
            }
        ]

    def _format_team_metrics(self, team):
        """Format actual metric values for LLM context."""
        style_desc = TeamStyleDescription(team)
        lines = [f"\nActual metrics for {team.name}:"]
        
        for metric in self.STYLE_METRICS:
            actual_value = team.ser_metrics.get(metric)
            if actual_value is None:
                continue
            readable_metric = style_desc.write_out_team_metric(metric)
            # Format the value with appropriate precision
            if isinstance(actual_value, float):
                formatted_value = f"{actual_value:.2f}"
            else:
                formatted_value = str(actual_value)
            lines.append(f"- {readable_metric}: {formatted_value}")
        
        return "\n".join(lines)

    def _get_team_style_summary(self):
        description = TeamStyleDescription(self.team).synthesize_text()
        metrics = self._format_team_metrics(self.team)
        return description + metrics

    def _find_team_name(self, team_name):
        if not team_name:
            return None

        all_names = self.teams.df["team"].dropna().astype(str).unique().tolist()
        by_lower = {name.lower(): name for name in all_names}
        candidate = team_name.strip().lower()

        if candidate in by_lower:
            return by_lower[candidate]

        partial_matches = [name for name in all_names if candidate in name.lower()]
        if len(partial_matches) == 1:
            return partial_matches[0]
        return None

    def _compare_team_styles(self, other_team_name):
        matched_team_name = self._find_team_name(other_team_name)
        if matched_team_name is None:
            return (
                f"I could not uniquely identify '{other_team_name}'. "
                "Please provide the exact team name from the selector."
            )

        other_team = self.teams.to_data_point_by_team(matched_team_name)
        selected_team = self.team

        if matched_team_name.lower() == selected_team.name.lower():
            return (
                f"{selected_team.name} is the selected team, so there is no second team to compare. "
                "Please provide a different team name."
            )

        style_description = TeamStyleDescription(selected_team)
        selected_summary = TeamStyleDescription(selected_team).synthesize_text()
        other_summary = TeamStyleDescription(other_team).synthesize_text()
        selected_metrics = self._format_team_metrics(selected_team)
        other_metrics = self._format_team_metrics(other_team)

        metric_diffs = []
        for metric in self.STYLE_METRICS:
            selected_z = selected_team.ser_metrics.get(metric + "_Z", None)
            other_z = other_team.ser_metrics.get(metric + "_Z", None)
            if selected_z is None or other_z is None:
                continue

            metric_diffs.append(
                {
                    "metric": metric,
                    "delta": float(selected_z - other_z),
                }
            )

        metric_diffs = sorted(metric_diffs, key=lambda item: abs(item["delta"]), reverse=True)
        top_diffs = metric_diffs[:3]

        comparison_text = (
            f"Selected team style summary:\n{selected_summary}{selected_metrics}\n\n"
            f"Comparison team style summary ({other_team.name}):\n{other_summary}{other_metrics}"
        )

        if top_diffs:
            key_differences = []
            for item in top_diffs:
                readable_metric = style_description.write_out_team_metric(item["metric"])
                direction = selected_team.name if item["delta"] > 0 else other_team.name
                key_differences.append(
                    f"{direction} is higher in {readable_metric}"
                )

            comparison_text += "\n\nKey style differences: " + "; ".join(key_differences) + "."

        return comparison_text

    def _extract_other_team_from_query(self, query):
        q = query.lower()
        possible_names = self.teams.df["team"].dropna().astype(str).unique().tolist()
        for name in possible_names:
            lower_name = name.lower()
            if lower_name in q and lower_name != self.team.name.lower():
                return name
        return None

    def handle_input(self, input, reasoning_effort=None, temperature=1, stream=False):
        if USE_GEMINI or USE_LM_STUDIO:
            super().handle_input(
                input,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                stream=stream,
            )
            return

        messages = self.instruction_messages()
        messages = messages + self.messages_to_display.copy()
        messages = [m for m in messages if isinstance(m["content"], str)]
        messages.append({"role": "user", "content": f"```User: {input}```"})

        self.messages_to_display.append({"role": "user", "content": input})

        client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)

        r1 = client.responses.create(
            model=GPT_CHAT_MODEL,
            input=messages,
            tools=self.tools,
            tool_choice="auto",
        )
        fc = next((item for item in r1.output if item.type == "function_call"), None)

        if fc is None:
            st.expander("Chat transcript", expanded=False).write(
                [
                    {"role": m.get("role"), "content": m.get("content", "")}
                    for m in messages
                    if isinstance(m, dict)
                ]
            )
            self.messages_to_display.append({"role": "assistant", "content": r1.output_text})
            return

        if fc.name == "get_team_style_summary":
            result = self._get_team_style_summary()
        else:
            args = json.loads(fc.arguments)
            result = self._compare_team_styles(args.get("other_team_name", ""))

        tool_inputs = list(messages) + list(r1.output) + [
            {"type": "function_call_output", "call_id": fc.call_id, "output": result}
        ]

        formatted = []
        for item in tool_inputs:
            if isinstance(item, dict):
                if item.get("type") == "function_call_output":
                    formatted.append(
                        {"tool_result": item["output"] or "(empty)", "call_id": item["call_id"]}
                    )
                else:
                    formatted.append({"role": item.get("role"), "content": item.get("content", "")})
            elif hasattr(item, "type"):
                if item.type == "function_call":
                    formatted.append(
                        {"tool_call": item.name, "arguments": json.loads(item.arguments)}
                    )
        st.expander("Chat transcript", expanded=False).write(formatted)

        if stream:
            if GPT_SUPPORTS_REASONING:
                reasoning_effort = (
                    reasoning_effort
                    if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS
                    else GPT_AVAILABLE_REASONING_EFFORTS[0]
                )
                response_stream = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    reasoning={"effort": reasoning_effort},
                    stream=True,
                )
            elif GPT_SUPPORTS_TEMPERATURE:
                response_stream = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    temperature=temperature,
                    stream=True,
                )
            else:
                response_stream = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    stream=True,
                )

            def streamed_chunks():
                for event in response_stream:
                    if event.type == "response.output_text.delta":
                        yield event.delta

            answer = streamed_chunks()
        else:
            if GPT_SUPPORTS_REASONING:
                reasoning_effort = (
                    reasoning_effort
                    if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS
                    else GPT_AVAILABLE_REASONING_EFFORTS[0]
                )
                response = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    reasoning={"effort": reasoning_effort},
                )
            elif GPT_SUPPORTS_TEMPERATURE:
                response = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                    temperature=temperature,
                )
            else:
                response = client.responses.create(
                    model=GPT_CHAT_MODEL,
                    input=tool_inputs,
                    tool_choice="none",
                    tools=self.tools,
                )
            answer = response.output_text

        self.messages_to_display.append({"role": "assistant", "content": answer})

    def get_relevant_info(self, query):
        if query == "":
            query = self.messages_to_display[-1]["content"]

        lower_query = query.lower()
        compare_keywords = ["compare", "vs", "versus", "difference", "against"]
        wants_comparison = any(keyword in lower_query for keyword in compare_keywords)

        if wants_comparison:
            other_team_name = self._extract_other_team_from_query(query)
            if other_team_name:
                return self._compare_team_styles(other_team_name)
            return (
                "Please include another team name if you want a style comparison. "
                "I can then compare the selected team against that team."
            )

        return self._get_team_style_summary()


# class TeamChat(Chat):

#     # Define which metrics are higher-is-better for quality
#     QUALITY_METRICS_INFO = {
#         "buildup_that_ends_with_finish_pct": True,
#         "turnover_pct_buildup": False,  # lower is better
#         "opp_box_entries_within_7s_after_turnover": False,  # lower is better
#         "opp_shot_probability_within_7s_after_turnover": False,  # lower is better
#         "first_line_break_pct_buildup": True,
#     }

#     STYLE_METRICS = [
#         "buildup_to_create_pct",
#         "buildup_to_direct_pct",
#     ]

#     def __init__(self, chat_state_hash, team, teams, state="empty"):
#         self.embeddings = TeamEmbeddings()
#         self.team = team
#         self.teams = teams
#         super().__init__(chat_state_hash, state=state)

#     def get_input(self):
#         if x := st.chat_input(
#             placeholder=f"What else would you like to know about {self.team.name}'s build-up?"
#         ):
#             if len(x) > 500:
#                 st.error(f"Your message is too long ({len(x)} characters). Please keep it under 500 characters.")
#             self.handle_input(x, stream=True)

#     def instruction_messages(self):
#         return [
#             {"role": "system", "content": "You are a football build-up analyst."},
#             {
#                 "role": "user",
#                 "content": (
#                     f"The user is asking about {self.team.name}.\n\n"

#                     "RULES:\n"
#                     "- Answer in MAX 2 sentences.\n"
#                     "- Use plain text only.\n"
#                     "- DO NOT use bullet points or lists.\n"
#                     "- DO NOT mention metrics that are not provided.\n\n"

#                     "STYLE METRICS (style only, NOT quality):\n"
#                     "- buildup to create (%)\n"
#                     "- buildup to direct (%)\n\n"

#                     "QUALITY METRICS (performance):\n"
#                     "- buildup that ends with finish (%) (higher is better)\n"
#                     "- turnover buildup (%) (lower is better)\n"
#                     "- opp box entries after turnover (lower is better)\n"
#                     "- opp shot probability after turnover (lower is better)\n"
#                     "- first line break (%) (higher is better)\n\n"

#                     "IMPORTANT:\n"
#                     "- If the user asks 'best', 'strongest', or 'good at' → ONLY use QUALITY metrics.\n"
#                     "- NEVER mention style metrics in that case.\n"
#                     "- Do NOT explain all metrics — summarise the key strength only."
#                 ),
#             },
#         ]

#     # -----------------------
#     # 🔍 QUERY TYPES
#     # -----------------------
#     def is_comparison_query(self, query):
#         keywords = ["compare", "vs", "versus", "better", "worse", "than"]
#         return any(k in query.lower() for k in keywords)

#     def is_style_query(self, query):
#         keywords = ["style", "playstyle", "how do they play"]
#         return any(k in query.lower() for k in keywords)

#     def is_quality_query(self, query):
#         keywords = ["best", "strongest", "weakest", "worst", "strength", "good at"]
#         return any(k in query.lower() for k in keywords)

#     # -----------------------
#     # 🧾 TEAM DESCRIPTION (NO BULLETS)
#     # -----------------------
#     def get_team_description(self, team):
#         desc = f"{team.name}. "

# #        # Style metrics
# #        desc += "Style: "
# #        for m in self.STYLE_METRICS:
# #            value = team.ser_metrics.get(m + "_rank", None)
# #            if value is not None:
# #                desc += f"{clean_metric_name(m)} is ranked {round(value, 2)}. "

#         # Quality metrics
#         desc += "Quality: "
#         for m, higher_is_better in self.QUALITY_METRICS_INFO.items():
#             value = team.ser_metrics.get(m + "_rank", None)
#             if value is not None:
#                 desc += f"{clean_metric_name(m)} is ranked {round(value, 2)}. "

#         return desc

#     # -----------------------
#     # 🤝 MULTI TEAM
#     # -----------------------
#     def extract_teams(self, query):
#         names = self.teams.df["team"].unique()
#         q = query.lower()
#         return [n for n in names if n.lower() in q]

#     def get_multiple_teams_info(self, query):
#         df = self.teams.df
#         teams = self.extract_teams(query)

#         if self.team.name not in teams:
#             teams.append(self.team.name)

#         if len(teams) == 1:
#             similar = (
#                 df.copy()
#                 .assign(sim=lambda x: (
#                     x[self.team.relevant_metrics] - self.team.ser_metrics
#                 ).abs().sum(axis=1))
#                 .sort_values("sim")
#             )
#             teams += similar["team"].iloc[1:3].tolist()

#         info = ""
#         for t in teams:
#             obj = self.teams.to_data_point_by_team(t)
#             info += f"{t}. "
#             info += self.get_team_description(obj)

#         return info

#     # -----------------------
#     # 🎯 MAIN CONTEXT
#     # -----------------------
#     def get_relevant_info(self, query):

#         # ---------------- COMPARISON ----------------
#         if self.is_comparison_query(query):
#             text = "Comparison data. "
#             text += self.get_multiple_teams_info(query)
#             return text

#         # ---------------- QUALITY ----------------
#         elif self.is_quality_query(query):
#             text = f"{self.team.name} quality metrics. "

#             for m, higher_is_better in self.QUALITY_METRICS_INFO.items():
#                 rank = self.team.ser_metrics.get(m + "_rank", None)
#                 if rank is not None:
#                     # invert rank for metrics where lower is better
#                     effective_rank = rank if higher_is_better else 1 - rank

#                     # Convert to natural language
#                     if effective_rank >= 0.66:
#                         perf = "strong"
#                     elif effective_rank <= 0.33:
#                         perf = "weak"
#                     else:
#                         perf = "average"

#                     text += f"{clean_metric_name(m)} is {perf}. "

# #        # ---------------- STYLE ----------------
# #        elif self.is_style_query(query):
# #            text = f"{self.team.name} style metrics. "
# #            for m in self.STYLE_METRICS:
# #                value = self.team.ser_metrics.get(m + "_rank", None)
# #                if value is not None:
# #                    text += f"{clean_metric_name(m)} is ranked {round(value, 2)}. "
# #            return text

#         # ---------------- DEFAULT ----------------
#         else:
#             text = self.get_team_description(self.team)
#             results = self.embeddings.search(query, top_n=5)
#             insights = " ".join(results["assistant"].to_list())
#             text += " Insights: " + insights
#             return text


import json
import streamlit as st
from difflib import get_close_matches
from openai import OpenAI


class TeamChat(Chat):

    # ---------------- TOOLS ----------------
    tools = [
        {
            "type": "function",
            "name": "summarise_style",
            "description": "Summarises the selected team's build-up style.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "summarise_performance",
            "description": "Summarises the selected team's build-up performance.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "compare_style",
            "description": "Compare style with another team.",
            "parameters": {
                "type": "object",
                "properties": {"other_team_name": {"type": "string"}},
                "required": ["other_team_name"],
            },
        },
        {
            "type": "function",
            "name": "compare_performance",
            "description": "Compare performance with another team.",
            "parameters": {
                "type": "object",
                "properties": {"other_team_name": {"type": "string"}},
                "required": ["other_team_name"],
            },
        },
    ]

    # ---------------- METRICS ----------------
    STYLE_METRICS = [
        "prop_gk_involved",
        "avg_passes",
        "avg_duration",
        "avg_players_involved",
        "build_ups_per_game",
        "prop_channel_center",
        "prop_channel_half_space_left",
        "prop_channel_wide_left",
        "prop_channel_half_space_right",
        "prop_channel_wide_right",
    ]

    PERFORMANCE_METRICS = {
        "buildup_that_ends_with_finish_pct": True,
        "turnover_pct_buildup": False,
        "opp_box_entries_within_7s_after_turnover": False,
        "opp_shot_probability_within_7s_after_turnover": False,
        "first_line_break_pct_buildup": True,
    }

    # ---------------- INIT ----------------
    def __init__(self, chat_state_hash, team, teams, state="empty"):
        self.teams = teams

        # 🔥 Persist team across reruns
        if "selected_team_name" not in st.session_state:
            st.session_state["selected_team_name"] = None

        self.selected_team = None
        if st.session_state["selected_team_name"]:
            self.selected_team = self.teams.to_data_point_by_team(
                st.session_state["selected_team_name"]
            )

        super().__init__(chat_state_hash, state=state)

    # ---------------- INPUT ----------------
    def get_input(self):
        if x := st.chat_input("Ask about a team..."):
            if len(x) > 500:
                st.error("Message too long.")
                return
            self.handle_input(x, stream=True)

    # ---------------- TEAM MATCHING ----------------
    def _match_team(self, query):
        names = self.teams.df["team"].dropna().tolist()
        q = query.lower()

        # exact
        for n in names:
            if q == n.lower():
                return n

        # partial
        for n in names:
            if q in n.lower():
                return n

        # fuzzy
        matches = get_close_matches(query, names, n=1, cutoff=0.6)
        if matches:
            return matches[0]

        return None

    # ---------------- STYLE ----------------
    def _summarise_style(self):
        t = self.selected_team

        passes = t.ser_metrics.get("avg_passes", 0)
        duration = t.ser_metrics.get("avg_duration", 0)
        central = t.ser_metrics.get("prop_channel_center", 0)

        text = f"{t.name} tend to build up in a "

        if passes > 0.6:
            text += "patient and possession-oriented way, "
        else:
            text += "more direct manner, "

        text += f"with sequences lasting around {round(duration, 2)} seconds. "

        if central > 0.5:
            text += "They prefer progressing through central areas rather than using width."
        else:
            text += "They make greater use of wide areas to advance the ball."

        return text

    # ---------------- PERFORMANCE ----------------
    def _summarise_performance(self):
        t = self.selected_team

        finish = t.ser_metrics.get("buildup_that_ends_with_finish_pct", 0)
        turnover = 1 - t.ser_metrics.get("turnover_pct_buildup", 0)

        text = f"{t.name} show "

        if finish > 0.66:
            text += "strong attacking output from build-up, "
        elif finish < 0.33:
            text += "limited attacking output from build-up, "
        else:
            text += "moderate attacking output, "

        if turnover > 0.66:
            text += "while maintaining good security in possession."
        else:
            text += "but remain vulnerable to turnovers under pressure."

        return text

    # ---------------- COMPARISONS ----------------
    def _compare_style(self, other_name):
        other_name = self._match_team(other_name)
        if not other_name:
            return "I couldn’t recognise the comparison team."

        other = self.teams.to_data_point_by_team(other_name)
        t = self.selected_team

        diff = t.ser_metrics["avg_passes"] - other.ser_metrics["avg_passes"]

        if diff > 0:
            return f"{t.name} build up more patiently than {other.name}, using longer passing sequences."
        else:
            return f"{other.name} build up more patiently than {t.name}, with more passes per sequence."

    def _compare_performance(self, other_name):
        other_name = self._match_team(other_name)
        if not other_name:
            return "I couldn’t recognise the comparison team."

        other = self.teams.to_data_point_by_team(other_name)
        t = self.selected_team

        diff = (
            t.ser_metrics["buildup_that_ends_with_finish_pct"]
            - other.ser_metrics["buildup_that_ends_with_finish_pct"]
        )

        if diff > 0:
            return f"{t.name} convert build-up into attacking situations more effectively than {other.name}."
        else:
            return f"{other.name} are more effective at turning build-up into attacking situations than {t.name}."

    # ---------------- HANDLE INPUT ----------------
    def handle_input(self, input, stream=False, **kwargs):

        # Save user message
        self.messages_to_display.append({"role": "user", "content": input})

        # ---------------- STEP 1: TEAM SELECTION ----------------
        if self.selected_team is None:
            match = self._match_team(input)

            if match:
                st.session_state["selected_team_name"] = match
                self.selected_team = self.teams.to_data_point_by_team(match)

                response = f"Nice choice — let’s take a closer look at {match}. What would you like to explore?"
            else:
                response = "I couldn’t find that team. Try something like Arsenal or Manchester United."

            self.messages_to_display.append({"role": "assistant", "content": response})
            return

        # ---------------- STEP 2: SIMPLE ROUTING (NO LLM CONFUSION) ----------------
        query = input.lower()

        if "style" in query:
            result = self._summarise_style()

        elif "performance" in query or "good" in query or "strong" in query:
            result = self._summarise_performance()

        elif "compare" in query:
            words = query.split()
            other_team = words[-1]  # simple heuristic
            result = self._compare_performance(other_team)

        else:
            result = self._summarise_performance()

        # ---------------- FINAL RESPONSE (LLM FOR POLISH) ----------------
        client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)

        final = client.responses.create(
            model=GPT_CHAT_MODEL,
            input=[
                {
                    "role": "system",
                    "content": "Rewrite the following into 2 natural, fluid sentences as a football analyst.",
                },
                {"role": "user", "content": result},
            ],
        )

        self.messages_to_display.append(
            {"role": "assistant", "content": final.output_text}
        )