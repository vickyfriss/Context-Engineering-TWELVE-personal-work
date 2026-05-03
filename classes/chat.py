import streamlit as st
from openai import OpenAI
from itertools import groupby
from types import GeneratorType
import pandas as pd
import json
from classes.description import TeamDescription, TeamStyleDescription
from classes.embeddings import TeamEmbeddings

from settings import USE_GEMINI
import pandas as pd
import json
from difflib import get_close_matches  # <-- LEGG TIL HER

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


class TeamChat(Chat):
    tools = [
        {
            "type": "function",
            "name": "summarise_style",
            "description": "Summarise one team's build-up style.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_name": {"type": "string"}
                },
                "required": ["team_name"],
            },
        },
        {
            "type": "function",
            "name": "summarise_performance",
            "description": "Summarise one team's build-up performance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_name": {"type": "string"}
                },
                "required": ["team_name"],
            },
        },
        {
            "type": "function",
            "name": "compare_style",
            "description": "Compare build-up style between two or more teams.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_names": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["team_names"],
            },
        },
        {
            "type": "function",
            "name": "compare_performance",
            "description": "Compare build-up performance between two or more teams.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_names": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["team_names"],
            },
        },
    ]

    STYLE_METRICS = [
        "avg_passes",
        "avg_duration",
        "build_ups_per_game",
        "prop_channel_center",
        "prop_channel_half_space_left",
        "prop_channel_wide_left",
        "prop_channel_half_space_right",
        "prop_channel_wide_right",
    ]

    PERFORMANCE_METRICS = [
        "first_line_break_pct_buildup",
        "progression_to_midfield_pct",
        "buildup_that_ends_with_finish_pct",
        "turnover_pct_buildup",
        "opp_box_entries_within_7s_after_turnover",
        "opp_shot_probability_within_7s_after_turnover",
    ]

    def __init__(self, chat_state_hash, team, teams, state="empty"):
        self.teams = teams
        self.selected_team = None
        super().__init__(chat_state_hash, state=state)

    def get_input(self):
        if x := st.chat_input("Ask about a team..."):
            self.handle_input(x)

    def instruction_messages(self):
        return [
            {
                "role": "system",
                "content": (
                    "You are a football build-up analyst. "
                    "Choose exactly one function based on the user's question. "
                    "Use summarise_style for one-team style questions. "
                    "Use summarise_performance for one-team performance, quality, strengths or weaknesses questions. "
                    "Use compare_style for style comparisons. "
                    "Use compare_performance for performance or quality comparisons. "
                    "Extract team names directly from the user's message."
                ),
            }
        ]

    def _extract_teams(self, text):
        names = self.teams.df["team"].dropna().tolist()
        found = []

        for name in names:
            if name.lower() in text.lower():
                found.append(name)

        for token in text.split():
            match = get_close_matches(token, names, n=1, cutoff=0.7)
            if match and match[0] not in found:
                found.append(match[0])

        return list(dict.fromkeys(found))

    def _resolve_team_names(self, team_names):
        names = self.teams.df["team"].dropna().tolist()
        resolved = []

        for team_name in team_names:
            if not team_name:
                continue

            exact = [name for name in names if name.lower() == team_name.lower()]
            if exact:
                resolved.append(exact[0])
                continue

            close = get_close_matches(team_name, names, n=1, cutoff=0.7)
            if close:
                resolved.append(close[0])

        return list(dict.fromkeys(resolved))

    def _safe_df(self, metrics):
        required_columns = []

        for metric in metrics:
            required_columns.extend([metric, metric + "_Z", metric + "_Ranks"])

        required_columns = [
            column for column in required_columns
            if column in self.teams.df.columns
        ]

        return self.teams.df.dropna(subset=required_columns).copy()

    def _build_plot(self, teams_list, metrics, title, subtitle, display_names=None, labels=None):
        df_clean = self._safe_df(metrics)

        dropped = len(self.teams.df) - len(df_clean)
        if dropped > 0:
            st.caption(f"ℹ️ {dropped} team(s) excluded from plot due to missing data in selected metrics.")

        plot = DistributionPlot(
            columns=metrics[::-1],
            labels=labels or ["Worse", "Average", "Better"],
            plot_type="default",
            display_names=display_names,
        )

        plot.add_title(title=title, subtitle=subtitle)

        original_df = self.teams.df
        try:
            self.teams.df = df_clean

            plot.add_players(self.teams, metrics=metrics)

            for i, team in enumerate(teams_list):
                if i == 0:
                    plot.add_player(team, len(df_clean), metrics=metrics)
                else:
                    original_annotation = plot.annotation_text
                    plot.annotation_text = ""
                    plot.add_player(team, len(df_clean), metrics=metrics)
                    plot.annotation_text = original_annotation
        finally:
            self.teams.df = original_df

        return plot

    def _describe_difference_size(self, delta):
        abs_delta = abs(delta)

        if abs_delta >= 1.5:
            return "much higher"
        if abs_delta >= 0.75:
            return "clearly higher"
        return "slightly higher"

    def _compare_from_metrics(self, teams_list, metrics, description_obj):
        reference_team = teams_list[0]
        sections = []

        for other_team in teams_list[1:]:
            diffs = []

            for metric in metrics:
                ref_z = reference_team.ser_metrics.get(metric + "_Z")
                other_z = other_team.ser_metrics.get(metric + "_Z")

                if ref_z is None or other_z is None:
                    continue
                if pd.isna(ref_z) or pd.isna(other_z):
                    continue

                diffs.append(
                    {
                        "metric": metric,
                        "delta": float(ref_z - other_z),
                    }
                )

            diffs = sorted(diffs, key=lambda x: abs(x["delta"]), reverse=True)[:3]

            if not diffs:
                sections.append(f"No valid comparison for {other_team.name}.")
                continue

            parts = []

            for diff in diffs:
                metric_name = description_obj.write_out_team_metric(diff["metric"])
                delta = diff["delta"]

                if delta > 0:
                    higher = reference_team.name
                    lower = other_team.name
                else:
                    higher = other_team.name
                    lower = reference_team.name

                size = self._describe_difference_size(delta)
                parts.append(f"{higher} are {size} than {lower} in {metric_name}")

            sections.append(
                f"Compared with {other_team.name}: "
                + "; ".join(parts)
                + "."
            )

        intro = (
            f"{reference_team.name} are used as the reference team. "
            "The comparison is based on the largest z-score differences."
        )

        return intro + "\n\n" + "\n\n".join(sections)

    def _summarise_style(self, teams_list):
        display_names = {
            "avg_passes": "Avg Passes",
            "avg_duration": "Avg Duration (s)",
            "build_ups_per_game": "Build-Ups / Game",
            "prop_channel_center": "Central (%)",
            "prop_channel_half_space_left": "Left Half-Space (%)",
            "prop_channel_wide_left": "Left Wide (%)",
            "prop_channel_half_space_right": "Right Half-Space (%)",
            "prop_channel_wide_right": "Right Wide (%)",
        }

        plot = self._build_plot(
            teams_list=teams_list,
            metrics=self.STYLE_METRICS,
            title=(
                "Build-Up Style Comparison"
                if len(teams_list) > 1
                else f"{teams_list[0].name} – Build-Up Style"
            ),
            subtitle="How teams build up play (z-scores)",
            display_names=display_names,
            labels=["Less", "Average", "More"],
        )

        if len(teams_list) == 1:
            text = TeamStyleDescription(teams_list[0]).synthesize_text()
        else:
            text = self._compare_from_metrics(
                teams_list=teams_list,
                metrics=self.STYLE_METRICS,
                description_obj=TeamStyleDescription(teams_list[0]),
            )

        return plot, text

    def _summarise_performance(self, teams_list):
        display_names = {
            "progression_to_midfield_pct": "Progression to Midfield (%)",
            "buildup_that_ends_with_finish_pct": "Buildup Ending in Finish (%)",
            "turnover_pct_buildup": "Turnover (%)",
            "opp_box_entries_within_7s_after_turnover": "Opp Box Entries After Turnover",
            "opp_shot_probability_within_7s_after_turnover": "Opp Shot Probability After Turnover",
            "first_line_break_pct_buildup": "First Line Break (%)",
        }

        plot = self._build_plot(
            teams_list=teams_list,
            metrics=self.PERFORMANCE_METRICS,
            title=(
                "Build-Up Performance Comparison"
                if len(teams_list) > 1
                else f"{teams_list[0].name} – Build-Up Performance"
            ),
            subtitle="Effectiveness and outcomes of build-up (z-scores)",
            display_names=display_names,
            labels=["Worse", "Average", "Better"],
        )

        if len(teams_list) == 1:
            text = TeamDescription(teams_list[0]).synthesize_text()
        else:
            text = self._compare_from_metrics(
                teams_list=teams_list,
                metrics=self.PERFORMANCE_METRICS,
                description_obj=TeamStyleDescription(teams_list[0]),
            )

        return plot, text

    def _run_tool(self, function_name, arguments):
        if function_name in ["summarise_style", "summarise_performance"]:
            team_names = self._resolve_team_names([arguments.get("team_name")])
        else:
            team_names = self._resolve_team_names(arguments.get("team_names", []))

        if not team_names:
            return None, "I could not identify the team. Please write the team name again."

        teams_list = [
            self.teams.to_data_point_by_team(team_name)
            for team_name in team_names
        ]

        if function_name == "summarise_style":
            return self._summarise_style([teams_list[0]])

        if function_name == "summarise_performance":
            return self._summarise_performance([teams_list[0]])

        if len(teams_list) < 2:
            return None, "Please include at least two teams for a comparison."

        if function_name == "compare_style":
            return self._summarise_style(teams_list)

        if function_name == "compare_performance":
            return self._summarise_performance(teams_list)

        return None, "I could not route the question to a valid analysis function."

    def handle_input(self, input):
        self.messages_to_display.append({"role": "user", "content": input})

        client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)

        messages = self.instruction_messages()
        messages.append({"role": "user", "content": input})

        response = client.responses.create(
            model=GPT_CHAT_MODEL,
            input=messages,
            tools=self.tools,
            tool_choice="auto",
        )

        function_call = next(
            (item for item in response.output if item.type == "function_call"),
            None,
        )

        if function_call is None:
            teams_found = self._extract_teams(input)

            if not teams_found:
                self.messages_to_display.append(
                    {
                        "role": "assistant",
                        "content": "Which team would you like to analyse?",
                    }
                )
                return

            fallback_name = "compare_performance" if len(teams_found) > 1 else "summarise_performance"
            fallback_args = (
                {"team_names": teams_found}
                if len(teams_found) > 1
                else {"team_name": teams_found[0]}
            )

            plot, tool_text = self._run_tool(fallback_name, fallback_args)
        else:
            tool_args = json.loads(function_call.arguments)
            plot, tool_text = self._run_tool(function_call.name, tool_args)

        final = client.responses.create(
            model=GPT_CHAT_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the provided analysis into 2-3 natural, fluent football analysis sentences. "
                        "Keep the meaning and the metric-based differences. "
                        "Do not add information that is not provided."
                    ),
                },
                {"role": "user", "content": tool_text},
            ],
        )

        if plot is not None:
            self.messages_to_display.append({"role": "assistant", "content": plot})

        self.messages_to_display.append(
            {"role": "assistant", "content": final.output_text}
        )