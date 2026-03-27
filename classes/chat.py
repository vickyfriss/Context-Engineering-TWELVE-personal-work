import streamlit as st
from openai import OpenAI
from itertools import groupby
from types import GeneratorType
import pandas as pd
import json
from classes.description import TeamDescription
from classes.embeddings import TeamEmbeddings

from settings import USE_GEMINI
if USE_GEMINI:
    from settings import USE_GEMINI, GEMINI_API_KEY, GEMINI_CHAT_MODEL
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
    def __init__(self, chat_state_hash, player, players, state="empty"):
        self.embeddings = PlayerEmbeddings()
        self.player = player
        self.players = players
        super().__init__(chat_state_hash, state=state)

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

    def get_relevant_info(self, query):

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

    # Define which metrics are higher-is-better for quality
    QUALITY_METRICS_INFO = {
        "buildup_that_ends_with_finish_pct": True,
        "turnover_pct_buildup": False,  # lower is better
        "opp_box_entries_within_7s_after_turnover": False,  # lower is better
        "opp_shot_probability_within_7s_after_turnover": False,  # lower is better
        "first_line_break_pct_buildup": True,
    }

    STYLE_METRICS = [
        "buildup_to_create_pct",
        "buildup_to_direct_pct",
    ]

    def __init__(self, chat_state_hash, team, teams, state="empty"):
        self.embeddings = TeamEmbeddings()
        self.team = team
        self.teams = teams
        super().__init__(chat_state_hash, state=state)

    def get_input(self):
        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.team.name}'s build-up?"
        ):
            if len(x) > 500:
                st.error(f"Your message is too long ({len(x)} characters). Please keep it under 500 characters.")
            self.handle_input(x, stream=True)

    def instruction_messages(self):
        return [
            {"role": "system", "content": "You are a football build-up analyst."},
            {
                "role": "user",
                "content": (
                    f"The user is asking about {self.team.name}.\n\n"

                    "RULES:\n"
                    "- Answer in MAX 2 sentences.\n"
                    "- Use plain text only.\n"
                    "- DO NOT use bullet points or lists.\n"
                    "- DO NOT mention metrics that are not provided.\n\n"

                    "STYLE METRICS (style only, NOT quality):\n"
                    "- buildup to create (%)\n"
                    "- buildup to direct (%)\n\n"

                    "QUALITY METRICS (performance):\n"
                    "- buildup that ends with finish (%) (higher is better)\n"
                    "- turnover buildup (%) (lower is better)\n"
                    "- opp box entries after turnover (lower is better)\n"
                    "- opp shot probability after turnover (lower is better)\n"
                    "- first line break (%) (higher is better)\n\n"

                    "IMPORTANT:\n"
                    "- If the user asks 'best', 'strongest', or 'good at' → ONLY use QUALITY metrics.\n"
                    "- NEVER mention style metrics in that case.\n"
                    "- Do NOT explain all metrics — summarise the key strength only."
                ),
            },
        ]

    # -----------------------
    # 🔍 QUERY TYPES
    # -----------------------
    def is_comparison_query(self, query):
        keywords = ["compare", "vs", "versus", "better", "worse", "than"]
        return any(k in query.lower() for k in keywords)

    def is_style_query(self, query):
        keywords = ["style", "playstyle", "how do they play"]
        return any(k in query.lower() for k in keywords)

    def is_quality_query(self, query):
        keywords = ["best", "strongest", "weakest", "worst", "strength", "good at"]
        return any(k in query.lower() for k in keywords)

    # -----------------------
    # 🧾 TEAM DESCRIPTION (NO BULLETS)
    # -----------------------
    def get_team_description(self, team):
        desc = f"{team.name}. "

#        # Style metrics
#        desc += "Style: "
#        for m in self.STYLE_METRICS:
#            value = team.ser_metrics.get(m + "_rank", None)
#            if value is not None:
#                desc += f"{clean_metric_name(m)} is ranked {round(value, 2)}. "

        # Quality metrics
        desc += "Quality: "
        for m, higher_is_better in self.QUALITY_METRICS_INFO.items():
            value = team.ser_metrics.get(m + "_rank", None)
            if value is not None:
                desc += f"{clean_metric_name(m)} is ranked {round(value, 2)}. "

        return desc

    # -----------------------
    # 🤝 MULTI TEAM
    # -----------------------
    def extract_teams(self, query):
        names = self.teams.df["team"].unique()
        q = query.lower()
        return [n for n in names if n.lower() in q]

    def get_multiple_teams_info(self, query):
        df = self.teams.df
        teams = self.extract_teams(query)

        if self.team.name not in teams:
            teams.append(self.team.name)

        if len(teams) == 1:
            similar = (
                df.copy()
                .assign(sim=lambda x: (
                    x[self.team.relevant_metrics] - self.team.ser_metrics
                ).abs().sum(axis=1))
                .sort_values("sim")
            )
            teams += similar["team"].iloc[1:3].tolist()

        info = ""
        for t in teams:
            obj = self.teams.to_data_point_by_team(t)
            info += f"{t}. "
            info += self.get_team_description(obj)

        return info

    # -----------------------
    # 🎯 MAIN CONTEXT
    # -----------------------
    def get_relevant_info(self, query):

        # ---------------- COMPARISON ----------------
        if self.is_comparison_query(query):
            text = "Comparison data. "
            text += self.get_multiple_teams_info(query)
            return text

        # ---------------- QUALITY ----------------
        elif self.is_quality_query(query):
            text = f"{self.team.name} quality metrics. "

            for m, higher_is_better in self.QUALITY_METRICS_INFO.items():
                rank = self.team.ser_metrics.get(m + "_rank", None)
                if rank is not None:
                    # invert rank for metrics where lower is better
                    effective_rank = rank if higher_is_better else 1 - rank

                    # Convert to natural language
                    if effective_rank >= 0.66:
                        perf = "strong"
                    elif effective_rank <= 0.33:
                        perf = "weak"
                    else:
                        perf = "average"

                    text += f"{clean_metric_name(m)} is {perf}. "

#        # ---------------- STYLE ----------------
#        elif self.is_style_query(query):
#            text = f"{self.team.name} style metrics. "
#            for m in self.STYLE_METRICS:
#                value = self.team.ser_metrics.get(m + "_rank", None)
#                if value is not None:
#                    text += f"{clean_metric_name(m)} is ranked {round(value, 2)}. "
#            return text

        # ---------------- DEFAULT ----------------
        else:
            text = self.get_team_description(self.team)
            results = self.embeddings.search(query, top_n=5)
            insights = " ".join(results["assistant"].to_list())
            text += " Insights: " + insights
            return text