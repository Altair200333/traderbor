from src.agents.marginal_agent import MarginalAgent
from src.api import *
from src.const import *


class SupervisorMarginTrader:
    def __init__(self) -> None:
        pass

    def decide(
        self,
        **kvargs,
    ):
        agent = MarginalAgent()
        _, messages = agent.decide(**kvargs)

        messages.append(
            ai_client.make_msg(
                f"""Evaluate this decision, does it make sence, is it going to be profitable, are calculations correct?
                If somethins is not right, update the original data with corrected values
                Respond with the same json, but add several fields in the begining:
                {{
                    "strategy_evaluation": does it make sense, is it improving current performance?
                    "calculations_evaluation": check if are all the calculations here correct. Do they add up?
                    "changes_description": if the strategy requires corrections, describe them here,
                    ...rest of the fields (update them if required),
                }}
                """,
            ),
        )

        response = ai_client.create(messages, format=JSON_MODE)

        messages.append(
            ai_client.make_msg(text=response, role=ROLE_ASSISTANT),
        )

        print(messages)

        return response, messages
