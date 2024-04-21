from src.agents.trading_agent import TradingAgent
from src.api import *
from src.const import *


class SupervisorTrader:
    def __init__(self) -> None:
        pass

    def decide(
        self,
        **kvargs,
    ):
        agent = TradingAgent()
        _, messages = agent.decide(**kvargs)

        messages.append(
            ai_client.make_msg(
                f"""Evaluate your prediction from different perspectives, does it make sence, is it profitable, are calculations correct?
                If somethins is not right, update the original data with corrected values
                Respond with the same json, but add several fields:
                {{
                    "strategy_evaluation": does it make sense, is it improving current performance?
                    "calculations_evaluation": check if are all the calculations here correct. Do they add up?
                    "changes_description": if the strategy requires corrections, describe them here,
                    ...same fields (update them if required),
                }}
                """,
            ),
        )

        response = ai_client.create(messages, format=JSON_MODE)

        messages.append(
            ai_client.make_msg(text=response, role=ROLE_ASSISTANT),
        )

        return response, messages
