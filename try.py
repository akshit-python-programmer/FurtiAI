import logging

from furti_ai import build_agent

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s",
)

agent = build_agent()
agent.run("Click on the Explorer icon in the left sidebar of the Visual Studio Code window")
