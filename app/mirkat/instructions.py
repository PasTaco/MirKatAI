import yaml
from langchain_core.messages import (
    AIMessage,
    SystemMessage
)
import os
from enum import Enum


absolute_path = os.path.abspath(__file__)
directory = os.path.dirname(absolute_path)
instruction_file = f"{directory}/instructions.yaml"
with open(instruction_file, "r") as file:
    instructions = yaml.safe_load(file)


class Instructions(Enum):
    sql='SQL'
    router='ROUTING'
    welcome='WELCOME_MSG'
    plot='PLOT'
    format_answer='FORMAT_ANSWER'
    literature='LITERATURE'

   # We don't need to override __init__ or modify self.value
    def get_instruction(self):
        if self == Instructions.router:
            return SystemMessage(content=instructions[f"{self.router.value}_INSTRUCTIONS"]), AIMessage(instructions[self.welcome.value])
        elif self == Instructions.sql:
            return instructions[f"{self.sql.value}_INSTRUCTIONS"]
        elif self == Instructions.plot:
            return instructions[f"{self.plot.value}_INSTRUCTIONS"]
        else:
            return instructions[f"{self.value}_INSTRUCTIONS"]
        #else:
        #    raise ValueError(f"Unknown instruction type: {self.value}")