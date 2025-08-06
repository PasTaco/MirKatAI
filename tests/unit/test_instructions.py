import pytest
from langchain_core.messages import (
    AIMessage,
    SystemMessage
)
from app.mirkat.instructions import Instructions


def check_valid_characters(text: str) -> bool:
    """
    Checks if a string contains only printable characters.
    Handles letters, numbers, symbols, but flags control/non-printable chars.
    """
    # Define the set of control characters you are willing to allow.
    allowed_control_chars = {'\n', '\t', '\r'} # Newline, Tab, Carriage Return

    for char in text:
        # If the character is printable, it's fine. Continue to the next one.
        if char.isprintable():
            continue
        
        # If the character is NOT printable, check if it's in our allowed set.
        # If it's not in our allowed set, we found an unwanted character.
        if char not in allowed_control_chars:
            return False # Found a bad, non-printable character!
            
    # If the loop completes, all characters were either printable or in the allowed set.
    return True


def test_get_instructions_router():
    """
    Test to ensure that the router instructions are loaded and valid from the YAML file.
    """

    MIRNA_ASSISTANT_SYSTEM_MESSAGE, WELCOME_MSG = Instructions.router.get_instruction()
    assert MIRNA_ASSISTANT_SYSTEM_MESSAGE is not None, "MIRNA_ASSISTANT_SYSTEM_MESSAGE should not be None"
    assert WELCOME_MSG is not None, "WELCOME_MSG should not be None"
    assert isinstance(MIRNA_ASSISTANT_SYSTEM_MESSAGE, SystemMessage), "MIRNA_ASSISTANT_SYSTEM_MESSAGE should be a SystemMessage"
    assert isinstance(WELCOME_MSG, AIMessage), "WELCOME_MSG should be an AIMessage"
    assert check_valid_characters(MIRNA_ASSISTANT_SYSTEM_MESSAGE.content), "MIRNA_ASSISTANT_SYSTEM_MESSAGE contains invalid characters"
    assert check_valid_characters(WELCOME_MSG.content), "WELCOME_MSG contains invalid characters"


def test_get_instructions_sql():
    """
    Test to ensure that the SQL instructions are loaded and valid from the YAML file.
    """
    SQL_INSTRUCTIONS = Instructions.sql.get_instruction()
    assert SQL_INSTRUCTIONS is not None, "SQL_INSTRUCTIONS should not be None"
    assert isinstance(SQL_INSTRUCTIONS, str), "SQL_INSTRUCTIONS should be a string"
    assert check_valid_characters(SQL_INSTRUCTIONS), "SQL_INSTRUCTIONS contains invalid characters"


def test_get_instructions_plot():
    """
    Test to ensure that the plot instructions are loaded and valid from the YAML file.
    """
    PLOT_INSTRUCTIONS = Instructions.plot.get_instruction()
    assert PLOT_INSTRUCTIONS is not None, "PLOT_INSTRUCTIONS should not be None"
    assert isinstance(PLOT_INSTRUCTIONS, str), "PLOT_INSTRUCTIONS should be a string"
    assert check_valid_characters(PLOT_INSTRUCTIONS), "PLOT_INSTRUCTIONS contains invalid characters"

def test_get_instructions_literature():
    """
    Test to ensure that the literature instructions are loaded and valid from the YAML file.
    """
    LITERATURE_INSTRUCTIONS = Instructions.literature.get_instruction()
    assert LITERATURE_INSTRUCTIONS is not None, "LITERATURE_INSTRUCTIONS should not be None"
    assert isinstance(LITERATURE_INSTRUCTIONS, str), "LITERATURE_INSTRUCTIONS should be a string"
    assert check_valid_characters(LITERATURE_INSTRUCTIONS), "LITERATURE_INSTRUCTIONS contains invalid characters"

