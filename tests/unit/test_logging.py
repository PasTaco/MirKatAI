from json import encoder
import pytest
import sys
import logging
import io



class RaisingHandler(logging.StreamHandler):
    def handleError(self, record):
        raise


def test_logging_beta_raises_with_latin1():
    """
    Logging a message containing a Greek beta to a Latin-1 encoded stream
    with errors='strict' should raise UnicodeEncodeError.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    logger = logging.getLogger('test_latin1_final')
    byte_stream = io.BytesIO()
    latin1_text_stream = io.TextIOWrapper(byte_stream, encoding='latin-1', errors='strict')
    handler = RaisingHandler(latin1_text_stream)
    logger.addHandler(handler)
    beta_msg = "Here is a beta: β"
    try:
        logger.info(beta_msg)
        assert False, "Expected UnicodeEncodeError when logging β to Latin-1 stream"
    except UnicodeEncodeError:
        assert True
    logger.removeHandler(handler)


def test_logging_beta_succeeds_with_utf8():
    """
    Logging a message containing a Greek beta to a UTF-8 encoded stream
    should succeed without raising an error.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8',  # Ensure log messages are UTF-8 encoded
        force=True
    )
    logger = logging.getLogger('test_utf8_final')
    byte_stream = io.BytesIO()
    utf8_text_stream = io.TextIOWrapper(byte_stream, encoding='utf-8', errors='strict')
    handler = RaisingHandler(utf8_text_stream)
    logger.addHandler(handler)
    beta_msg = "Here is a beta: β"
    try:
        logger.info(beta_msg)
        assert True  # Should not raise an error
    except UnicodeEncodeError:
        assert False, "Did not expect UnicodeEncodeError when logging β to UTF-8 stream"