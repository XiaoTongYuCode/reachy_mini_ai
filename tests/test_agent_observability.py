import logging

from reachy_mini_conversation_app.agent_observability import AgentMessageListLog


def test_agent_message_list_log_redacts_inline_images_and_response_params(caplog) -> None:  # type: ignore[no-untyped-def]
    """Agent message-list logs should be readable and avoid dumping inline image data."""
    logger = logging.getLogger("tests.agent_observability")
    message_log = AgentMessageListLog(logger)

    caplog.set_level(logging.INFO, logger=logger.name)
    message_log.reset("system instructions")
    message_log.append(
        "user",
        [
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64,abc123",
            },
        ],
    )
    message_log.log(
        "response.create",
        response_kwargs={
            "response": {
                "instructions": "answer with tool calls only",
                "tool_choice": "required",
            },
        },
    )

    output = caplog.records[-1].getMessage()
    assert "LLM message_list (response.create)" in output
    assert "system instructions" in output
    assert "<inline image data url," in output
    assert "abc123" not in output
    assert "answer with tool calls only" in output
    assert "tool_choice=required" in output


def test_agent_message_list_log_once_per_turn(caplog) -> None:  # type: ignore[no-untyped-def]
    """Repeated model-start callbacks in one turn should print once."""
    logger = logging.getLogger("tests.agent_observability.once")
    message_log = AgentMessageListLog(logger)

    caplog.set_level(logging.INFO, logger=logger.name)
    message_log.reset("system instructions")
    message_log.append("user", "hello")

    message_log.log_once_for_turn("model response")
    message_log.log_once_for_turn("model response")
    assert len(caplog.records) == 1

    message_log.reset_turn_log()
    message_log.log_once_for_turn("model response")
    assert len(caplog.records) == 2


def test_agent_message_list_log_replaces_scoped_messages(caplog) -> None:  # type: ignore[no-untyped-def]
    """Scoped observability messages should be replaced instead of duplicated."""
    logger = logging.getLogger("tests.agent_observability.scoped")
    message_log = AgentMessageListLog(logger)

    caplog.set_level(logging.INFO, logger=logger.name)
    message_log.reset("system instructions")
    message_log.set_scoped_message("system", "old memory", scope="memory_context")
    message_log.set_scoped_message("system", "new memory", scope="memory_context")
    message_log.append("user", "hello")
    message_log.log("model response")

    output = caplog.records[-1].getMessage()
    assert "new memory" in output
    assert "old memory" not in output

    message_log.set_scoped_message("system", "", scope="memory_context")
    message_log.log("model response")
    output = caplog.records[-1].getMessage()
    assert "new memory" not in output
