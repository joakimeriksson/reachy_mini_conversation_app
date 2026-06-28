"""Backend-agnostic conversation logic shared by the robot handler and the fake-robot runner.

The turn pipeline (understand an utterance → reply) and streamed synthesis live here so
both I/O front-ends (fastrtc ``ollama_handler`` and sounddevice ``local_chat``) delegate to
one tested implementation instead of duplicating it.
"""
