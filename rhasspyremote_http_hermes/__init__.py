"""Hermes MQTT server for Rhasspy remote server"""
import io
import json
import logging
import subprocess
import time
import typing
import wave

import attr
import requests
from rhasspyhermes.asr import (
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrError,
    AsrAudioCaptured,
)
from rhasspyhermes.audioserver import AudioFrame, AudioPlayBytes
from rhasspyhermes.base import Message
from rhasspyhermes.intent import Intent, Slot, SlotRange
from rhasspyhermes.nlu import NluError, NluIntent, NluIntentNotRecognized, NluQuery
from rhasspyhermes.tts import TtsSay, TtsSayFinished

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


@attr.s(auto_attribs=True, slots=True)
class AsrSession:
    """WAV buffer for an ASR session"""

    start_listening: AsrStartListening
    wav_io: io.BytesIO
    wav_file: typing.Optional[wave.Wave_write] = None


# -----------------------------------------------------------------------------


class RemoteHermesMqtt:
    """Hermes MQTT server for Rhasspy remote server."""

    def __init__(
        self,
        client,
        asr_url: typing.Optional[str] = None,
        asr_command: typing.Optional[typing.List[str]] = None,
        nlu_url: typing.Optional[str] = None,
        nlu_command: typing.Optional[typing.List[str]] = None,
        tts_url: typing.Optional[str] = None,
        tts_command: typing.Optional[typing.List[str]] = None,
        word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        siteIds: typing.Optional[typing.List[str]] = None,
    ):
        self.client = client

        self.asr_url = asr_url
        self.asr_command = asr_command

        self.nlu_url = nlu_url
        self.nlu_command = nlu_command

        self.tts_url = tts_url
        self.tts_command = tts_command

        self.word_transform = word_transform
        self.siteIds = siteIds or []

        # sessionId -> AsrSession
        self.asr_sessions: typing.Dict[str, AsrSession] = {}

        self.first_audio: bool = True

    # -------------------------------------------------------------------------

    def handle_query(self, query: NluQuery):
        """Do intent recognition."""
        _LOGGER.debug("<- %s", query)

        try:
            input_text = query.input

            # Fix casing
            if self.word_transform:
                input_text = self.word_transform(input_text)

            if self.nlu_url:
                # Use remote server
                _LOGGER.debug(self.nlu_url)
                response = requests.post(self.nlu_url, data=input_text)
                response.raise_for_status()
                intent_dict = response.json()
            elif self.nlu_command:
                # Run external command
                _LOGGER.debug(self.nlu_command)
                proc = subprocess.Popen(
                    self.nlu_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    universal_newlines=True,
                )

                print(input_text, file=proc.stdin)
                output, _ = proc.communicate()

                intent_dict = json.loads(output)
            else:
                _LOGGER.warning("Not handling NLU query (no URL or command)")
                return

            intent_name = intent_dict["intent"].get("name", "")

            if intent_name:
                # Recognized
                self.publish(
                    NluIntent(
                        input=query.input,
                        id=query.id,
                        siteId=query.siteId,
                        sessionId=query.sessionId,
                        intent=Intent(
                            intentName=intent_name,
                            confidenceScore=intent_dict["intent"].get(
                                "confidence", 1.0
                            ),
                        ),
                        slots=[
                            Slot(
                                entity=e["entity"],
                                slotName=e["entity"],
                                confidence=1,
                                value=e["value"],
                                raw_value=e.get("raw_value", e["value"]),
                                range=SlotRange(
                                    start=e.get("raw_start", e.get("start", 0)),
                                    end=e.get("raw_end", e.get("end", 1)),
                                ),
                            )
                            for e in intent_dict.get("entities", [])
                        ],
                    ),
                    intentName=intent_name,
                )
            else:
                # Not recognized
                self.publish(
                    NluIntentNotRecognized(
                        input=query.input,
                        id=query.id,
                        siteId=query.siteId,
                        sessionId=query.sessionId,
                    )
                )
        except Exception as e:
            _LOGGER.exception("handle_query")
            self.publish(
                NluError(error=repr(e), context=repr(query)),
                siteId=query.siteId,
                sessionId=query.sessionId,
            )

    # -------------------------------------------------------------------------

    def handle_say(self, say: TtsSay):
        """Do text to speech."""
        _LOGGER.debug("<- %s", say)

        try:
            if self.tts_url:
                post_args = {"data": say.text}
                if say.lang:
                    post_args["language"] = say.lang

                response = requests.post(self.tts_url, **post_args)
                response.raise_for_status()

                content_type = response.headers["Content-Type"]
                if content_type == "audio/wav":
                    self.publish(
                        AudioPlayBytes.topic(siteId=say.siteId, requestId=say.id),
                        response.content,
                    )
                else:
                    _LOGGER.warning(
                        "Expected audio/wav content type, got %s", content_type
                    )
            elif self.tts_command:
                _LOGGER(self.tts_command)
                proc = subprocess.run(
                    self.tts_command,
                    input=say.text.encode(),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                if proc.stderr:
                    _LOGGER.debug(proc.stderr.decode())

                self.client.publish(
                    AudioPlayBytes(proc.stdout), siteId=say.siteId, requestId=say.id
                )

        except Exception:
            _LOGGER.exception("handle_say")
        finally:
            self.publish(TtsSayFinished(id=say.id, sessionId=say.sessionId))

    # -------------------------------------------------------------------------

    def handle_start_listening(self, start_listening: AsrStartListening):
        """Start ASR session."""
        _LOGGER.debug("<- %s", start_listening)

        try:
            wav_io = io.BytesIO()
            self.asr_sessions[start_listening.sessionId] = AsrSession(
                start_listening=start_listening, wav_io=wav_io
            )
        except Exception:
            _LOGGER.exception("handle_start_listening")

    # -------------------------------------------------------------------------

    def handle_audio_frame(self, wav_bytes: bytes, siteId: str = "default"):
        """Add audio frame to open sessions."""
        try:
            # Add to all open sessions
            with io.BytesIO(wav_bytes) as in_io:
                with wave.open(in_io) as in_wav:
                    for session in self.asr_sessions.values():
                        if session.wav_file is None:
                            session.wav_file = wave.open(session.wav_io, "wb")
                            session.wav_file.setframerate(in_wav.getframerate())
                            session.wav_file.setsampwidth(in_wav.getsampwidth())
                            session.wav_file.setnchannels(in_wav.getnchannels())

                        session.wav_file.writeframes(
                            in_wav.readframes(in_wav.getnframes())
                        )

        except Exception:
            _LOGGER.exception("handle_audio_frame")

    # -------------------------------------------------------------------------

    def handle_stop_listening(self, stop_listening: AsrStopListening):
        """Stop ASR session."""
        _LOGGER.debug("<- %s", stop_listening)

        try:
            session = self.asr_sessions.pop(stop_listening.sessionId)
            assert session.wav_file
            session.wav_file.close()

            # Process entire WAV file
            assert session.wav_io
            wav_bytes = session.wav_io.getvalue()

            if self.asr_url:
                # Remote ASR server
                response = requests.post(
                    self.asr_url,
                    data=wav_bytes,
                    headers={"Content-Type": "audio/wav", "Accept": "application/json"},
                )
                response.raise_for_status()

                transcription_dict = response.json()
            elif self.asr_command:
                # Local ASR command
                _LOGGER.debug(self.asr_command)

                start_time = time.perf_counter()
                proc = subprocess.run(
                    self.asr_command,
                    input=wav_bytes,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )

                if proc.stderr:
                    _LOGGER.debug(proc.stderr.decode())

                text = proc.stdout.decode()
                end_time = time.perf_counter()

                transcription_dict = {
                    "text": text,
                    "transcribe_seconds": (end_time - start_time),
                }
            else:
                # Empty transcription
                _LOGGER.warning(
                    "No ASR URL or command. Only empty transcriptions will be returned."
                )
                transcription_dict = {}

            # Publish transcription
            self.publish(
                AsrTextCaptured(
                    text=transcription_dict.get("text", ""),
                    likelihood=float(transcription_dict.get("likelihood", 0)),
                    seconds=float(transcription_dict.get("transcribe_seconds", 0)),
                    siteId=stop_listening.siteId,
                    sessionId=stop_listening.sessionId,
                )
            )

            if session.start_listening.sendAudioCaptured:
                # Send audio data
                self.publish(
                    AsrAudioCaptured(wav_bytes),
                    siteId=stop_listening.siteId,
                    sessionId=stop_listening.sessionId,
                )

        except Exception as e:
            _LOGGER.exception("handle_stop_listening")
            self.publish(
                AsrError(
                    error=str(e),
                    context=f"url='{self.asr_url}', command='{self.asr_command}'",
                    siteId=stop_listening.siteId,
                    sessionId=stop_listening.sessionId,
                )
            )

    # -------------------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        """Connected to MQTT broker."""
        try:
            topics = []

            if self.asr_url or self.asr_command:
                topics.extend([AsrStartListening.topic(), AsrStopListening.topic()])

                # Subscribe to audio too
                if self.siteIds:
                    # Specific site ids
                    topics.extend(
                        AudioFrame.topic(siteId=siteId) for siteId in self.siteIds
                    )
                else:
                    # All site ids
                    topics.append(AudioFrame.topic(siteId="+"))

            if self.nlu_url or self.nlu_command:
                topics.append(NluQuery.topic())

            if self.tts_url or self.tts_command:
                topics.append(TtsSay.topic())

            for topic in topics:
                self.client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)
        except Exception:
            _LOGGER.exception("on_connect")

    def on_message(self, client, userdata, msg):
        """Received message from MQTT broker."""
        try:
            if not msg.topic.endswith("/audioFrame"):
                _LOGGER.debug("Received %s byte(s) on %s", len(msg.payload), msg.topic)

            if AudioFrame.is_topic(msg.topic):
                # Check siteId
                siteId = AudioFrame.get_siteId(msg.topic)
                if (not self.siteIds) or (siteId in self.siteIds):
                    # Add to all active sessions
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    self.handle_audio_frame(msg.payload, siteId=siteId)
            elif msg.topic == NluQuery.topic():
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload):
                    return

                self.handle_query(NluQuery(**json_payload))
            elif msg.topic == TtsSay.topic():
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload):
                    return

                self.handle_say(TtsSay(**json_payload))
            elif msg.topic == AsrStartListening.topic():
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload):
                    return

                self.handle_start_listening(AsrStartListening(**json_payload))
            elif msg.topic == AsrStopListening.topic():
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload):
                    return

                self.handle_stop_listening(AsrStopListening(**json_payload))
        except Exception:
            _LOGGER.exception("on_message")

    def publish(self, message: Message, **topic_args):
        """Publish a Hermes message to MQTT."""
        try:
            if isinstance(message, (AudioPlayBytes, AsrAudioCaptured)):
                _LOGGER.debug(
                    "-> %s(%s byte(s))",
                    message.__class__.__name__,
                    len(message.wav_bytes),
                )
                payload = message.wav_bytes
            else:
                _LOGGER.debug("-> %s", message)
                payload = json.dumps(attr.asdict(message))

            topic = message.topic(**topic_args)
            _LOGGER.debug("Publishing %s char(s) to %s", len(payload), topic)
            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("on_message")

    # -------------------------------------------------------------------------

    def _check_siteId(self, json_payload: typing.Dict[str, typing.Any]) -> bool:
        if self.siteIds:
            return json_payload.get("siteId", "default") in self.siteIds

        # All sites
        return True
