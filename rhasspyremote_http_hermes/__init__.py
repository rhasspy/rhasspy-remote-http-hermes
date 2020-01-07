"""Hermes MQTT server for Rhasspy remote server"""
import io
import json
import logging
import typing
import wave

import attr
from rhasspyhermes.asr import AsrStartListening, AsrStopListening, AsrTextCaptured
from rhasspyhermes.base import Message
from rhasspyhermes.audioserver import AudioPlayBytes, AudioFrame
from rhasspyhermes.intent import Intent, Slot, SlotRange
from rhasspyhermes.nlu import NluError, NluIntent, NluIntentNotRecognized, NluQuery
from rhasspyhermes.tts import TtsSay, TtsSayFinished

import requests

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


@attr.s
class AsrSession:
    """WAV buffer for an ASR session"""

    start_listening: AsrStartListening = attr.ib()
    wav_io: typing.BinaryIO = attr.ib()
    wav_file: typing.Optional[wave.Wave_write] = attr.ib(default=None)


# -----------------------------------------------------------------------------


class RemoteHermesMqtt:
    """Hermes MQTT server for Rhasspy remote server."""

    def __init__(
        self,
        client,
        nlu_url: typing.Optional[str] = None,
        asr_url: typing.Optional[str] = None,
        tts_url: typing.Optional[str] = None,
        siteIds: typing.Optional[typing.List[str]] = None,
    ):
        self.client = client
        self.nlu_url = nlu_url
        self.asr_url = asr_url
        self.tts_url = tts_url
        self.siteIds = siteIds or []

        # sessionId -> AsrSession
        self.asr_sessions: typing.Dict[str, AsrSession] = {}

        # Topic to listen for WAV chunks on
        self.audioframe_topics: typing.List[str] = []
        for siteId in self.siteIds:
            self.audioframe_topics.append(AudioFrame.topic(siteId=siteId))

        self.first_audio: bool = True

    # -------------------------------------------------------------------------

    def handle_query(self, query: NluQuery):
        """Do intent recognition."""
        _LOGGER.debug("<- %s", query)

        try:
            response = requests.post(self.nlu_url, data=query.input)
            response.raise_for_status()
            intent_dict = response.json()
            intent_name = intent_dict["intent"]["name"]

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
                            confidenceScore=intent_dict["intent"]["confidence"],
                        ),
                        slots=[
                            Slot(
                                entity=e["entity"],
                                slotName=e["entity"],
                                confidence=1,
                                value=e["value"],
                                raw_value=e["raw_value"],
                                range=SlotRange(start=e["raw_start"], end=e["raw_end"]),
                            )
                            for e in intent_dict["entities"]
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
            post_args = {"data": say.text}
            if say.lang:
                post_args["language"] = say.lang

            response = requests.post(self.tts_url, **post_args)
            response.raise_for_status()

            print(response.headers)
            if response.headers["Content-Type"] == "audio/wav":
                self.client.publish(
                    AudioPlayBytes.topic(siteId=say.siteId, requestId=say.id),
                    response.content,
                )

            self.publish(TtsSayFinished(id=say.id, sessionId=say.sessionId))
        except Exception:
            _LOGGER.exception("handle_say")

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
            # Add to all open session
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
            session.wav_file.close()

            # Post entire WAV file
            response = requests.post(
                self.asr_url,
                data=session.wav_io.getvalue(),
                headers={"Content-Type": "audio/wav", "Accept": "application/json"},
            )
            response.raise_for_status()

            transcription_dict = response.json()
            self.publish(
                AsrTextCaptured(
                    text=transcription_dict["text"],
                    likelihood=transcription_dict["likelihood"],
                    seconds=transcription_dict["transcribe_seconds"],
                    siteId=stop_listening.siteId,
                    sessionId=stop_listening.sessionId,
                )
            )

        except Exception:
            _LOGGER.exception("handle_stop_listening")

    # -------------------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        """Connected to MQTT broker."""
        try:
            topics = [
                NluQuery.topic(),
                AsrStartListening.topic(),
                AsrStopListening.topic(),
                TtsSay.topic(),
            ]

            if self.audioframe_topics:
                # Specific siteIds
                topics.extend(self.audioframe_topics)
            else:
                # All siteIds
                topics.append(AudioFrame.topic(siteId="+"))

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
                if (not self.audioframe_topics) or (
                    msg.topic in self.audioframe_topics
                ):
                    # Add to all active sessions
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    siteId = AudioFrame.get_siteId(msg.topic)
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
            _LOGGER.debug("-> %s", message)
            topic = message.topic(**topic_args)
            payload = json.dumps(attr.asdict(message))
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
