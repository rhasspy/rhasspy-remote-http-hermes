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
    AsrTrain,
    AsrTrainSuccess,
    AsrToggleOn,
    AsrToggleOff,
)
from rhasspyhermes.audioserver import AudioFrame, AudioPlayBytes
from rhasspyhermes.base import Message
from rhasspyhermes.intent import Intent, Slot, SlotRange
from rhasspyhermes.nlu import (
    NluError,
    NluIntent,
    NluIntentNotRecognized,
    NluQuery,
    NluTrain,
    NluTrainSuccess,
)
from rhasspyhermes.tts import TtsSay, TtsSayFinished
from rhasspyhermes.wake import HotwordDetected, HotwordToggleOn, HotwordToggleOff

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
        asr_train_url: typing.Optional[str] = None,
        asr_train_command: typing.Optional[typing.List[str]] = None,
        nlu_url: typing.Optional[str] = None,
        nlu_command: typing.Optional[typing.List[str]] = None,
        nlu_train_url: typing.Optional[str] = None,
        nlu_train_command: typing.Optional[typing.List[str]] = None,
        tts_url: typing.Optional[str] = None,
        tts_command: typing.Optional[typing.List[str]] = None,
        wake_command: typing.Optional[typing.List[str]] = None,
        wake_sample_rate: int = 16000,
        wake_sample_width: int = 2,
        wake_channels: int = 1,
        word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        siteIds: typing.Optional[typing.List[str]] = None,
    ):
        self.client = client

        self.asr_url = asr_url
        self.asr_command = asr_command
        self.asr_train_url = asr_train_url
        self.asr_train_command = asr_train_command
        self.asr_enabled = True

        self.nlu_url = nlu_url
        self.nlu_command = nlu_command
        self.nlu_train_url = nlu_train_url
        self.nlu_train_command = nlu_train_command

        self.tts_url = tts_url
        self.tts_command = tts_command

        self.wake_command = wake_command
        self.wake_enabled = True
        self.wake_proc: typing.Optional[subprocess.Popen] = None
        self.wake_sample_rate = wake_sample_rate
        self.wake_sample_width = wake_sample_width
        self.wake_channels = wake_channels

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
                if content_type != "audio/wav":
                    _LOGGER.warning(
                        "Expected audio/wav content type, got %s", content_type
                    )

                wav_bytes = response.content
                if wav_bytes:
                    self.publish(
                        AudioPlayBytes(wav_bytes), siteId=say.siteId, requestId=say.id
                    )
                else:
                    _LOGGER.error("Received empty response")
            elif self.tts_command:
                _LOGGER.debug(self.tts_command)
                proc = subprocess.run(
                    self.tts_command,
                    input=say.text.encode(),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                if proc.stderr:
                    _LOGGER.debug(proc.stderr.decode())

                self.publish(
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
            if self.asr_enabled:
                # Add to all open ASR sessions
                # TODO: Convert WAV
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

            if self.wake_enabled and self.wake_proc:
                # Convert and send to wake command
                audio_bytes = self.maybe_convert_wav(
                    wav_bytes,
                    self.wake_sample_rate,
                    self.wake_sample_width,
                    self.wake_channels,
                )
                self.wake_proc.stdin.write(audio_bytes)
                if self.wake_proc.poll():
                    stdout, stderr = self.wake_proc.communicate()
                    if stderr:
                        _LOGGER.debug(stderr.decode())

                    wakewordId = stdout.decode().strip()
                    _LOGGER.debug("Detected wake word %s", wakewordId)
                    self.publish(
                        HotwordDetected(
                            modeId=wakewordId,
                            modelVersion="",
                            modelType="personal",
                            siteId=siteId,
                        ),
                        wakewordId=wakewordId,
                    )

                    # Restart wake process
                    self.start_wake_command()

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

    def handle_asr_train(self, train: AsrTrain, siteId: str = "default"):
        """Re-trains ASR system"""
        _LOGGER.debug("<- %s(%s)", train.__class__.__name__, train.id)
        try:
            # Get JSON intent graph
            json_graph = json.dumps(train.graph_dict)

            if self.asr_train_url:
                # Remote ASR server
                response = requests.post(self.asr_train_url, json=json_graph)
                response.raise_for_status()
            elif self.asr_train_command:
                # Local ASR training command
                _LOGGER.debug(self.asr_train_command)

                proc = subprocess.run(
                    self.asr_train_command,
                    input=json_graph,
                    stderr=subprocess.PIPE,
                    check=True,
                )

                if proc.stderr:
                    _LOGGER.debug(proc.stderr.decode())
            else:
                _LOGGER.warning("Can't train ASR system. No train URL or command.")

            # Report success
            self.publish(AsrTrainSuccess(id=train.id))
        except Exception as e:
            _LOGGER.exception("handle_asr_train")
            self.publish(
                AsrError(
                    error=str(e),
                    context=f"url='{self.asr_train_url}', command='{self.asr_train_command}'",
                    siteId=siteId,
                    sessionId=train.id,
                )
            )

    # -------------------------------------------------------------------------

    def handle_nlu_train(self, train: NluTrain, siteId: str = "default"):
        """Re-trains NLU system"""
        _LOGGER.debug("<- %s(%s)", train.__class__.__name__, train.id)
        try:
            # Get JSON intent graph
            json_graph = json.dumps(train.graph_dict)

            if self.nlu_train_url:
                # Remote NLU server
                response = requests.post(self.nlu_train_url, json=json_graph)
                response.raise_for_status()
            elif self.nlu_train_command:
                # Local NLU training command
                _LOGGER.debug(self.nlu_train_command)

                proc = subprocess.run(
                    self.nlu_train_command,
                    input=json_graph,
                    stderr=subprocess.PIPE,
                    check=True,
                )

                if proc.stderr:
                    _LOGGER.debug(proc.stderr.decode())
            else:
                _LOGGER.warning("Can't train NLU system. No train URL or command.")

            # Report success
            self.publish(NluTrainSuccess(id=train.id))
        except Exception as e:
            _LOGGER.exception("handle_nlu_train")
            self.publish(
                NluError(
                    error=str(e),
                    context=f"url='{self.nlu_train_url}', command='{self.nlu_train_command}'",
                    siteId=siteId,
                    sessionId=train.id,
                )
            )

    # -------------------------------------------------------------------------

    def start_wake_command(self):
        """Run wake command."""
        self.stop_wake_command()

        try:
            _LOGGER.debug(self.wake_command)
            self.wake_proc = subprocess.Popen(
                self.wake_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception:
            _LOGGER.exception("start_wake_command")

    def stop_wake_command(self):
        """Terminate wake command."""
        try:
            if self.wake_proc:
                self.wake_proc.terminate()
                self.wake_proc.wait()
                _LOGGER.debug("Wake command terminated.")

            self.wake_proc = None
        except Exception:
            _LOGGER.exception("stop_wake_command")

    # -------------------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        """Connected to MQTT broker."""
        try:
            if self.wake_command:
                self.start_wake_command()

            topics = []

            # ASR
            if self.asr_url or self.asr_command:
                topics.extend(
                    [
                        AsrStartListening.topic(),
                        AsrStopListening.topic(),
                        AsrToggleOn.topic(),
                        AsrToggleOff.topic(),
                    ]
                )

            # Wake
            if self.wake_command:
                topics.extend([HotwordToggleOn.topic(), HotwordToggleOff.topic()])

            if self.siteIds:
                # Specific site ids
                for siteId in self.siteIds:
                    # ASR audio
                    if self.asr_url or self.asr_command:
                        topics.append(AudioFrame.topic(siteId=siteId))

                    # Training
                    if self.asr_train_url or self.asr_train_command:
                        topics.append(AsrTrain.topic(siteId=siteId))

                    if self.nlu_train_url or self.nlu_train_command:
                        topics.append(NluTrain.topic(siteId=siteId))
            else:
                # All site ids
                if self.asr_url or self.asr_command:
                    # ASR audio
                    topics.append(AudioFrame.topic(siteId="+"))

                # Training
                if self.asr_train_url or self.asr_train_command:
                    topics.append(
                        AudioFrame.topic(siteId="+"), AsrTrain.topic(siteId="+")
                    )

                if self.nlu_train_url or self.nlu_train_command:
                    topics.append(
                        AudioFrame.topic(siteId="+"), NluTrain.topic(siteId="+")
                    )

            # NLU
            if self.nlu_url or self.nlu_command:
                topics.append(NluQuery.topic())

            # TTS
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
            elif AsrTrain.is_topic(msg.topic):
                siteId = AsrTrain.get_siteId(msg.topic)
                if (not self.siteIds) or (siteId in self.siteIds):
                    json_payload = json.loads(msg.payload)
                    self.handle_asr_train(AsrTrain(**json_payload), siteId=siteId)
            elif NluTrain.is_topic(msg.topic):
                siteId = NluTrain.get_siteId(msg.topic)
                if (not self.siteIds) or (siteId in self.siteIds):
                    json_payload = json.loads(msg.payload)
                    self.handle_nlu_train(NluTrain(**json_payload), siteId=siteId)
            elif AsrToggleOn.is_topic(msg.topic):
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload):
                    return

                self.asr_enabled = True
                _LOGGER.debug("ASR enabled")
            elif AsrToggleOff.is_topic(msg.topic):
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload):
                    return

                self.asr_enabled = False
                _LOGGER.debug("ASR disabled")
            elif HotwordToggleOn.is_topic(msg.topic):
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload):
                    return

                self.wake_enabled = True
                _LOGGER.debug("Wake word detection enabled")
            elif HotwordToggleOff.is_topic(msg.topic):
                json_payload = json.loads(msg.payload)
                if not self._check_siteId(json_payload):
                    return

                self.wake_enabled = False
                _LOGGER.debug("Wake word detection disabled")
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

    def _convert_wav(
        self, wav_bytes: bytes, sample_rate: int, sample_width: int, channels: int
    ) -> bytes:
        """Converts WAV data to required format with sox. Return raw audio."""
        return subprocess.run(
            [
                "sox",
                "-t",
                "wav",
                "-",
                "-r",
                str(sample_rate),
                "-e",
                "signed-integer",
                "-b",
                str(sample_width * 8),
                "-c",
                str(channels),
                "-t",
                "raw",
                "-",
            ],
            check=True,
            stdout=subprocess.PIPE,
            input=wav_bytes,
        ).stdout

    def maybe_convert_wav(
        self, wav_bytes: bytes, sample_rate: int, sample_width: int, channels: int
    ) -> bytes:
        """Converts WAV data to required format if necessary. Returns raw audio."""
        with io.BytesIO(wav_bytes) as wav_io:
            with wave.open(wav_io, "rb") as wav_file:
                if (
                    (wav_file.getframerate() != sample_rate)
                    or (wav_file.getsampwidth() != sample_width)
                    or (wav_file.getnchannels() != channels)
                ):
                    # Return converted wav
                    return self._convert_wav(
                        wav_bytes, sample_rate, sample_width, channels
                    )

                # Return original audio
                return wav_file.readframes(wav_file.getnframes())
