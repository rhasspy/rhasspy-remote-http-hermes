"""Hermes MQTT server for Rhasspy remote server"""
import asyncio
import io
import json
import logging
import ssl
import subprocess
import time
import typing
import wave
from uuid import uuid4

import aiohttp
import attr
from paho.mqtt.matcher import MQTTMatcher
from rhasspyhermes.asr import (
    AsrAudioCaptured,
    AsrError,
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrToggleOff,
    AsrToggleOn,
    AsrTrain,
    AsrTrainSuccess,
)
from rhasspyhermes.audioserver import AudioFrame, AudioPlayBytes, AudioSessionFrame
from rhasspyhermes.base import Message
from rhasspyhermes.handle import HandleToggleOff, HandleToggleOn
from rhasspyhermes.intent import Intent, Slot, SlotRange
from rhasspyhermes.nlu import (
    NluError,
    NluIntent,
    NluIntentNotRecognized,
    NluIntentParsed,
    NluQuery,
    NluTrain,
    NluTrainSuccess,
)
from rhasspyhermes.tts import TtsSay, TtsSayFinished
from rhasspyhermes.wake import HotwordDetected, HotwordToggleOff, HotwordToggleOn
from rhasspysilence import VoiceCommandRecorder, VoiceCommandResult, WebRtcVadRecorder

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

TopicArgs = typing.Mapping[str, typing.Any]
GeneratorType = typing.AsyncIterable[
    typing.Union[Message, typing.Tuple[Message, TopicArgs]]
]


@attr.s(auto_attribs=True, slots=True)
class AsrSession:
    """WAV buffer for an ASR session"""

    start_listening: AsrStartListening
    wav_io: io.BytesIO
    recorder: VoiceCommandRecorder
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
        handle_url: typing.Optional[str] = None,
        handle_command: typing.Optional[typing.List[str]] = None,
        word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        certfile: typing.Optional[str] = None,
        keyfile: typing.Optional[str] = None,
        make_recorder: typing.Callable[[], VoiceCommandRecorder] = None,
        recorder_sample_rate: int = 16000,
        recorder_sample_width: int = 2,
        recorder_channels: int = 1,
        webhooks: typing.Optional[typing.Dict[str, typing.List[str]]] = None,
        siteIds: typing.Optional[typing.List[str]] = None,
        loop=None,
    ):
        self.client = client

        # Speech to text
        self.asr_url = asr_url
        self.asr_command = asr_command
        self.asr_train_url = asr_train_url
        self.asr_train_command = asr_train_command
        self.asr_enabled = True

        # Intent recognition
        self.nlu_url = nlu_url
        self.nlu_command = nlu_command
        self.nlu_train_url = nlu_train_url
        self.nlu_train_command = nlu_train_command

        # Text to speech
        self.tts_url = tts_url
        self.tts_command = tts_command

        # Wake word detection
        self.wake_command = wake_command
        self.wake_enabled = True
        self.wake_proc: typing.Optional[subprocess.Popen] = None
        self.wake_sample_rate = wake_sample_rate
        self.wake_sample_width = wake_sample_width
        self.wake_channels = wake_channels

        # Intent handling
        self.handle_url = handle_url
        self.handle_command = handle_command
        self.handle_enabled = True

        self.word_transform = word_transform

        # SSL
        self.ssl_context = ssl.SSLContext()
        if certfile:
            _LOGGER.debug("Using SSL with certfile=%s, keyfile=%s", certfile, keyfile)
            self.ssl_context.load_cert_chain(certfile, keyfile)

        # Async HTTP
        self.loop = loop or asyncio.get_event_loop()
        self.http_session = aiohttp.ClientSession()

        # No timeout
        def default_recorder():
            return WebRtcVadRecorder(max_seconds=None)

        self.make_recorder = make_recorder or default_recorder
        self.recorder_sample_rate = recorder_sample_rate
        self.recorder_sample_width = recorder_sample_width
        self.recorder_channels = recorder_channels

        # Webhooks
        self.webhook_matcher: typing.Optional[MQTTMatcher] = None
        self.webhook_topics: typing.List[str] = []

        if webhooks:
            self.webhook_matcher = MQTTMatcher()
            self.webhook_topics = list(webhooks.keys())
            for topic, urls in webhooks.items():
                for url in urls:
                    self.webhook_matcher[topic] = url

        self.siteIds = siteIds or []

        # sessionId -> AsrSession
        self.asr_sessions: typing.Dict[str, AsrSession] = {}

        self.first_audio: bool = True

    # -------------------------------------------------------------------------

    async def handle_query(
        self, query: NluQuery
    ) -> typing.AsyncIterable[
        typing.Union[
            typing.Tuple[NluIntent, TopicArgs],
            NluIntentParsed,
            NluIntentNotRecognized,
            NluError,
        ]
    ]:
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

                async with self.http_session.post(
                    self.nlu_url, data=input_text, ssl=self.ssl_context
                ) as response:
                    response.raise_for_status()
                    intent_dict = await response.json()
            elif self.nlu_command:
                # Run external command
                _LOGGER.debug(self.nlu_command)
                proc = await asyncio.create_subprocess_exec(
                    *self.nlu_command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                )

                input_bytes = (input_text.strip() + "\n").encode()
                output, error = await proc.communicate(input_bytes)
                if error:
                    _LOGGER.debug(error.decode())

                intent_dict = json.loads(output)
            else:
                _LOGGER.warning("Not handling NLU query (no URL or command)")
                return

            intent_name = intent_dict["intent"].get("name", "")

            if intent_name:
                # Recognized
                tokens = query.input.split()

                yield NluIntentParsed(
                    input=query.input,
                    id=query.id,
                    siteId=query.siteId,
                    sessionId=query.sessionId,
                    intent=Intent(
                        intentName=intent_name,
                        confidenceScore=intent_dict["intent"].get("confidence", 1.0),
                    ),
                    slots=[
                        Slot(
                            entity=e["entity"],
                            slotName=e["entity"],
                            confidence=1,
                            value=e["value"],
                            raw_value=e.get("raw_value", e["value"]),
                            range=SlotRange(
                                start=e.get("start", 0),
                                end=e.get("end", 1),
                                raw_start=e.get("raw_start"),
                                raw_end=e.get("raw_end"),
                            ),
                        )
                        for e in intent_dict.get("entities", [])
                    ],
                )

                yield (
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
                                    start=e.get("start", 0),
                                    end=e.get("end", 1),
                                    raw_start=e.get("raw_start"),
                                    raw_end=e.get("raw_end"),
                                ),
                            )
                            for e in intent_dict.get("entities", [])
                        ],
                        asrTokens=tokens,
                        rawAsrTokens=tokens,
                    ),
                    {"intentName": intent_name},
                )
            else:
                # Not recognized
                yield NluIntentNotRecognized(
                    input=query.input,
                    id=query.id,
                    siteId=query.siteId,
                    sessionId=query.sessionId,
                )
        except Exception as e:
            _LOGGER.exception("handle_query")
            yield NluError(
                error=repr(e),
                context=repr(query),
                siteId=query.siteId,
                sessionId=query.sessionId,
            )

    # -------------------------------------------------------------------------

    async def handle_say(
        self, say: TtsSay
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[AudioPlayBytes, TopicArgs], TtsSayFinished]
    ]:
        """Do text to speech."""
        _LOGGER.debug("<- %s", say)

        try:
            if self.tts_url:
                # Remote text to speech server
                _LOGGER.debug(self.tts_url)

                params = {}
                if say.lang:
                    params["language"] = say.lang

                async with self.http_session.post(
                    self.tts_url, data=say.text, params=params, ssl=self.ssl_context
                ) as response:
                    response.raise_for_status()
                    content_type = response.headers["Content-Type"]
                    if content_type != "audio/wav":
                        _LOGGER.warning(
                            "Expected audio/wav content type, got %s", content_type
                        )

                    wav_bytes = await response.read()
                    if wav_bytes:
                        yield (
                            AudioPlayBytes(wav_bytes=wav_bytes),
                            {"siteId": say.siteId, "requestId": say.id},
                        )
                    else:
                        _LOGGER.error("Received empty response")
            elif self.tts_command:
                # Local text to speech process
                _LOGGER.debug(self.tts_command)

                proc = await asyncio.create_subprocess_exec(
                    *self.tts_command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                )

                output, error = await proc.communicate()

                if error:
                    _LOGGER.debug(error.decode())

                yield (
                    AudioPlayBytes(wav_bytes=output),
                    {"siteId": say.siteId, "requestId": say.id},
                )

        except Exception:
            _LOGGER.exception("handle_say")
        finally:
            yield TtsSayFinished(id=say.id, sessionId=say.sessionId)

    # -------------------------------------------------------------------------

    def handle_start_listening(self, start_listening: AsrStartListening):
        """Start ASR session."""
        _LOGGER.debug("<- %s", start_listening)

        try:
            wav_io = io.BytesIO()
            session = AsrSession(
                start_listening=start_listening,
                wav_io=wav_io,
                recorder=self.make_recorder(),
            )

            self.asr_sessions[start_listening.sessionId] = session
            session.recorder.start()
        except Exception:
            _LOGGER.exception("handle_start_listening")

    # -------------------------------------------------------------------------

    async def handle_audio_frame(
        self,
        wav_bytes: bytes,
        siteId: str = "default",
        sessionId: typing.Optional[str] = None,
    ) -> typing.AsyncIterable[
        typing.Union[
            typing.Tuple[HotwordDetected, TopicArgs],
            AsrTextCaptured,
            typing.Tuple[AsrAudioCaptured, TopicArgs],
            AsrError,
        ]
    ]:
        """Add audio frame to open sessions."""
        try:
            if self.asr_enabled:
                if sessionId is None:
                    # Add to every open session
                    target_sessions = list(self.asr_sessions.items())
                else:
                    # Add to single session
                    target_sessions = [(sessionId, self.asr_sessions[sessionId])]

                # Add to all open ASR sessions
                with io.BytesIO(wav_bytes) as in_io:
                    with wave.open(in_io) as in_wav:
                        for target_id, session in target_sessions:
                            if session.wav_file is None:
                                session.wav_file = wave.open(session.wav_io, "wb")
                                session.wav_file.setframerate(in_wav.getframerate())
                                session.wav_file.setsampwidth(in_wav.getsampwidth())
                                session.wav_file.setnchannels(in_wav.getnchannels())

                            session.wav_file.writeframes(
                                in_wav.readframes(in_wav.getnframes())
                            )

                            if session.start_listening.stopOnSilence:
                                # Detect silence (end of command)
                                audio_data = RemoteHermesMqtt.maybe_convert_wav(
                                    wav_bytes,
                                    self.recorder_sample_rate,
                                    self.recorder_sample_width,
                                    self.recorder_channels,
                                )
                                command = session.recorder.process_chunk(audio_data)
                                if command and (
                                    command.result == VoiceCommandResult.SUCCESS
                                ):
                                    # Complete session
                                    stop_listening = AsrStopListening(
                                        siteId=siteId, sessionId=target_id
                                    )
                                    async for message in self.handle_stop_listening(
                                        stop_listening
                                    ):
                                        yield message

            if self.wake_enabled and (sessionId is None) and self.wake_proc:
                # Convert and send to wake command
                audio_bytes = RemoteHermesMqtt.maybe_convert_wav(
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
                    yield (
                        HotwordDetected(
                            modelId=wakewordId,
                            modelVersion="",
                            modelType="personal",
                            currentSensitivity=1.0,
                            siteId=siteId,
                        ),
                        {"wakewordId": wakewordId},
                    )

                    # Restart wake process
                    self.start_wake_command()

        except Exception:
            _LOGGER.exception("handle_audio_frame")

    # -------------------------------------------------------------------------

    async def handle_stop_listening(
        self, stop_listening: AsrStopListening
    ) -> typing.AsyncIterable[
        typing.Union[
            AsrTextCaptured, typing.Tuple[AsrAudioCaptured, TopicArgs], AsrError
        ]
    ]:
        """Stop ASR session."""
        _LOGGER.debug("<- %s", stop_listening)

        try:
            session = self.asr_sessions.pop(stop_listening.sessionId, None)
            if session is None:
                _LOGGER.warning("Session not found for %s", stop_listening.sessionId)
                return

            assert session.wav_file
            session.wav_file.close()
            session.recorder.stop()

            # Process entire WAV file
            assert session.wav_io
            wav_bytes = session.wav_io.getvalue()

            if self.asr_url:
                _LOGGER.debug(self.asr_url)

                # Remote ASR server
                async with self.http_session.post(
                    self.asr_url,
                    data=wav_bytes,
                    headers={"Content-Type": "audio/wav", "Accept": "application/json"},
                    ssl=self.ssl_context,
                ) as response:
                    response.raise_for_status()
                    transcription_dict = await response.json()
            elif self.asr_command:
                # Local ASR command
                _LOGGER.debug(self.asr_command)

                start_time = time.perf_counter()
                proc = await asyncio.create_subprocess_exec(
                    *self.asr_command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                output, error = await proc.communicate(wav_bytes)

                if error:
                    _LOGGER.debug(error.decode())

                text = output.decode()
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
            yield AsrTextCaptured(
                text=transcription_dict.get("text", ""),
                likelihood=float(transcription_dict.get("likelihood", 0)),
                seconds=float(transcription_dict.get("transcribe_seconds", 0)),
                siteId=stop_listening.siteId,
                sessionId=stop_listening.sessionId,
            )

            if session.start_listening.sendAudioCaptured:
                # Send audio data
                yield (
                    AsrAudioCaptured(wav_bytes=wav_bytes),
                    {
                        "siteId": stop_listening.siteId,
                        "sessionId": stop_listening.sessionId,
                    },
                )

        except Exception as e:
            _LOGGER.exception("handle_stop_listening")
            yield AsrError(
                error=str(e),
                context=f"url='{self.asr_url}', command='{self.asr_command}'",
                siteId=stop_listening.siteId,
                sessionId=stop_listening.sessionId,
            )

    # -------------------------------------------------------------------------

    async def handle_asr_train(
        self, train: AsrTrain, siteId: str = "default"
    ) -> typing.AsyncIterable[typing.Union[AsrTrainSuccess, AsrError]]:
        """Re-trains ASR system"""
        _LOGGER.debug("<- %s(%s)", train.__class__.__name__, train.id)
        try:
            # Get JSON intent graph
            json_graph = json.dumps(train.graph_dict)

            if self.asr_train_url:
                # Remote ASR server
                _LOGGER.debug(self.asr_train_url)

                async with self.http_session.post(
                    self.asr_train_url, json=json_graph, ssl=self.ssl_context
                ) as response:
                    # No data expected back
                    response.raise_for_status()
            elif self.asr_train_command:
                # Local ASR training command
                _LOGGER.debug(self.asr_train_command)

                proc = await asyncio.create_subprocess_exec(
                    *self.asr_train_command,
                    stdin=asyncio.subprocess.PIPE,
                    sterr=asyncio.subprocess.PIPE,
                )

                output, error = await proc.communicate(json_graph.encode())

                if output:
                    _LOGGER.debug(output.decode())

                if error:
                    _LOGGER.debug(error.decode())
            else:
                _LOGGER.warning("Can't train ASR system. No train URL or command.")

            # Report success
            yield AsrTrainSuccess(id=train.id)
        except Exception as e:
            _LOGGER.exception("handle_asr_train")
            yield AsrError(
                error=str(e),
                context=f"url='{self.asr_train_url}', command='{self.asr_train_command}'",
                siteId=siteId,
                sessionId=train.id,
            )

    # -------------------------------------------------------------------------

    async def handle_nlu_train(
        self, train: NluTrain, siteId: str = "default"
    ) -> typing.AsyncIterable[typing.Union[NluTrainSuccess, NluError]]:
        """Re-trains NLU system"""
        _LOGGER.debug("<- %s(%s)", train.__class__.__name__, train.id)
        try:
            # Get JSON intent graph
            json_graph = json.dumps(train.graph_dict)

            if self.nlu_train_url:
                # Remote NLU server
                _LOGGER.debug(self.nlu_train_url)

                async with self.http_session.post(
                    self.nlu_train_url, json=json_graph, ssl=self.ssl_context
                ) as response:
                    # No data expected in response
                    response.raise_for_status()
            elif self.nlu_train_command:
                # Local NLU training command
                _LOGGER.debug(self.nlu_train_command)

                proc = await asyncio.create_subprocess_exec(
                    *self.nlu_train_command,
                    stdin=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                output, error = await proc.communicate(json_graph.encode())

                if output:
                    _LOGGER.debug(output.decode())

                if error:
                    _LOGGER.debug(error.decode())
            else:
                _LOGGER.warning("Can't train NLU system. No train URL or command.")

            # Report success
            yield NluTrainSuccess(id=train.id)
        except Exception as e:
            _LOGGER.exception("handle_nlu_train")
            yield NluError(
                error=str(e),
                context=f"url='{self.nlu_train_url}', command='{self.nlu_train_command}'",
                siteId=siteId,
                sessionId=train.id,
            )

    # -------------------------------------------------------------------------

    async def handle_intent(
        self, intent: NluIntent
    ) -> typing.AsyncIterable[typing.Union[TtsSay]]:
        """Handle intent with remote server or local command."""
        try:
            if not self.handle_enabled:
                _LOGGER.debug("Intent handling is disabled")
                return

            tts_text = ""
            intent_dict = intent.to_rhasspy_dict()

            # Add siteId
            intent_dict["siteId"] = intent.siteId

            if self.handle_url:
                # Remote server
                _LOGGER.debug(self.handle_url)

                async with self.http_session.post(
                    self.handle_url, json=intent_dict, ssl=self.ssl_context
                ) as response:
                    response.raise_for_status()
                    response_dict = await response.json()

                # Check for speech response
                tts_text = response_dict.get("speech", {}).get("text", "")
            elif self.handle_command:
                intent_json = json.dumps(intent_dict)

                # Local handling command
                _LOGGER.debug(self.handle_command)

                proc = await asyncio.create_subprocess_exec(
                    *self.handle_command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                output, error = await proc.communicate(intent_json.encode())

                if error:
                    _LOGGER.debug(error.decode())

                response_dict = json.loads(output)

                # Check for speech response
                tts_text = response_dict.get("speech", {}).get("text", "")
            else:
                _LOGGER.warning("Can't handle intent. No handle URL or command.")

            if tts_text:
                # Forward to TTS system
                yield TtsSay(
                    text=tts_text,
                    id=str(uuid4()),
                    siteId=intent.siteId,
                    sessionId=intent.sessionId,
                )
        except Exception:
            _LOGGER.exception("handle_intent")

    async def handle_webhook(self, topic: str, payload: bytes):
        """POSTs JSON payload to URL(s)"""
        try:
            assert self.webhook_matcher is not None
            json_payload: typing.Optional[typing.Dict[str, typing.Any]] = None

            # Call for each URL in matching topic
            for webhook_url in self.webhook_matcher.iter_match(topic):

                # Only parse if there's at least one match
                if json_payload is None:
                    # Parse and check siteId
                    json_payload = json.loads(payload)
                    if not self._check_siteId(json_payload):
                        return

                _LOGGER.debug(
                    "webhook %s => %s (%s byte(s))", topic, webhook_url, len(payload)
                )
                async with self.http_session.post(
                    webhook_url, json=json_payload, ssl=self.ssl_context
                ) as response:
                    if response.status != 200:
                        _LOGGER.warning(
                            "Got status %s from %s", response.status, webhook_url
                        )
        except Exception:
            _LOGGER.exception("handle_webhook")

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

            topics = self.webhook_topics

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
                        topics.append(
                            AudioSessionFrame.topic(siteId=siteId, sessionId="+")
                        )

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
                    topics.append(AudioSessionFrame.topic(siteId="+", sessionId="+"))

                # Training
                if self.asr_train_url or self.asr_train_command:
                    topics.append(AsrTrain.topic(siteId="+"))

                if self.nlu_train_url or self.nlu_train_command:
                    topics.append(NluTrain.topic(siteId="+"))

            # NLU
            if self.nlu_url or self.nlu_command:
                topics.append(NluQuery.topic())

            # TTS
            if self.tts_url or self.tts_command:
                topics.append(TtsSay.topic())

            # Intent Handling
            if self.handle_url or self.handle_command:
                topics.extend(
                    [
                        NluIntent.topic(intentName="#"),
                        HandleToggleOn.topic(),
                        HandleToggleOff.topic(),
                    ]
                )

            for topic in topics:
                self.client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)
        except Exception:
            _LOGGER.exception("on_connect")

    def on_message(self, client, userdata, msg):
        """Received message from MQTT broker."""
        try:
            if AudioFrame.is_topic(msg.topic):
                # Check siteId
                siteId = AudioFrame.get_siteId(msg.topic)
                if (not self.siteIds) or (siteId in self.siteIds):
                    # Add to all active sessions
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    # Run outside event loop
                    self.publish_all(
                        self.handle_audio_frame(msg.payload, siteId=siteId)
                    )
            elif AudioSessionFrame.is_topic(msg.topic):
                # Check siteId
                siteId = AudioSessionFrame.get_siteId(msg.topic)
                sessionId = AudioSessionFrame.get_sessionId(msg.topic)
                if ((not self.siteIds) or (siteId in self.siteIds)) and (
                    sessionId in self.asr_sessions
                ):
                    # Add to active session
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    # Run outside event loop
                    self.publish_all(
                        self.handle_audio_frame(
                            msg.payload, siteId=siteId, sessionId=sessionId
                        )
                    )
            elif msg.topic == NluQuery.topic():
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.publish_all(
                        self.handle_query(NluQuery.from_dict(json_payload))
                    )
            elif msg.topic == TtsSay.topic():
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.publish_all(self.handle_say(TtsSay.from_dict(json_payload)))
            elif msg.topic == AsrStartListening.topic():
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    # Run outside event loop
                    self.handle_start_listening(
                        AsrStartListening.from_dict(json_payload)
                    )
            elif msg.topic == AsrStopListening.topic():
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.publish_all(
                        self.handle_stop_listening(
                            AsrStopListening.from_dict(json_payload)
                        )
                    )
            elif AsrTrain.is_topic(msg.topic):
                siteId = AsrTrain.get_siteId(msg.topic)
                if (not self.siteIds) or (siteId in self.siteIds):
                    json_payload = json.loads(msg.payload)
                    self.publish_all(
                        self.handle_asr_train(
                            AsrTrain.from_dict(json_payload), siteId=siteId
                        )
                    )
            elif NluTrain.is_topic(msg.topic):
                siteId = NluTrain.get_siteId(msg.topic)
                if (not self.siteIds) or (siteId in self.siteIds):
                    json_payload = json.loads(msg.payload)
                    self.publish_all(
                        self.handle_nlu_train(
                            NluTrain.from_dict(json_payload), siteId=siteId
                        )
                    )
            elif NluIntent.is_topic(msg.topic):
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.publish_all(
                        self.handle_intent(NluIntent.from_dict(json_payload))
                    )
            elif AsrToggleOn.is_topic(msg.topic):
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.asr_enabled = True
                    _LOGGER.debug("ASR enabled")
            elif AsrToggleOff.is_topic(msg.topic):
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.asr_enabled = False
                    _LOGGER.debug("ASR disabled")
            elif HotwordToggleOn.is_topic(msg.topic):
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.wake_enabled = True
                    _LOGGER.debug("Wake word detection enabled")
            elif HotwordToggleOff.is_topic(msg.topic):
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.wake_enabled = False
                    _LOGGER.debug("Wake word detection disabled")
            elif HandleToggleOn.is_topic(msg.topic):
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.handle_enabled = True
                    _LOGGER.debug("Intent handling enabled")
            elif HandleToggleOff.is_topic(msg.topic):
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.handle_enabled = False
                    _LOGGER.debug("Intent handling disabled")

            # Webhooks
            if self.webhook_matcher:
                asyncio.run_coroutine_threadsafe(
                    self.handle_webhook(msg.topic, msg.payload), self.loop
                )

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
                payload = json.dumps(attr.asdict(message)).encode()

            topic = message.topic(**topic_args)
            _LOGGER.debug("Publishing %s char(s) to %s", len(payload), topic)
            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("publish")

    def publish_all(self, async_generator: GeneratorType):
        """Publish all messages from an async generator"""
        asyncio.run_coroutine_threadsafe(
            self.async_publish_all(async_generator), self.loop
        )

    async def async_publish_all(self, async_generator: GeneratorType):
        """Enumerate all messages in an async generator publish them"""
        async for maybe_message in async_generator:
            if isinstance(maybe_message, Message):
                self.publish(maybe_message)
            else:
                message, kwargs = maybe_message
                self.publish(message, **kwargs)

    # -------------------------------------------------------------------------

    def _check_siteId(self, json_payload: typing.Dict[str, typing.Any]) -> bool:
        if self.siteIds:
            return json_payload.get("siteId", "default") in self.siteIds

        # All sites
        return True

    @classmethod
    def convert_wav(
        cls, wav_bytes: bytes, sample_rate: int, sample_width: int, channels: int
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

    @classmethod
    def maybe_convert_wav(
        cls, wav_bytes: bytes, sample_rate: int, sample_width: int, channels: int
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
                    return RemoteHermesMqtt.convert_wav(
                        wav_bytes, sample_rate, sample_width, channels
                    )

                # Return original audio
                return wav_file.readframes(wav_file.getnframes())
