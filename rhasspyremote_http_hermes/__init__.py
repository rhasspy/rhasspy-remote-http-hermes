"""Hermes MQTT server for Rhasspy remote server"""
import asyncio
import gzip
import io
import json
import logging
import ssl
import subprocess
import time
import typing
import wave
from dataclasses import dataclass
from uuid import uuid4

import aiohttp
import networkx as nx
import rhasspynlu
from paho.mqtt.matcher import MQTTMatcher
from rhasspyhermes.asr import (
    AsrAudioCaptured,
    AsrError,
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrToggleOff,
    AsrToggleOn,
    AsrToggleReason,
    AsrTrain,
    AsrTrainSuccess,
)
from rhasspyhermes.audioserver import AudioFrame, AudioPlayBytes, AudioSessionFrame
from rhasspyhermes.base import Message
from rhasspyhermes.client import GeneratorType, HermesClient, TopicArgs
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
from rhasspyhermes.tts import TtsError, TtsSay, TtsSayFinished
from rhasspyhermes.wake import (
    HotwordDetected,
    HotwordToggleOff,
    HotwordToggleOn,
    HotwordToggleReason,
)
from rhasspysilence import VoiceCommandRecorder, VoiceCommandResult, WebRtcVadRecorder

_LOGGER = logging.getLogger("rhasspyremote_http_hermes")

# -----------------------------------------------------------------------------


@dataclass
class AsrSession:
    """WAV buffer for an ASR session"""

    start_listening: AsrStartListening
    recorder: VoiceCommandRecorder
    sample_rate: typing.Optional[int] = None
    sample_width: typing.Optional[int] = None
    channels: typing.Optional[int] = None
    audio_data: bytes = bytes()


# -----------------------------------------------------------------------------


class RemoteHermesMqtt(HermesClient):
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
        skip_seconds: float = 0.0,
        min_seconds: float = 1.0,
        speech_seconds: float = 0.3,
        silence_seconds: float = 0.5,
        before_seconds: float = 0.5,
        vad_mode: int = 3,
        site_ids: typing.Optional[typing.List[str]] = None,
    ):
        super().__init__("rhasspyremote_http_hermes", client, site_ids=site_ids)

        # Speech to text
        self.asr_url = asr_url
        self.asr_command = asr_command
        self.asr_train_url = asr_train_url
        self.asr_train_command = asr_train_command
        self.asr_enabled = True
        self.asr_used = self.asr_url or self.asr_command
        self.asr_train_used = self.asr_train_url or self.asr_train_command
        self.asr_disabled_reasons: typing.Set[str] = set()

        # Intent recognition
        self.nlu_url = nlu_url
        self.nlu_command = nlu_command
        self.nlu_train_url = nlu_train_url
        self.nlu_train_command = nlu_train_command
        self.nlu_used = self.nlu_url or self.nlu_command
        self.nlu_train_used = self.nlu_train_url or self.nlu_train_command

        # Text to speech
        self.tts_url = tts_url
        self.tts_used = self.tts_url

        # Wake word detection
        self.wake_command = wake_command
        self.wake_enabled = True
        self.wake_proc: typing.Optional[subprocess.Popen] = None
        self.wake_sample_rate = wake_sample_rate
        self.wake_sample_width = wake_sample_width
        self.wake_channels = wake_channels
        self.wake_used = self.wake_command
        self.wake_disabled_reasons: typing.Set[str] = set()

        # Intent handling
        self.handle_url = handle_url
        self.handle_command = handle_command
        self.handle_enabled = True
        self.handle_used = self.handle_url or self.handle_command

        self.word_transform = word_transform

        # SSL
        self.ssl_context = ssl.SSLContext()
        if certfile:
            _LOGGER.debug("Using SSL with certfile=%s, keyfile=%s", certfile, keyfile)
            self.ssl_context.load_cert_chain(certfile, keyfile)

        # Async HTTP
        self._http_session: typing.Optional[aiohttp.ClientSession] = None

        # No timeout
        def default_recorder():
            return WebRtcVadRecorder(
                max_seconds=None,
                vad_mode=vad_mode,
                skip_seconds=skip_seconds,
                min_seconds=min_seconds,
                speech_seconds=speech_seconds,
                silence_seconds=silence_seconds,
                before_seconds=before_seconds,
            )

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

        # session_id -> AsrSession
        self.asr_sessions: typing.Dict[typing.Optional[str], AsrSession] = {}

        self.first_audio: bool = True

        # Start up
        if self.wake_command:
            self.start_wake_command()

        # Webhooks
        self.subscribe_topics(*self.webhook_topics)

        # Wake
        if self.wake_used:
            self.subscribe(HotwordToggleOn, HotwordToggleOff)

        # ASR
        if self.asr_used:
            self.subscribe(
                AsrStartListening,
                AsrStopListening,
                AsrToggleOn,
                AsrToggleOff,
                AudioFrame,
                AudioSessionFrame,
            )

        if self.asr_train_used:
            self.subscribe(AsrTrain)

        # NLU
        if self.nlu_used:
            self.subscribe(NluQuery)

        if self.nlu_train_used:
            self.subscribe(NluTrain)

        # TTS
        if self.tts_used:
            self.subscribe(TtsSay)

        # Intent Handling
        if self.handle_used:
            self.subscribe(NluIntent, HandleToggleOn, HandleToggleOff)

    @property
    def http_session(self):
        """Get or create async HTTP session"""
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()

        return self._http_session

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
        try:
            input_text = query.input

            # Fix casing
            if self.word_transform:
                input_text = self.word_transform(input_text)

            if self.nlu_url:
                # Use remote server
                _LOGGER.debug(self.nlu_url)

                params = {}

                # Add intent filter
                if query.intent_filter:
                    params["intentFilter"] = ",".join(query.intent_filter)

                async with self.http_session.post(
                    self.nlu_url, data=input_text, params=params, ssl=self.ssl_context
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
                slots = [
                    Slot(
                        entity=e["entity"],
                        slot_name=e["entity"],
                        confidence=1,
                        value=e.get("value_details", {"value": ["value"]}),
                        raw_value=e.get("raw_value", e["value"]),
                        range=SlotRange(
                            start=e.get("start", 0),
                            end=e.get("end", 1),
                            raw_start=e.get("raw_start"),
                            raw_end=e.get("raw_end"),
                        ),
                    )
                    for e in intent_dict.get("entities", [])
                ]

                yield NluIntentParsed(
                    input=query.input,
                    id=query.id,
                    site_id=query.site_id,
                    session_id=query.session_id,
                    intent=Intent(
                        intent_name=intent_name,
                        confidence_score=intent_dict["intent"].get("confidence", 1.0),
                    ),
                    slots=slots,
                )

                yield (
                    NluIntent(
                        input=query.input,
                        id=query.id,
                        site_id=query.site_id,
                        session_id=query.session_id,
                        intent=Intent(
                            intent_name=intent_name,
                            confidence_score=intent_dict["intent"].get(
                                "confidence", 1.0
                            ),
                        ),
                        slots=slots,
                        asr_tokens=[NluIntent.make_asr_tokens(tokens)],
                        raw_input=query.input,
                        wakeword_id=query.wakeword_id,
                        lang=query.lang,
                    ),
                    {"intent_name": intent_name},
                )
            else:
                # Not recognized
                yield NluIntentNotRecognized(
                    input=query.input,
                    id=query.id,
                    site_id=query.site_id,
                    session_id=query.session_id,
                )
        except Exception as e:
            _LOGGER.exception("handle_query")
            yield NluError(
                error=repr(e),
                context=repr(query),
                site_id=query.site_id,
                session_id=query.session_id,
            )

    # -------------------------------------------------------------------------

    async def handle_say(
        self, say: TtsSay
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[AudioPlayBytes, TopicArgs], TtsSayFinished, TtsError]
    ]:
        """Do text to speech."""
        try:
            if self.tts_url:
                # Remote text to speech server
                _LOGGER.debug(self.tts_url)

                params = {"play": "false"}
                if say.lang:
                    # Add ?language=<lang> query parameter
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
                            {"site_id": say.site_id, "request_id": say.id},
                        )
                    else:
                        _LOGGER.error("Received empty response")
        except Exception as e:
            _LOGGER.exception("handle_say")
            yield TtsError(
                error=str(e),
                context=say.id,
                site_id=say.site_id,
                session_id=say.session_id,
            )
        finally:
            yield TtsSayFinished(
                id=say.id, site_id=say.site_id, session_id=say.session_id
            )

    # -------------------------------------------------------------------------

    async def handle_start_listening(
        self, start_listening: AsrStartListening
    ) -> typing.AsyncIterable[AsrError]:
        """Start ASR session."""
        _LOGGER.debug("<- %s", start_listening)

        try:
            session = AsrSession(
                start_listening=start_listening, recorder=self.make_recorder()
            )

            self.asr_sessions[start_listening.session_id] = session
            session.recorder.start()
        except Exception as e:
            _LOGGER.exception("handle_start_listening")
            yield AsrError(
                error=str(e),
                context="",
                site_id=start_listening.site_id,
                session_id=start_listening.session_id,
            )

    # -------------------------------------------------------------------------

    async def handle_audio_frame(
        self,
        wav_bytes: bytes,
        site_id: str = "default",
        session_id: typing.Optional[str] = None,
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
                if session_id is None:
                    # Add to every open session
                    target_sessions = list(self.asr_sessions.items())
                else:
                    # Add to single session
                    target_sessions = [(session_id, self.asr_sessions[session_id])]

                with io.BytesIO(wav_bytes) as in_io:
                    with wave.open(in_io) as in_wav:
                        # Get WAV details from first frame
                        sample_rate = in_wav.getframerate()
                        sample_width = in_wav.getsampwidth()
                        channels = in_wav.getnchannels()
                        audio_data = in_wav.readframes(in_wav.getnframes())

                # Add to target ASR sessions
                for target_id, session in target_sessions:
                    # Skip non-matching site_id
                    if session.start_listening.site_id != site_id:
                        continue

                    session.sample_rate = sample_rate
                    session.sample_width = sample_width
                    session.channels = channels
                    session.audio_data += audio_data

                    if session.start_listening.stop_on_silence:
                        # Detect silence (end of command)
                        audio_data = self.maybe_convert_wav(
                            wav_bytes,
                            self.recorder_sample_rate,
                            self.recorder_sample_width,
                            self.recorder_channels,
                        )
                        command = session.recorder.process_chunk(audio_data)
                        if command and (command.result == VoiceCommandResult.SUCCESS):
                            # Complete session
                            stop_listening = AsrStopListening(
                                site_id=site_id, session_id=target_id
                            )
                            async for message in self.handle_stop_listening(
                                stop_listening
                            ):
                                yield message

            if self.wake_enabled and (session_id is None) and self.wake_proc:
                # Convert and send to wake command
                audio_bytes = self.maybe_convert_wav(
                    wav_bytes,
                    self.wake_sample_rate,
                    self.wake_sample_width,
                    self.wake_channels,
                )
                assert self.wake_proc.stdin
                self.wake_proc.stdin.write(audio_bytes)
                if self.wake_proc.poll():
                    stdout, stderr = self.wake_proc.communicate()
                    if stderr:
                        _LOGGER.debug(stderr.decode())

                    wakeword_id = stdout.decode().strip()
                    _LOGGER.debug("Detected wake word %s", wakeword_id)
                    yield (
                        HotwordDetected(
                            model_id=wakeword_id,
                            model_version="",
                            model_type="personal",
                            current_sensitivity=1.0,
                            site_id=site_id,
                        ),
                        {"wakeword_id": wakeword_id},
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
            session = self.asr_sessions.pop(stop_listening.session_id, None)
            if session is None:
                _LOGGER.warning("Session not found for %s", stop_listening.session_id)
                return

            assert session.sample_rate is not None, "No sample rate"
            assert session.sample_width is not None, "No sample width"
            assert session.channels is not None, "No channels"

            if session.start_listening.stop_on_silence:
                # Use recorded voice command
                audio_data = session.recorder.stop()
            else:
                # Use entire audio
                audio_data = session.audio_data

            # Process entire WAV file
            wav_bytes = self.to_wav_bytes(
                audio_data, session.sample_rate, session.sample_width, session.channels
            )
            _LOGGER.debug("Received %s byte(s) of WAV data", len(wav_bytes))

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
                site_id=stop_listening.site_id,
                session_id=stop_listening.session_id,
                lang=session.start_listening.lang,
            )

            if session.start_listening.send_audio_captured:
                # Send audio data
                yield (
                    AsrAudioCaptured(wav_bytes=wav_bytes),
                    {
                        "site_id": stop_listening.site_id,
                        "session_id": stop_listening.session_id,
                    },
                )

        except Exception as e:
            _LOGGER.exception("handle_stop_listening")
            yield AsrError(
                error=str(e),
                context=f"url='{self.asr_url}', command='{self.asr_command}'",
                site_id=stop_listening.site_id,
                session_id=stop_listening.session_id,
            )

    # -------------------------------------------------------------------------

    async def handle_asr_train(
        self, train: AsrTrain, site_id: str = "default"
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[AsrTrainSuccess, TopicArgs], AsrError]
    ]:
        """Re-trains ASR system"""
        try:
            # Load gzipped graph pickle
            _LOGGER.debug("Loading %s", train.graph_path)
            with gzip.GzipFile(train.graph_path, mode="rb") as graph_gzip:
                intent_graph = nx.readwrite.gpickle.read_gpickle(graph_gzip)

            # Get JSON intent graph
            json_graph = rhasspynlu.graph_to_json(intent_graph)

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

                output, error = await proc.communicate(
                    json.dumps(json_graph, ensure_ascii=False).encode()
                )

                if output:
                    _LOGGER.debug(output.decode())

                if error:
                    _LOGGER.debug(error.decode())
            else:
                _LOGGER.warning("Can't train ASR system. No train URL or command.")

            # Report success
            yield (AsrTrainSuccess(id=train.id), {"site_id": site_id})
        except Exception as e:
            _LOGGER.exception("handle_asr_train")
            yield AsrError(
                error=str(e),
                context=f"url='{self.asr_train_url}', command='{self.asr_train_command}'",
                site_id=site_id,
                session_id=train.id,
            )

    # -------------------------------------------------------------------------

    async def handle_nlu_train(
        self, train: NluTrain, site_id: str = "default"
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[NluTrainSuccess, TopicArgs], NluError]
    ]:
        """Re-trains NLU system"""
        try:
            # Load gzipped graph pickle
            _LOGGER.debug("Loading %s", train.graph_path)
            with gzip.GzipFile(train.graph_path, mode="rb") as graph_gzip:
                intent_graph = nx.readwrite.gpickle.read_gpickle(graph_gzip)

            # Get JSON intent graph
            json_graph = rhasspynlu.graph_to_json(intent_graph)

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

                output, error = await proc.communicate(
                    json.dumps(json_graph, ensure_ascii=False).encode()
                )

                if output:
                    _LOGGER.debug(output.decode())

                if error:
                    _LOGGER.debug(error.decode())
            else:
                _LOGGER.warning("Can't train NLU system. No train URL or command.")

            # Report success
            yield (NluTrainSuccess(id=train.id), {"site_id": site_id})
        except Exception as e:
            _LOGGER.exception("handle_nlu_train")
            yield NluError(
                error=str(e),
                context=f"url='{self.nlu_train_url}', command='{self.nlu_train_command}'",
                site_id=site_id,
                session_id=train.id,
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

            # Add site_id
            intent_dict["site_id"] = intent.site_id

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
                intent_json = json.dumps(intent_dict, ensure_ascii=False)

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

                try:
                    response_dict = json.loads(output)

                    # Check for speech response
                    tts_text = response_dict.get("speech", {}).get("text", "")
                except json.JSONDecodeError as e:
                    if output:
                        # Only report error if non-empty output
                        _LOGGER.warning("Failed to parse output as JSON: %s", e)
                        _LOGGER.warning("Output: %s", output)
            else:
                _LOGGER.warning("Can't handle intent. No handle URL or command.")

            if tts_text:
                # Forward to TTS system
                yield TtsSay(
                    text=tts_text,
                    id=str(uuid4()),
                    site_id=intent.site_id,
                    session_id=intent.session_id,
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
                    # Parse and check site_id
                    json_payload = json.loads(payload)
                    site_id = json_payload.get("siteId", "default")
                    if not self.valid_site_id(site_id):
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

    async def on_message(
        self,
        message: Message,
        site_id: typing.Optional[str] = None,
        session_id: typing.Optional[str] = None,
        topic: typing.Optional[str] = None,
    ) -> GeneratorType:
        """Received message from MQTT broker."""
        if isinstance(message, AudioFrame):
            # Add to all active sessions
            assert site_id, "Missing site id"
            if self.first_audio:
                _LOGGER.debug("Receiving audio")
                self.first_audio = False

            async for frame_result in self.handle_audio_frame(
                message.wav_bytes, site_id=site_id
            ):
                yield frame_result
        elif isinstance(message, AudioSessionFrame):
            # Check site_id
            assert site_id and session_id, "Missing site id or session id"
            if session_id in self.asr_sessions:
                # Add to active session
                if self.first_audio:
                    _LOGGER.debug("Receiving audio")
                    self.first_audio = False

                async for session_frame_result in self.handle_audio_frame(
                    message.wav_bytes, site_id=site_id, session_id=session_id
                ):
                    yield session_frame_result
        elif isinstance(message, NluQuery):
            async for query_result in self.handle_query(message):
                yield query_result
        elif isinstance(message, TtsSay):
            async for say_result in self.handle_say(message):
                yield say_result
        elif isinstance(message, AsrStartListening):
            async for start_listening_result in self.handle_start_listening(message):
                yield start_listening_result
        elif isinstance(message, AsrStopListening):
            async for stop_result in self.handle_stop_listening(message):
                yield stop_result
        elif isinstance(message, AsrTrain):
            assert site_id, "Missing site id"
            async for asr_train_result in self.handle_asr_train(
                message, site_id=site_id
            ):
                yield asr_train_result
        elif isinstance(message, NluTrain):
            assert site_id, "Missing site id"
            async for nlu_train_result in self.handle_nlu_train(
                message, site_id=site_id
            ):
                yield nlu_train_result
        elif isinstance(message, NluIntent):
            async for intent_result in self.handle_intent(message):
                yield intent_result
        elif isinstance(message, AsrToggleOn):
            if message.reason == AsrToggleReason.UNKNOWN:
                # Always enable on unknown
                self.asr_disabled_reasons.clear()
            else:
                self.asr_disabled_reasons.discard(message.reason)

            if self.asr_disabled_reasons:
                _LOGGER.debug("Still disabled: %s", self.asr_disabled_reasons)
            else:
                self.asr_enabled = True
                _LOGGER.debug("ASR enabled")
        elif isinstance(message, AsrToggleOff):
            self.asr_enabled = False
            self.asr_disabled_reasons.add(message.reason)
            _LOGGER.debug("ASR disabled")
        elif isinstance(message, HotwordToggleOn):
            if message.reason == HotwordToggleReason.UNKNOWN:
                # Always enable on unknown
                self.wake_disabled_reasons.clear()
            else:
                self.wake_disabled_reasons.discard(message.reason)

            if self.wake_disabled_reasons:
                _LOGGER.debug("Still disabled: %s", self.wake_disabled_reasons)
            else:
                self.wake_enabled = True
                _LOGGER.debug("Wake word detection enabled")
        elif isinstance(message, HotwordToggleOff):
            self.wake_enabled = False
            self.wake_disabled_reasons.add(message.reason)
            _LOGGER.debug("Wake word detection disabled")
        elif isinstance(message, HandleToggleOn):
            self.handle_enabled = True
            _LOGGER.debug("Intent handling enabled")
        elif isinstance(message, HandleToggleOff):
            self.handle_enabled = False
            _LOGGER.debug("Intent handling disabled")
        else:
            _LOGGER.warning("Unexpected message: %s", message)

    async def on_raw_message(self, topic: str, payload: bytes):
        """Handle raw MQTT messages from broker."""
        # Webhooks
        if self.webhook_matcher:
            await self.handle_webhook(topic, payload)
