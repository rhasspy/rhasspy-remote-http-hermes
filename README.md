# Rhasspy Remote HTTP Hermes

[![Continous Integration](https://github.com/rhasspy/rhasspy-remote-http-hermes/workflows/Tests/badge.svg)](https://github.com/rhasspy/rhasspy-remote-http-hermes/actions)
[![GitHub license](https://img.shields.io/github/license/rhasspy/rhasspy-remote-http-hermes.svg)](https://github.com/rhasspy/rhasspy-remote-http-hermes/blob/master/LICENSE)

Implements `hermes/asr`, `hermes/nlu`, `hermes/tts`, and intent handling functionality from [Hermes protocol](https://docs.snips.ai/reference/hermes) using a remote Rhasspy server's HTTP API.

## Requirements

* Python 3.7
* A [remote Rhasspy server](https://rhasspy.readthedocs.io/en/latest/reference/#http-api)

## Installation

```bash
$ git clone https://github.com/rhasspy/rhasspy-remote-http-hermes
$ cd rhasspy-remote-http-hermes
$ ./configure
$ make
$ make install
```

## Running

```bash
$ bin/rhasspy-remote-http-hermes <ARGS>
```

## Command-Line Options

```
usage: rhasspy-remote-http-hermes [-h] [--asr-url ASR_URL]
                                  [--asr-command ASR_COMMAND]
                                  [--asr-train-url ASR_TRAIN_URL]
                                  [--asr-train-command ASR_TRAIN_COMMAND]
                                  [--nlu-url NLU_URL]
                                  [--nlu-command NLU_COMMAND]
                                  [--nlu-train-url NLU_TRAIN_URL]
                                  [--nlu-train-command NLU_TRAIN_COMMAND]
                                  [--tts-url TTS_URL]
                                  [--wake-command WAKE_COMMAND]
                                  [--wake-sample-rate WAKE_SAMPLE_RATE]
                                  [--wake-sample-width WAKE_SAMPLE_WIDTH]
                                  [--wake-channels WAKE_CHANNELS]
                                  [--handle-url HANDLE_URL]
                                  [--handle-command HANDLE_COMMAND]
                                  [--casing {upper,lower,ignore}]
                                  [--certfile CERTFILE] [--keyfile KEYFILE]
                                  [--webhook WEBHOOK WEBHOOK]
                                  [--voice-skip-seconds VOICE_SKIP_SECONDS]
                                  [--voice-min-seconds VOICE_MIN_SECONDS]
                                  [--voice-speech-seconds VOICE_SPEECH_SECONDS]
                                  [--voice-silence-seconds VOICE_SILENCE_SECONDS]
                                  [--voice-before-seconds VOICE_BEFORE_SECONDS]
                                  [--voice-sensitivity {1,2,3}] [--host HOST]
                                  [--port PORT] [--username USERNAME]
                                  [--password PASSWORD] [--tls]
                                  [--tls-ca-certs TLS_CA_CERTS]
                                  [--tls-certfile TLS_CERTFILE]
                                  [--tls-keyfile TLS_KEYFILE]
                                  [--tls-cert-reqs {CERT_REQUIRED,CERT_OPTIONAL,CERT_NONE}]
                                  [--tls-version TLS_VERSION]
                                  [--tls-ciphers TLS_CIPHERS]
                                  [--site-id SITE_ID] [--debug]
                                  [--log-format LOG_FORMAT]

optional arguments:
  -h, --help            show this help message and exit
  --asr-url ASR_URL     URL of remote speech to text server (e.g.,
                        http://localhost:12101/api/speech-to-text)
  --asr-command ASR_COMMAND
                        Command to execute for ASR (WAV to text)
  --asr-train-url ASR_TRAIN_URL
                        URL for training speech to text server (POST with
                        JSON)
  --asr-train-command ASR_TRAIN_COMMAND
                        Command to train ASR system (JSON to stdin)
  --nlu-url NLU_URL     URL of remote intent recognition server (e.g.,
                        http://localhost:12101/api/text-to-intent)
  --nlu-command NLU_COMMAND
                        Command to execute for NLU (text to intent)
  --nlu-train-url NLU_TRAIN_URL
                        URL for training intent recognition server (POST with
                        JSON)
  --nlu-train-command NLU_TRAIN_COMMAND
                        Command to train NLU system (JSON to stdin)
  --tts-url TTS_URL     URL of remote text to speech server (e.g.,
                        http://localhost:12101/api/text-to-speech)
  --wake-command WAKE_COMMAND
                        Command to execute for wake word detection (raw audio
                        to wakeword id)
  --wake-sample-rate WAKE_SAMPLE_RATE
                        Sample rate in hertz required by wake command
                        (default: 16000)
  --wake-sample-width WAKE_SAMPLE_WIDTH
                        Sample width in bytes required by wake command
                        (default: 2)
  --wake-channels WAKE_CHANNELS
                        Number of channels required by wake command (default:
                        1)
  --handle-url HANDLE_URL
                        URL of remote intent handling server
  --handle-command HANDLE_COMMAND
                        Command to execute for intent handling (JSON on stdin)
  --casing {upper,lower,ignore}
                        Case transformation for words (default: ignore)
  --certfile CERTFILE   SSL certificate file
  --keyfile KEYFILE     SSL private key file (optional)
  --webhook WEBHOOK WEBHOOK
                        Topic/URL pairs for webhook(s)
  --voice-skip-seconds VOICE_SKIP_SECONDS
                        Seconds of audio to skip before a voice command
  --voice-min-seconds VOICE_MIN_SECONDS
                        Minimum number of seconds for a voice command
  --voice-speech-seconds VOICE_SPEECH_SECONDS
                        Consecutive seconds of speech before start
  --voice-silence-seconds VOICE_SILENCE_SECONDS
                        Consecutive seconds of silence before stop
  --voice-before-seconds VOICE_BEFORE_SECONDS
                        Seconds to record before start
  --voice-sensitivity {1,2,3}
                        VAD sensitivity (1-3)
  --host HOST           MQTT host (default: localhost)
  --port PORT           MQTT port (default: 1883)
  --username USERNAME   MQTT username
  --password PASSWORD   MQTT password
  --tls                 Enable MQTT TLS
  --tls-ca-certs TLS_CA_CERTS
                        MQTT TLS Certificate Authority certificate files
  --tls-certfile TLS_CERTFILE
                        MQTT TLS certificate file (PEM)
  --tls-keyfile TLS_KEYFILE
                        MQTT TLS key file (PEM)
  --tls-cert-reqs {CERT_REQUIRED,CERT_OPTIONAL,CERT_NONE}
                        MQTT TLS certificate requirements (default:
                        CERT_REQUIRED)
  --tls-version TLS_VERSION
                        MQTT TLS version (default: highest)
  --tls-ciphers TLS_CIPHERS
                        MQTT TLS ciphers to use
  --site-id SITE_ID     Hermes site id(s) to listen for (default: all)
  --debug               Print DEBUG messages to the console
  --log-format LOG_FORMAT
                        Python logger format
```
