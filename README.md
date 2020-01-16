# Rhasspy Remote HTTP Hermes

Implements `hermes/asr`, `hermes/nlu`, `hermes/tts`, and intent handling functionality from [Hermes protocol](https://docs.snips.ai/reference/hermes) using a remote Rhasspy server's HTTP API.

## Running With Docker

```bash
docker run -it rhasspy/rhasspy-remote-http-hermes:<VERSION> <ARGS>
```

## Building From Source

Clone the repository and create the virtual environment:

```bash
git clone https://github.com/rhasspy/rhasspy-remote-http-hermes.git
cd rhasspy-remote-http-hermes
make venv
```

Run the `bin/rhasspy-remote-http-hermes` script to access the command-line interface:

```bash
bin/rhasspy-remote-http-hermes --help
```

## Building the Debian Package

Follow the instructions to build from source, then run:

```bash
source .venv/bin/activate
make debian
```

If successful, you'll find a `.deb` file in the `dist` directory that can be installed with `apt`.

## Building the Docker Image

Follow the instructions to build from source, then run:

```bash
source .venv/bin/activate
make docker
```

This will create a Docker image tagged `rhasspy/rhasspy-remote-http-hermes:<VERSION>` where `VERSION` comes from the file of the same name in the source root directory.

NOTE: If you add things to the Docker image, make sure to whitelist them in `.dockerignore`.

## Command-Line Options

```
usage: rhasspy-remote-http-hermes [-h] [--asr-url ASR_URL]
                                  [--asr-command ASR_COMMAND]
                                  [--nlu-url NLU_URL]
                                  [--nlu-command NLU_COMMAND]
                                  [--tts-url TTS_URL]
                                  [--tts-command TTS_COMMAND] [--host HOST]
                                  [--port PORT] [--siteId SITEID] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --asr-url ASR_URL     URL of remote speech to text server (e.g.,
                        http://localhost:12101/api/speech-to-text)
  --asr-command ASR_COMMAND
                        Command to execute for ASR (WAV to text)
  --nlu-url NLU_URL     URL of remote intent recognition server (e.g.,
                        http://localhost:12101/api/text-to-intent)
  --nlu-command NLU_COMMAND
                        Command to execute for NLU (text to intent)
  --tts-url TTS_URL     URL of remote text to speech server (e.g.,
                        http://localhost:12101/api/text-to-speech)
  --tts-command TTS_COMMAND
                        Command to execute for TTS (text to WAV)
  --host HOST           MQTT host (default: localhost)
  --port PORT           MQTT port (default: 1883)
  --siteId SITEID       Hermes siteId(s) to listen for (default: all)
  --debug               Print DEBUG messages to the console
```
