# Introduction
This is a simple matrix bot that transcribes your voice to text message using faster-whisper, a reimplementation of OpenAI's Whisper model using CTranslate2.

## Feature

1. Liberate your hands: support automatically speech to text transcribtion
2. Support E2EE Room
3. Self host your service without privacy problem

## Installation and Setup

1. Edit `config.json` or `.env` with proper values
2. Edit `compose.yaml`
3. Launch the container

Here is a guide to make bot works on E2E encrypted room.

For explainations and complete parameter list see: https://github.com/hibobmaster/matrix-stt-bot/wiki

1. Create `config.json`

Tips: set a non-exist `room_id` at the first time to prevent bot handling historical message which may mess your room up.
```json
{
    "homeserver": "https://matrix.org",
    "user_id": "@xxxx:matrix.org",
    "password": "xxxxxxxxxx",
    "device_id": "GMIAZSVFF",
    "room_id": "!xxxxxxxx:xxx.xxx.xxx",
    "model_size": "base",
    "import_keys_path": "element-keys.txt",
    "import_keys_password": "xxxxxxxxxxxx"
}
```
2. Create `compose.yaml`

```yaml
services:
  app:
    image: ghcr.io/hibobmaster/matrix-stt-bot:latest
    container_name: matrix-stt-bot
    restart: always
    # build:
    #   context: .
    #   dockerfile: ./Dockerfile
    # env_file:
    #   - .env
    volumes:
      # use env file or config.json
      - ./config.json:/app/config.json
      # use touch to create an empty file stt_db, for persist database only
      - ./stt_db:/app/db
      # import_keys path
      - ./element-keys.txt:/app/element-keys.txt
      # store whisper models that program will download
      - ./models:/app/models
    networks:
      - matrix_network

networks:
  matrix_network:
```
Get your E2E room keys here:
![e2e-room-keys](https://i.imgur.com/WTKlXob.jpg)
Notice: If you deploy [matrix_chatgpt_bot](https://github.com/hibobmaster/matrix_chatgpt_bot) along with this project, remember do not use the same database name.

3. Launch for the first time
```sh
docker compose up
```
You will get notice: `INFO - start import_keys process`

After `INFO - import_keys success, please remove import_keys configuration!!!`

Wait a second, to see if `stt_db` has finished syncing_progress (The space occupied is about 100kb and above?)

Then `Ctrl+C` stop the container

4. Edit `config.json` again

Remove `import_keys_path` and `import_keys_password` options

Set a correct `room_id` or remove it if you hope the bot to work in the rooms it is in.

Tips: if bot exits because `RuntimeError: Unable to open file 'model.bin' in model`, try `rm -r models` then relaunch the container.

5. Finally

Launch the container
```sh
docker compose up -d
```

## Demo
![demo1](https://i.imgur.com/vntImys.png)
![demo2](https://i.imgur.com/VkOOVZA.png)

## Thanks
1. https://github.com/guillaumekln/faster-whisper
2. https://github.com/poljar/matrix-nio
3. https://github.com/8go/matrix-commander
<a href="https://jb.gg/OpenSourceSupport" target="_blank">
<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.png" alt="JetBrains Logo (Main) logo." width="200" height="200">
</a>
