import os
import signal
import sys
import traceback
from typing import Union, Optional
import aiofiles
import asyncio
import uuid
import json
from nio import (
    AsyncClient,
    AsyncClientConfig,
    InviteMemberEvent,
    JoinError,
    KeyVerificationCancel,
    KeyVerificationEvent,
    DownloadError,
    KeyVerificationKey,
    KeyVerificationMac,
    KeyVerificationStart,
    LocalProtocolError,
    LoginResponse,
    MatrixRoom,
    MegolmEvent,
    RoomMessageAudio,
    RoomEncryptedAudio,
    ToDeviceError,
    crypto,
    EncryptionError,
    WhoamiError,
)
from nio.store.database import SqliteStore

from faster_whisper import WhisperModel

from log import getlogger
from send_message import send_room_message

logger = getlogger()


class Bot:
    def __init__(
        self,
        homeserver: str,
        user_id: str,
        device_id: str,
        room_id: Union[str, None] = None,
        password: Union[str, None] = None,
        access_token: Union[str, None] = None,
        device_name: Union[str, None] = None,
        import_keys_path: Optional[str] = None,
        import_keys_password: Optional[str] = None,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: str = "models",
    ):
        if homeserver is None or user_id is None or device_id is None:
            logger.warning("homeserver && user_id && device_id is required")
            sys.exit(1)

        if password is None and access_token is None:
            logger.warning("password or access_toekn is required")
            sys.exit(1)

        self.homeserver = homeserver
        self.user_id = user_id
        self.password = password
        self.access_token = access_token
        self.device_name = device_name if device_name is not None else "matrix-stt-bot"
        self.device_id = device_id
        self.room_id = room_id
        self.import_keys_path = import_keys_path
        self.import_keys_password = import_keys_password
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.download_root = download_root

        if model_size is None:
            self.model_size = "tiny"

        if device is None:
            self.device = "cpu"

        if compute_type is None:
            self.compute_type = "int8"

        if cpu_threads is None:
            self.cpu_threads = 0

        if num_workers is None:
            self.num_workers = 1

        if download_root is None:
            cwd = os.getcwd()
            self.download_root = os.path.join(cwd, "models")
            if not os.path.exists(self.download_root):
                os.mkdir(self.download_root)

        # initialize AsyncClient object
        self.store_path = os.getcwd()
        self.config = AsyncClientConfig(
            store=SqliteStore,
            store_name="db",
            store_sync_tokens=True,
            encryption_enabled=True,
        )
        self.client = AsyncClient(
            homeserver=self.homeserver,
            user=self.user_id,
            device_id=self.device_id,
            config=self.config,
            store_path=self.store_path,
        )

        if self.access_token is not None:
            self.client.access_token = self.access_token

        # setup event callbacks
        self.client.add_event_callback(
            self.message_callback,
            (
                RoomMessageAudio,
                RoomEncryptedAudio,
            ),
        )
        self.client.add_event_callback(self.decryption_failure, (MegolmEvent,))
        self.client.add_event_callback(self.invite_callback, (InviteMemberEvent,))
        self.client.add_to_device_callback(
            self.to_device_callback, (KeyVerificationEvent,)
        )

        # intialize whisper model
        self.model = WhisperModel(
            model_size_or_path=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            num_workers=self.num_workers,
            download_root=self.download_root,
        )

        # create output folder
        if not os.path.exists("output"):
            os.mkdir("output")

    async def close(self, task: asyncio.Task = None) -> None:
        await self.client.close()
        task.cancel()
        logger.info("Bot closed!")

        # message_callback event

    async def message_callback(
        self, room: MatrixRoom, event: Union[RoomMessageAudio, RoomEncryptedAudio]
    ) -> None:
        if self.room_id is None:
            room_id = room.room_id
        else:
            # if event room id does not match the room id in config, return
            if room.room_id != self.room_id:
                return
            room_id = self.room_id

        # reply event_id
        reply_to_event_id = event.event_id

        # sender_id
        sender_id = event.sender

        if isinstance(event, RoomMessageAudio) or isinstance(event, RoomEncryptedAudio):
            try:
                asyncio.create_task(
                    self.main_function(event, room_id, sender_id, reply_to_event_id)
                )
            except Exception as e:
                logger.error(e, exc_info=True)

    async def main_function(
        self,
        event: Union[RoomMessageAudio, RoomEncryptedAudio],
        room_id: str,
        sender_id: str,
        reply_to_event_id: str,
    ):
        media_type = None
        if isinstance(event, RoomMessageAudio):  # for audio event
            # construct filename
            ext = os.path.splitext(event.body)[-1]
            filename = os.path.join("output", str(uuid.uuid4()) + ext)

            mxc = event.url  # audio mxc
            # download unencrypted audio file
            resp = await self.download_mxc(mxc=mxc)
            if isinstance(resp, DownloadError):
                logger.error("Download of media file failed")
            else:
                media_data = resp.body
                media_type = resp.content_type

                async with aiofiles.open(filename, "wb") as f:
                    await f.write(media_data)
                    await f.close()

        elif isinstance(event, RoomEncryptedAudio):  # for encrypted audio event
            # construct filename
            ext = os.path.splitext(event.body)[-1]
            filename = os.path.join("output", str(uuid.uuid4()) + ext)

            mxc = event.url  # audio mxc
            # download encrypted audio file
            resp = await self.download_mxc(mxc=mxc)
            if isinstance(resp, DownloadError):
                logger.error("Download of media file failed")
            else:
                media_data = resp.body
                media_type = event.mimetype

                async with aiofiles.open(filename, "wb") as f:
                    await f.write(
                        crypto.attachments.decrypt_attachment(
                            media_data,
                            event.source["content"]["file"]["key"]["k"],
                            event.source["content"]["file"]["hashes"]["sha256"],
                            event.source["content"]["file"]["iv"],
                        )
                    )
                    await f.close()

        # Whatsapp audio messages are sent as audio/ogg.
        # Matrix sends its messages as audio/mp4 but the filename starts with
        # "recording".
        # Ignore the other formats so we don't try to decode random music.
        evt_filename = event.source["content"].get("filename", "")
        if media_type == "audio/ogg" or (
            media_type.startswith("audio/") and evt_filename.startswith("recording")
        ):
            # use whisper to transribe audio to text
            try:
                await self.client.room_typing(room_id)
                message = await asyncio.to_thread(self.transcribe, filename)
                await send_room_message(
                    client=self.client,
                    room_id=room_id,
                    reply_message=message,
                    sender_id=sender_id,
                    reply_to_event_id=reply_to_event_id,
                )

            except Exception as e:
                logger.error(e)

            # remove audio file
            logger.info("audio file removed")
            os.remove(filename)
        else:
            logger.warning(f"Ignoring unsupported media type {media_type}")

    # message_callback decryption_failure event
    async def decryption_failure(self, room: MatrixRoom, event: MegolmEvent) -> None:
        if not isinstance(event, MegolmEvent):
            return

        logger.error(
            f"Failed to decrypt message: {event.event_id} from {event.sender} \
            in {room.room_id}\n"
            + "Please make sure the bot current session is verified"
        )

    # invite_callback event
    async def invite_callback(self, room: MatrixRoom, event: InviteMemberEvent) -> None:
        """Handle an incoming invite event.
        If an invite is received, then join the room specified in the invite.
        code copied from: https://github.com/8go/matrix-eno-bot/blob/ad037e02bd2960941109e9526c1033dd157bb212/callbacks.py#L104
        """
        logger.debug(f"Got invite to {room.room_id} from {event.sender}.")
        # Attempt to join 3 times before giving up
        for attempt in range(3):
            result = await self.client.join(room.room_id)
            if type(result) == JoinError:
                logger.error(
                    f"Error joining room {room.room_id} (attempt %d): %s",
                    attempt,
                    result.message,
                )
            else:
                break
        else:
            logger.error("Unable to join room: %s", room.room_id)

        # Successfully joined room
        logger.info(f"Joined {room.room_id}")

    # to_device_callback event
    async def to_device_callback(self, event: KeyVerificationEvent) -> None:
        """Handle events sent to device.

        Specifically this will perform Emoji verification.
        It will accept an incoming Emoji verification requests
        and follow the verification protocol.
        code copied from: https://github.com/8go/matrix-eno-bot/blob/ad037e02bd2960941109e9526c1033dd157bb212/callbacks.py#L127
        """
        try:
            client = self.client
            logger.debug(
                f"Device Event of type {type(event)} received in " "to_device_cb()."
            )

            if isinstance(event, KeyVerificationStart):  # first step
                """first step: receive KeyVerificationStart
                KeyVerificationStart(
                    source={'content':
                            {'method': 'm.sas.v1',
                             'from_device': 'DEVICEIDXY',
                             'key_agreement_protocols':
                                ['curve25519-hkdf-sha256', 'curve25519'],
                             'hashes': ['sha256'],
                             'message_authentication_codes':
                                ['hkdf-hmac-sha256', 'hmac-sha256'],
                             'short_authentication_string':
                                ['decimal', 'emoji'],
                             'transaction_id': 'SomeTxId'
                             },
                            'type': 'm.key.verification.start',
                            'sender': '@user2:example.org'
                            },
                    sender='@user2:example.org',
                    transaction_id='SomeTxId',
                    from_device='DEVICEIDXY',
                    method='m.sas.v1',
                    key_agreement_protocols=[
                        'curve25519-hkdf-sha256', 'curve25519'],
                    hashes=['sha256'],
                    message_authentication_codes=[
                        'hkdf-hmac-sha256', 'hmac-sha256'],
                    short_authentication_string=['decimal', 'emoji'])
                """

                if "emoji" not in event.short_authentication_string:
                    estr = (
                        "Other device does not support emoji verification "
                        f"{event.short_authentication_string}. Aborting."
                    )
                    print(estr)
                    logger.info(estr)
                    return
                resp = await client.accept_key_verification(event.transaction_id)
                if isinstance(resp, ToDeviceError):
                    estr = f"accept_key_verification() failed with {resp}"
                    print(estr)
                    logger.info(estr)

                sas = client.key_verifications[event.transaction_id]

                todevice_msg = sas.share_key()
                resp = await client.to_device(todevice_msg)
                if isinstance(resp, ToDeviceError):
                    estr = f"to_device() failed with {resp}"
                    print(estr)
                    logger.info(estr)

            elif isinstance(event, KeyVerificationCancel):  # anytime
                """at any time: receive KeyVerificationCancel
                KeyVerificationCancel(source={
                    'content': {'code': 'm.mismatched_sas',
                                'reason': 'Mismatched authentication string',
                                'transaction_id': 'SomeTxId'},
                    'type': 'm.key.verification.cancel',
                    'sender': '@user2:example.org'},
                    sender='@user2:example.org',
                    transaction_id='SomeTxId',
                    code='m.mismatched_sas',
                    reason='Mismatched short authentication string')
                """

                # There is no need to issue a
                # client.cancel_key_verification(tx_id, reject=False)
                # here. The SAS flow is already cancelled.
                # We only need to inform the user.
                estr = (
                    f"Verification has been cancelled by {event.sender} "
                    f'for reason "{event.reason}".'
                )
                print(estr)
                logger.info(estr)

            elif isinstance(event, KeyVerificationKey):  # second step
                """Second step is to receive KeyVerificationKey
                KeyVerificationKey(
                    source={'content': {
                            'key': 'SomeCryptoKey',
                            'transaction_id': 'SomeTxId'},
                        'type': 'm.key.verification.key',
                        'sender': '@user2:example.org'
                    },
                    sender='@user2:example.org',
                    transaction_id='SomeTxId',
                    key='SomeCryptoKey')
                """
                sas = client.key_verifications[event.transaction_id]

                print(f"{sas.get_emoji()}")
                # don't log the emojis

                # The bot process must run in forground with a screen and
                # keyboard so that user can accept/reject via keyboard.
                # For emoji verification bot must not run as service or
                # in background.
                # yn = input("Do the emojis match? (Y/N) (C for Cancel) ")
                # automatic match, so we use y
                yn = "y"
                if yn.lower() == "y":
                    estr = (
                        "Match! The verification for this " "device will be accepted."
                    )
                    print(estr)
                    logger.info(estr)
                    resp = await client.confirm_short_auth_string(event.transaction_id)
                    if isinstance(resp, ToDeviceError):
                        estr = "confirm_short_auth_string() " f"failed with {resp}"
                        print(estr)
                        logger.info(estr)
                elif yn.lower() == "n":  # no, don't match, reject
                    estr = (
                        "No match! Device will NOT be verified "
                        "by rejecting verification."
                    )
                    print(estr)
                    logger.info(estr)
                    resp = await client.cancel_key_verification(
                        event.transaction_id, reject=True
                    )
                    if isinstance(resp, ToDeviceError):
                        estr = f"cancel_key_verification failed with {resp}"
                        print(estr)
                        logger.info(estr)
                else:  # C or anything for cancel
                    estr = "Cancelled by user! Verification will be " "cancelled."
                    print(estr)
                    logger.info(estr)
                    resp = await client.cancel_key_verification(
                        event.transaction_id, reject=False
                    )
                    if isinstance(resp, ToDeviceError):
                        estr = f"cancel_key_verification failed with {resp}"
                        print(estr)
                        logger.info(estr)

            elif isinstance(event, KeyVerificationMac):  # third step
                """Third step is to receive KeyVerificationMac
                KeyVerificationMac(
                    source={'content': {
                        'mac': {'ed25519:DEVICEIDXY': 'SomeKey1',
                                'ed25519:SomeKey2': 'SomeKey3'},
                        'keys': 'SomeCryptoKey4',
                        'transaction_id': 'SomeTxId'},
                        'type': 'm.key.verification.mac',
                        'sender': '@user2:example.org'},
                    sender='@user2:example.org',
                    transaction_id='SomeTxId',
                    mac={'ed25519:DEVICEIDXY': 'SomeKey1',
                         'ed25519:SomeKey2': 'SomeKey3'},
                    keys='SomeCryptoKey4')
                """
                sas = client.key_verifications[event.transaction_id]
                try:
                    todevice_msg = sas.get_mac()
                except LocalProtocolError as e:
                    # e.g. it might have been cancelled by ourselves
                    estr = (
                        f"Cancelled or protocol error: Reason: {e}.\n"
                        f"Verification with {event.sender} not concluded. "
                        "Try again?"
                    )
                    print(estr)
                    logger.info(estr)
                else:
                    resp = await client.to_device(todevice_msg)
                    if isinstance(resp, ToDeviceError):
                        estr = f"to_device failed with {resp}"
                        print(estr)
                        logger.info(estr)
                    estr = (
                        f"sas.we_started_it = {sas.we_started_it}\n"
                        f"sas.sas_accepted = {sas.sas_accepted}\n"
                        f"sas.canceled = {sas.canceled}\n"
                        f"sas.timed_out = {sas.timed_out}\n"
                        f"sas.verified = {sas.verified}\n"
                        f"sas.verified_devices = {sas.verified_devices}\n"
                    )
                    print(estr)
                    logger.info(estr)
                    estr = (
                        "Emoji verification was successful!\n"
                        "Initiate another Emoji verification from "
                        "another device or room if desired. "
                        "Or if done verifying, hit Control-C to stop the "
                        "bot in order to restart it as a service or to "
                        "run it in the background."
                    )
                    print(estr)
                    logger.info(estr)
            else:
                estr = (
                    f"Received unexpected event type {type(event)}. "
                    f"Event is {event}. Event will be ignored."
                )
                print(estr)
                logger.info(estr)
        except BaseException:
            estr = traceback.format_exc()
            print(estr)
            logger.info(estr)

    # bot login
    async def login(self) -> None:
        if self.access_token is not None:
            self.client.restore_login(
                user_id=self.user_id,
                device_id=self.device_id,
                access_token=self.access_token,
            )
            try:
                resp = await self.client.whoami()
            except Exception as e:
                await self.client.close()
                logger.error(e, exc_info=True)
                sys.exit(1)
            if isinstance(resp, WhoamiError):
                logger.error(
                    f"Login Failed with {resp}, please check your access_token"
                )
                sys.exit(1)
            logger.info("Successfully login via access_token")

        else:
            try:
                resp = await self.client.login(
                    password=self.password, device_name=self.device_name
                )
                if not isinstance(resp, LoginResponse):
                    logger.error("Login Failed")
                    print(f"Login Failed: {resp}")
                    sys.exit(1)
                logger.info("Successfully login via password")
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)

    # sync messages in the room
    async def sync_forever(self, timeout=30000, full_state=True) -> None:
        await self.client.sync_forever(timeout=timeout, full_state=full_state)

    # download mxc
    async def download_mxc(self, mxc: str, filename: Optional[str] = None):
        response = await self.client.download(mxc=mxc, filename=filename)
        logger.info(f"download_mxc response: {response}")
        return response

    # import keys
    async def import_keys(self):
        resp = await self.client.import_keys(
            self.import_keys_path, self.import_keys_password
        )
        if isinstance(resp, EncryptionError):
            logger.error(f"import_keys failed with {resp}")
        else:
            logger.info(
                "import_keys success, you can remove import_keys configuration!"
            )

    # whisper function
    def transcribe(self, filename: str) -> str:
        logger.info("Start transcribe!")
        segments, _ = self.model.transcribe(filename, vad_filter=True)
        message = ""
        for segment in segments:
            message += segment.text

        return message


async def main():
    need_import_keys = False
    if os.path.exists("config.json"):
        fp = open("config.json", "r", encoding="utf-8")
        config = json.load(fp)

        bot = Bot(
            homeserver=config.get("homeserver"),
            user_id=config.get("user_id"),
            password=config.get("password"),
            device_id=config.get("device_id"),
            room_id=config.get("room_id"),
            access_token=config.get("access_token"),
            import_keys_path=config.get("import_keys_path"),
            import_keys_password=config.get("import_keys_password"),
            model_size=config.get("model_size"),
            device=config.get("device"),
            compute_type=config.get("compute_type"),
            cpu_threads=config.get("cpu_threads"),
            num_workers=config.get("num_workers"),
            download_root=config.get("download_root"),
        )

        if (
            config.get("import_keys_path")
            and config.get("import_keys_password") is not None
        ):
            need_import_keys = True

    else:
        bot = Bot(
            homeserver=os.environ.get("HOMESERVER"),
            user_id=os.environ.get("USER_ID"),
            password=os.environ.get("PASSWORD"),
            device_id=os.environ.get("DEVICE_ID"),
            room_id=os.environ.get("ROOM_ID"),
            access_token=os.environ.get("ACCESS_TOKEN"),
            import_keys_path=os.environ.get("IMPORT_KEYS_PATH"),
            import_keys_password=os.environ.get("IMPORT_KEYS_PASSWORD"),
            model_size=os.environ.get("MODEL_SIZE"),
            device=os.environ.get("DEVICE"),
            compute_type=os.environ.get("COMPUTE_TYPE"),
            cpu_threads=int(os.environ.get("CPU_THREADS", 0)),
            num_workers=int(os.environ.get("NUM_WORKERS", 1)),
            download_root=os.environ.get("DOWNLOAD_ROOT"),
        )
        if (
            os.environ.get("IMPORT_KEYS_PATH")
            and os.environ.get("IMPORT_KEYS_PASSWORD") is not None
        ):
            need_import_keys = True

    await bot.login()
    if need_import_keys:
        logger.info("start import_keys process, this may take a while...")
        await bot.import_keys()

    sync_task = asyncio.create_task(bot.sync_forever())

    # handle signal interrupt
    loop = asyncio.get_running_loop()
    for signame in (
        "SIGINT",
        "SIGTERM",
    ):
        loop.add_signal_handler(
            getattr(signal, signame), lambda: asyncio.create_task(bot.close(sync_task))
        )

    if bot.client.should_upload_keys:
        await bot.client.keys_upload()
    await sync_task


if __name__ == "__main__":
    logger.info("Bot started!")
    asyncio.run(main())
