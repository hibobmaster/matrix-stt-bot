from nio import AsyncClient


async def send_room_message(
    client: AsyncClient,
    room_id: str,
    reply_message: str,
    sender_id: str = "",
    reply_to_event_id: str = "",
) -> None:
    NORMAL_BODY = content = {
        "msgtype": "m.text",
        "body": reply_message,
    }
    if reply_to_event_id == "":
        content = NORMAL_BODY
    else:
        body = r"> <" + sender_id + r"> sent an audio file.\n\n" + reply_message
        format = r"org.matrix.custom.html"
        formatted_body = (
            r'<mx-reply><blockquote><a href="https://matrix.to/#/'
            + room_id
            + r"/"
            + reply_to_event_id
            + r'">In reply to</a> <a href="https://matrix.to/#/'
            + sender_id
            + r'">'
            + sender_id
            + r"</a><br>sent an audio file.</blockquote></mx-reply>"
            + reply_message
        )

        content = {
            "msgtype": "m.text",
            "body": body,
            "format": format,
            "formatted_body": formatted_body,
            "m.relates_to": {"m.in_reply_to": {"event_id": reply_to_event_id}},
        }
    await client.room_send(
        room_id,
        message_type="m.room.message",
        content=content,
        ignore_unverified_devices=True,
    )
    await client.room_typing(room_id, typing_state=False)
