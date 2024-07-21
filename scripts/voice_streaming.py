import asyncio
import signal
from asyncio import Queue
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp
import click
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
    Microphone,
)
from deepgram.utils import verboselogs
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DEEPGRAM_API_KEY: str = "deepgram-api-key"
    BATCH_SIZE: int

    class Config:
        env_file = ".env"


@dataclass
class SentenceChunk:
    text: str
    created_at: datetime


class TranscriptCollector:
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.transcript_parts = []
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part: str):
        self.transcript_parts.append(
            SentenceChunk(text=part, created_at=datetime.now(tz=timezone.utc))
        )

    def length_check(self) -> bool:
        if len(self.transcript_parts) >= 1:
            return (
                datetime.now(tz=timezone.utc) - self.transcript_parts[0].created_at
            ).seconds > self.batch_size
        else:
            return False

    def get_full_transcript(self):
        transcript_text = "\n".join([part.text for part in self.transcript_parts])

        if self.transcript_parts:
            end_time = self.transcript_parts[-1].created_at.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            start_time = self.transcript_parts[0].created_at.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            # Handle the case where there are no transcript parts
            end_time = start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return transcript_text, (end_time, start_time)


async def get_transcript(
    api_key: str,
    sentence_queue: Queue,
):
    shutdown_event = asyncio.Event()

    def signal_handler():
        click.echo("\nShutting down gracefully...", err=True)
        shutdown_event.set()

    try:
        config = DeepgramClientOptions(
            options={"keepalive": "true"}, verbose=verboselogs.ERROR
        )
        deepgram: DeepgramClient = DeepgramClient(api_key=api_key, config=config)

        dg_connection = deepgram.listen.asyncwebsocket.v("1")

        messages = 0

        async def on_message(self, result, **kwargs):

            sentence = result.channel.alternatives[0].transcript

            if sentence:
                click.echo(
                    f"-{sentence}",
                )
                await sentence_queue.put(sentence)

        async def on_error(self, error, **kwargs):
            click.echo(f"\n\n{error}\n\n", err=True)

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=True,
        )

        await dg_connection.start(options)

        microphone = Microphone(dg_connection.send)
        microphone.start()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        click.echo("Listening... Press Ctrl+C to stop.")

        await shutdown_event.wait()

        click.echo("Stopping microphone...")
        microphone.finish()

        click.echo("Closing Deepgram connection...")
        await dg_connection.finish()

        click.echo("Shutdown complete.")

    except Exception as e:
        click.echo(f"An error occurred: {e}", err=True)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


async def process_queue(queue: Queue, batch_size: int, shutdown_event: asyncio.Event):
    transcript_collector = TranscriptCollector(batch_size=batch_size)
    timeout = aiohttp.ClientTimeout(total=0.5)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        while not shutdown_event.is_set():
            try:
                message = await asyncio.wait_for(queue.get(), timeout=0.1)
                if message:
                    transcript_collector.add_part(message)
            except asyncio.TimeoutError:
                pass

            if transcript_collector.length_check():
                click.echo(
                    f"Processing queue from {transcript_collector.transcript_parts[0].created_at.strftime('%Y-%m-%d %H:%M:%S')} to {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
                )
                full_sentence, _ = transcript_collector.get_full_transcript()
                transcript_collector.reset()
                try:
                    async with session.post(
                        url="http://localhost:8000/text-chunk/",
                        data={"text": full_sentence},
                    ) as response:
                        print(f"Request successful: {response.status == 200}")
                except asyncio.TimeoutError:
                    print("Request timed out after 1 second")
                except aiohttp.ClientError as e:
                    print(f"An error occurred during the request: {e}")

    click.echo("Queue processing complete")


async def main_async(api_key: str, batch_size: int):

    sentence_queue = Queue()
    shutdown_event = asyncio.Event()

    transcript_task = asyncio.create_task(
        get_transcript(
            api_key=api_key,
            sentence_queue=sentence_queue,
        )
    )
    process_task = asyncio.create_task(
        process_queue(
            queue=sentence_queue, batch_size=batch_size, shutdown_event=shutdown_event
        )
    )

    await transcript_task

    shutdown_event.set()

    await process_task


@click.command()
@click.option("--batch-size", help="size of the batch to send to the server")
def main(batch_size: int):
    """Run real-time transcription using Deepgram."""
    settings = Settings()
    settings.BATCH_SIZE = int(batch_size)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            main_async(settings.DEEPGRAM_API_KEY, settings.BATCH_SIZE)
        )
    finally:
        loop.close()


if __name__ == "__main__":
    main()
