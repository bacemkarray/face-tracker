import redis
import json
import threading



class RedisHelper:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        A thin wrapper around redis-py for pub/sub with handler dispatch.
        """
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.pubsub = self.client.pubsub()
        self._handlers: dict[str, callable] = {}
        self._thread: threading.Thread | None = None

    def publish(self, channel: str, message: dict):
        """
        Publish a JSON-serializable dict to a channel.
        """
        payload = json.dumps(message)
        self.client.publish(channel, payload)

    def subscribe(self, channel: str, handler: callable):
        """
        Register a handler for messages on `channel`. Handler gets the parsed dict.
        """
        self._handlers[channel] = handler
        self.pubsub.subscribe(channel)

    def _listen_loop(self):
        for msg in self.pubsub.listen():
            if msg["type"] != "message":
                continue
            chan: str = msg["channel"]
            try:
                data = json.loads(msg["data"])
            except json.JSONDecodeError:
                print(f"[RedisHelper] Invalid JSON on {chan}: {msg['data']}")
                continue

            handler = self._handlers.get(chan)
            if handler:
                try:
                    handler(data)
                except Exception as e:
                    print(f"[RedisHelper] Error in handler for {chan}: {e}")

    def start(self):
        """
        Kick off the background thread to listen for all subscribed channels.
        """
        if self._thread is None:
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()
