from collections import deque
from datetime import datetime

EVENT_LOGS = deque(maxlen=100)  # keep last 100 events

def add_event(event_type, message, severity="normal"):
    EVENT_LOGS.appendleft({
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": event_type,
        "message": message,
        "severity": severity
    })
