from .tasks import Scan, Track
from collections import deque


class TaskExecutor:
    """
    Manage a queue of Task instances.
    """
    def __init__(self):
        self.tasks = deque()

    def add_task(self, task_dict: dict):
        """
        Accepts a dict like {"mode": "scan", "duration": 5}.
        """
        t_type = task_dict.get("mode")
        if t_type == "search":
            duration = float(task_dict.get("duration", 5)) # default 5 seconds
            task = Scan(duration)
        elif t_type == "track":
            task = Track()
        else:
            raise ValueError(f"Unknown task type: {t_type}")
        self.tasks.append(task)
        return task.task_id

    def step(self, frame_input):
        """
        Run the current task: observe, reason, and pop when done.
        Returns the (x, y) goal for this timestep.
        """
        if not self.tasks:
            # No tasks: default center
            return

        current = self.tasks[0]
        current.observe(frame_input)
        goal = current.reason()
        if current.is_done():
            self.tasks.popleft()
        return goal