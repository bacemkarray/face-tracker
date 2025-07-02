from .task_executor import TaskExecutor

class TaskInterface:
    def __init__(self):
        self.executor = TaskExecutor()

    def add_task(self, task_dict):
        return self.executor.add_task(task_dict)

    def step(self, face_center):
        return self.executor.step(face_center)