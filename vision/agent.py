import tasks

class FaceAgent:
    def __init__(self):
        self.executor = tasks.TaskExecutor()

    def add_task(self, task_dict):
        self.executor.add_task(task_dict)

    def step(self, face_center):
        return self.executor.step(face_center)