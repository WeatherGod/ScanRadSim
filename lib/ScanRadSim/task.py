

class Task(object) :
    def __init__(self, updatePeriod, timeToComplete, name) :
        self.U = updatePeriod
        self.T = timeToComplete
        self.name = name
